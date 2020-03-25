// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_DFTFIELDBACKEND_HH
#define	DUNE_RANDOMFIELD_DFTFIELDBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Extended field backend that uses discrete Fourier transform (DFT)
     */
    template<typename Traits>
      class DFTFieldBackend
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        const std::shared_ptr<Traits> traits;

        int rank, commSize;

        ptrdiff_t allocLocal, localN0, local0Start;

        Indices localCells;
        Index   localDomainSize;
        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Index   localExtendedDomainSize;

        mutable fftw_complex* fieldData;

        bool transposed;

        enum {toExtended, fromExtended};

        public:

        DFTFieldBackend<Traits>(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            fieldData(nullptr)
        {}

        ~DFTFieldBackend<Traits>()
        {
          if (fieldData != nullptr)
          {
            fftw_free(fieldData);
            fieldData = nullptr;
          }
        }

        /*
         * @brief Update internal data after creation or refinement
         */
        void update()
        {
          rank     = (*traits).rank;
          commSize = (*traits).commSize;

          localCells              = (*traits).localCells;
          localDomainSize         = (*traits).localDomainSize;
          extendedCells           = (*traits).extendedCells;
          extendedDomainSize      = (*traits).extendedDomainSize;
          localExtendedCells      = (*traits).localExtendedCells;
          localExtendedDomainSize = (*traits).localExtendedDomainSize;
          transposed              = (*traits).transposed;

          getDFTData();

          if (fieldData != nullptr)
          {
            fftw_free(fieldData);
            fieldData = nullptr;
          }
        }

        /**
         * @brief Number of extended field entries stored on this processor
         */
        Index localFieldSize() const
        {
          return localExtendedDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         */
        const Indices& localFieldCells() const
        {
          return localExtendedCells;
        }

        /**
         * @brief Reserve memory before storing any field entries
         */
        void allocate()
        {
          if (fieldData == nullptr)
            fieldData = fftw_alloc_complex(allocLocal);
        }

        /**
         * @brief Switch last two dimensions (for transposed transforms)
         */
        void transposeIfNeeded()
        {
          if (transposed)
          {
            std::swap(extendedCells[dim-1],extendedCells[dim-2]);
            localExtendedCells[dim-1] = extendedCells[dim-1] / commSize;
            localExtendedCells[dim-2] = extendedCells[dim-2];
          }
        }

        /**
         * @brief Transform into Fourier (i.e., frequency) space
         */
        void forwardTransform()
        {
          unsigned int flags;
          if ((*traits).config.template get<bool>("fftw.measure",false))
            flags = FFTW_MEASURE;
          else
            flags = FFTW_ESTIMATE;
          if (transposed)
            flags |= FFTW_MPI_TRANSPOSED_OUT;

          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            n[i] = extendedCells[dim-1-i];

          fftw_plan plan_forward = fftw_mpi_plan_dft(dim,n,fieldData,
              fieldData,(*traits).comm,FFTW_FORWARD,flags);

          if (plan_forward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create forward plan");

          fftw_execute(plan_forward);
          fftw_destroy_plan(plan_forward);

          for (Index i = 0; i < allocLocal; i++)
          {
            fieldData[i][0] /= extendedDomainSize;
            fieldData[i][1] /= extendedDomainSize;
          }

          transposeIfNeeded();
        }

        /**
         * @brief Transform from Fourier (i.e., frequency) space
         */
        void backwardTransform()
        {
          transposeIfNeeded();

          unsigned int flags;
          if ((*traits).config.template get<bool>("fftw.measure",false))
            flags = FFTW_MEASURE;
          else
            flags = FFTW_ESTIMATE;
          if (transposed)
            flags |= FFTW_MPI_TRANSPOSED_IN;

          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            n[i] = extendedCells[dim-1-i];

          fftw_plan plan_backward = fftw_mpi_plan_dft(dim,n,fieldData,
              fieldData,(*traits).comm,FFTW_BACKWARD,flags);

          if (plan_backward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create backward plan");

          fftw_execute(plan_backward);
          fftw_destroy_plan(plan_backward);
        }

        /**
         * @brief Whether this kind of backend produces two fields at once
         */
        bool hasSpareField() const
        {
          return true;
        }

        void set(Index index, RF lambda, RF rand1, RF rand2)
        {
          fieldData[index][0] = lambda * rand1;
          fieldData[index][1] = lambda * rand2;
        }

        void mult(Index i, RF lambda)
        {
          fieldData[i][0] *= lambda;
          fieldData[i][1] *= lambda;
        }

        /**
         * @brief Embed a random field in the extended domain
         */
        void fieldToExtendedField(std::vector<RF>& field) const
        {
          if (fieldData == nullptr)
            fieldData = fftw_alloc_complex(allocLocal);

          for(Index i = 0; i < localExtendedDomainSize; i++)
            fieldData[i][1] = 0.;

          if (commSize == 1)
          {
            Indices indices;
            for (Index index = 0; index < localDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localCells);
              const Index extIndex = Traits::indicesToIndex(indices,localExtendedCells);

              fieldData[extIndex][0] = field[index];
            }
          }
          else
          {
            const int embeddingFactor = (*traits).embeddingFactor;
            MPI_Request request;

            MPI_Isend(&(field[0]), localDomainSize, MPI_DOUBLE,
                rank/embeddingFactor, toExtended, (*traits).comm, &request);

            if (rank*embeddingFactor < commSize)
            {
              std::vector<RF> localCopy(localDomainSize);
              Indices indices;

              Index receiveSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
              for (Index i = 0; i < receiveSize; i++)
              {
                MPI_Recv(&(localCopy[0]), localDomainSize, MPI_DOUBLE,
                    rank*embeddingFactor + i, toExtended, (*traits).comm, MPI_STATUS_IGNORE);

                for (Index index = 0; index < localDomainSize; index++)
                {
                  Traits::indexToIndices(index,indices,localCells);
                  const Index offset =  i * localExtendedDomainSize/embeddingFactor;
                  const Index extIndex
                    = Traits::indicesToIndex(indices,localExtendedCells) + offset;

                  fieldData[extIndex][0] = localCopy[index];
                }
              }
            }

            MPI_Wait(&request,MPI_STATUS_IGNORE);
          }
        }

        /**
         * @brief Restrict an extended random field to the original domain
         */
        void extendedFieldToField(
            std::vector<RF>& field,
            unsigned int component = 0
            ) const
        {
          if (commSize == 1)
          {
            Indices indices;
            for (Index index = 0; index < localDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localCells);
              const Index extIndex = Traits::indicesToIndex(indices,localExtendedCells);

              field[index] = fieldData[extIndex][component];
            }
          }
          else
          {
            const int embeddingFactor = (*traits).embeddingFactor;

            if (rank*embeddingFactor < commSize)
            {
              std::vector<std::vector<RF>> localCopy;
              std::vector<MPI_Request>     request;

              unsigned int sendSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
              localCopy.resize(sendSize);
              request.resize(sendSize);
              Indices indices;

              for (unsigned int i = 0; i < sendSize; i++)
              {
                localCopy[i].resize(localDomainSize);
                for (Index index = 0; index < localDomainSize; index++)
                {
                  Traits::indexToIndices(index,indices,localCells);
                  const Index offset = i * localExtendedDomainSize/embeddingFactor;
                  const Index extIndex = Traits::indicesToIndex(indices,localExtendedCells);

                  localCopy[i][index] = fieldData[extIndex + offset][component];
                }

                MPI_Isend(&(localCopy[i][0]), localDomainSize, MPI_DOUBLE,
                    rank*embeddingFactor + i, fromExtended, (*traits).comm, &(request[i]));
              }

              MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE,
                  rank/embeddingFactor, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);

              MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);
            }
            else
            {
              MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE,
                  rank/embeddingFactor, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);
            }
          }
        }

        private:

        /**
         * @brief Get the domain decomposition data of the Fourier transform
         */
        void getDFTData()
        {
          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            n[i] = extendedCells[dim-1-i];

          if (dim == 1)
          {
            ptrdiff_t localN02, local0Start2;
            allocLocal = fftw_mpi_local_size_1d(n[0],(*traits).comm,FFTW_FORWARD,FFTW_ESTIMATE,
                &localN0,&local0Start,&localN02,&local0Start2);
            if (localN0 != localN02 || local0Start != local0Start2)
              DUNE_THROW(Dune::Exception,"1d size / offset results don't match");
          }
          else
            allocLocal = fftw_mpi_local_size(dim,n,(*traits).comm,&localN0,&local0Start);
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_DFTFIELDBACKEND_HH
