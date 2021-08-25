#ifndef DUNE_RANDOMFIELD_R2CFIELDBACKEND_HH
#define DUNE_RANDOMFIELD_R2CFIELDBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Field backend that uses real-input discrete Fourier transform (R2C)
     */
    template<typename Traits>
      class R2CFieldBackend
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        static_assert(dim != 1, "R2CFieldBackend requires dim > 1");

        const std::shared_ptr<Traits> traits;

        int rank, commSize;

        ptrdiff_t allocLocal, localN0, local0Start;

        Indices localCells;
        Index   localDomainSize;
        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Indices localExtendedOffset;
        Index   localExtendedDomainSize;

        Indices localR2CComplexCells;
        Index localR2CComplexDomainSize;
        Indices localR2CRealCells;
        Index localR2CRealDomainSize;

        mutable typename FFTW<RF>::complex* fieldData;
        mutable Indices indices;

        bool transposed;

        enum {toExtended, fromExtended};

        public:

        R2CFieldBackend<Traits>(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            fieldData(nullptr)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using R2CFieldBackend" << std::endl;

          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            if ((*traits).rank == 0)
              FFTW<RF>::import_wisdom_from_filename("wisdom-R2CField.ini");

            FFTW<RF>::mpi_broadcast_wisdom((*traits).comm);
          }
        }

        ~R2CFieldBackend<Traits>()
        {
          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            FFTW<RF>::mpi_gather_wisdom((*traits).comm);

            if ((*traits).rank == 0)
              FFTW<RF>::export_wisdom_to_filename("wisdom-R2CField.ini");
          }

          if (fieldData != nullptr)
          {
            FFTW<RF>::free(fieldData);
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
          localExtendedOffset     = (*traits).localExtendedOffset;
          localExtendedDomainSize = (*traits).localExtendedDomainSize;
          transposed              = (*traits).transposed;

          if (dim == 2 && transposed)
            DUNE_THROW(Dune::Exception,
                "R2CFieldBackend supports transposed output only for dim > 2");

          getR2CData();

          getR2CCells();

          if (fieldData != nullptr)
          {
            FFTW<RF>::free(fieldData);
            fieldData = nullptr;
          }
        }

        /**
         * @brief Number of extended field entries stored on this processor
         */
        Index localFieldSize() const
        {
          return localR2CComplexDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         */
        const Indices& localFieldCells() const
        {
          return localR2CComplexCells;
        }

        /**
         * @brief Reserve memory before storing any field entries
         */
        void allocate()
        {
          if (fieldData == nullptr)
            fieldData = FFTW<RF>::alloc_complex(allocLocal);
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

            getR2CCells();
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

          typename FFTW<RF>::plan plan_forward = FFTW<RF>::mpi_plan_dft_r2c(dim,n,(RF*)fieldData,
              fieldData,(*traits).comm,flags);

          if (plan_forward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create forward plan");

          FFTW<RF>::execute(plan_forward);
          FFTW<RF>::destroy_plan(plan_forward);

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

          typename FFTW<RF>::plan plan_backward = FFTW<RF>::mpi_plan_dft_c2r(dim,n,fieldData,
              (RF*)fieldData,(*traits).comm,flags);

          if (plan_backward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create backward plan");

          FFTW<RF>::execute(plan_backward);
          FFTW<RF>::destroy_plan(plan_backward);
        }

        /**
         * @brief Whether this kind of backend produces two fields at once
         */
        bool hasSpareField() const
        {
          return false;
        }

        /**
         * @brief Set entry based on pair of random numbers
         */
        void set(Index index, RF lambda, RF rand1, RF rand2)
        {
          static const RF sqrtTwo = std::sqrt(2.);

          Traits::indexToIndices(index,indices,localR2CComplexCells);

          bool allMultiple = true;
          for (unsigned int i = 0; i < dim; i++)
          {
            const Index globalIndex = indices[i] + localExtendedOffset[i];
            if ((2*globalIndex) % extendedCells[i] != 0)
              allMultiple = false;
          }

          if (allMultiple)
          {
            fieldData[index][0] = lambda * rand1;
            fieldData[index][1] = 0;
          }
          else
          {
            fieldData[index][0] = lambda/sqrtTwo * rand1;
            fieldData[index][1] = lambda/sqrtTwo * rand2;
          }
        }

        /**
         * @brief Multiply entry with given number
         */
        void mult(Index index, RF lambda)
        {
          fieldData[index][0] *= lambda;
          fieldData[index][1] *= lambda;
        }

        /**
         * @brief Embed a random field in the extended domain
         */
        void fieldToExtendedField(std::vector<RF>& field)
        {
          if (fieldData == nullptr)
            fieldData = FFTW<RF>::alloc_complex(allocLocal);

          for(Index i = 0; i < localR2CRealDomainSize; i++)
            ((RF*)fieldData)[i] = 0.;

          if (commSize == 1)
          {
            Indices indices;
            for (Index index = 0; index < localDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localCells);
              const Index extIndex = Traits::indicesToIndex(indices,localR2CRealCells);

              ((RF*)fieldData)[extIndex] = field[index];
            }
          }
          else
          {
            const int embeddingFactor = (*traits).embeddingFactor;
            MPI_Request request;

            MPI_Isend(&(field[0]), localDomainSize, mpiType<RF>,
                rank/embeddingFactor, toExtended, (*traits).comm, &request);

            if (rank*embeddingFactor < commSize)
            {
              std::vector<RF> localCopy(localDomainSize);
              Indices indices;

              Index receiveSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
              for (Index i = 0; i < receiveSize; i++)
              {
                MPI_Recv(&(localCopy[0]), localDomainSize, mpiType<RF>,
                    rank*embeddingFactor + i, toExtended, (*traits).comm, MPI_STATUS_IGNORE);

                for (Index index = 0; index < localDomainSize; index++)
                {
                  Traits::indexToIndices(index,indices,localCells);
                  const Index offset =  i * localR2CRealDomainSize/embeddingFactor;
                  const Index extIndex
                    = Traits::indicesToIndex(indices,localR2CRealCells) + offset;

                  ((RF*)fieldData)[extIndex] = localCopy[index];
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
          field.resize(localDomainSize);

          if (component != 0)
            DUNE_THROW(Dune::Exception,"tried to extract more than one field from R2CFieldBackend");

          if (commSize == 1)
          {
            Indices indices;
            for (Index index = 0; index < localDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localCells);
              const Index extIndex = Traits::indicesToIndex(indices,localR2CRealCells);

              field[index] = ((RF*)fieldData)[extIndex];
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
                  const Index offset =  i * localR2CRealDomainSize/embeddingFactor;
                  const Index extIndex = Traits::indicesToIndex(indices,localR2CRealCells);

                  localCopy[i][index] = ((RF*)fieldData)[extIndex + offset];
                }

                MPI_Isend(&(localCopy[i][0]), localDomainSize, mpiType<RF>,
                    rank*embeddingFactor + i, fromExtended, (*traits).comm, &(request[i]));
              }

              MPI_Recv(&(field[0]), localDomainSize, mpiType<RF>,
                  rank/embeddingFactor, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);

              MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);
            }
            else
            {
              MPI_Recv(&(field[0]), localDomainSize, mpiType<RF>,
                  rank/embeddingFactor, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);
            }
          }
        }

        private:

        /**
         * @brief Get the domain decomposition data of the Fourier transform
         */
        void getR2CData()
        {
          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim-1; i++)
            n[i] = extendedCells[dim-1-i];
          n[dim-1] = extendedCells[0]/2+1;

          allocLocal = FFTW<RF>::mpi_local_size(dim, n, (*traits).comm, &localN0, &local0Start);
        }

        void getR2CCells()
        {
          localR2CComplexCells = localExtendedCells;
          localR2CComplexCells[0] /= 2;
          localR2CComplexCells[0]++;
          localR2CComplexDomainSize = localExtendedDomainSize / localExtendedCells[0] * localR2CComplexCells[0];

          localR2CRealCells = localExtendedCells;
          localR2CRealCells[0] = 2 * (localExtendedCells[0]/2 + 1);
          localR2CRealDomainSize = localExtendedDomainSize / localExtendedCells[0] * localR2CRealCells[0];
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_R2CFIELDBACKEND_HH
