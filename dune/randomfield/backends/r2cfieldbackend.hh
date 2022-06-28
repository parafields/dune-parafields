#ifndef DUNE_RANDOMFIELD_R2CFIELDBACKEND_HH
#define DUNE_RANDOMFIELD_R2CFIELDBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Field backend that uses real-input discrete Fourier transform (R2C)
     *
     * This field backend is an implementation of the extended field that generates
     * one sample at a time and saves memory in the process. The classical circulant
     * embedding method transforms complex-valued noise into two random fields. This
     * backend creates the Hermitian part of the noise instead, which is the part
     * that ends up as the real part of the generated field. The anti-Hermitian part,
     * which would end up as the imaginary part, is never generated, and so half the
     * memory can be saved at little extra cost in runtime.
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

        /**
         * @brief Constructor
         *
         * Imports FFTW wisdom if configured to do so.
         *
         * @param traits_ traits object with parameters and communication
         */
        R2CFieldBackend(const std::shared_ptr<Traits>& traits_)
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

        /**
         * @brief Destructor
         *
         * Cleans up allocated arrays and FFTW plans. Exports FFTW
         * wisdom if configured to do so.
         */
        ~R2CFieldBackend()
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
         *
         * This function is has to be called after the creation of
         * the random field object or its refinement. It updates
         * parameters like the number of cells per dimension.
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
         *
         * This is the number of extended field entries that are stored,
         * in the frequency domain. The backend produces a single real-valued
         * random field from Hermitian complex noise, which means that
         * half the entries of the noise are redundant and are not actually
         * stored.
         *
         * @return number of local degrees of freedom
         */
        Index localFieldSize() const
        {
          return localR2CComplexDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         *
         * This is the number of cells per dimension of the
         * extended domain, or the number of cells per dimension
         * for the local part of the extended domain in the case of
         * parallel data distribution. This is in the frequency
         * domain, so one of the dimensions is cut in half to account
         * for the redundancy due to the Hermitian property.
         *
         * @return tuple of local cells per dimension
         */
        const Indices& localFieldCells() const
        {
          return localR2CComplexCells;
        }

        /**
         * @brief Reserve memory before storing any field entries
         *
         * Explicitly request the field backend to reserve storage for the
         * multidimensional array. This ensures that the backend doesn't
         * waste memory when it won't be used.
         */
        void allocate()
        {
          if (fieldData == nullptr)
            fieldData = FFTW<RF>::alloc_complex(allocLocal);
        }

        /**
         * @brief Switch last two dimensions (for transposed transforms)
         *
         * This function switches the last two dimensions, which is needed
         * for FFTW transposed transforms, where the Fourier transform of
         * the matrix is stored transposed to eliminate the final transpose
         * step. Is automatically called by the transform methods, but may
         * be needed when a newly created backend should be constructed
         * directly in frequency space.
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
         *
         * Perform a forward Fourier transform, mapping from the original
         * domain to the frequency domain. Uses a single FFTW real-to-complex
         * DFT transform: the input is a multidimensional array of real numbers,
         * the output a Hermitian array of complex numbers, with half the
         * data not stored because of redundancy.
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
         *
         * Perform a backward Fourier transform, mapping from the frequency
         * domain back to the original domain. Uses a single FFTW complex-to-real
         * DFT transform: the input is a multidimensional Hermitian array of
         * complex numbers, with half the data not stored because of redundancy,
         * and the output is an array of real numbers.
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
         *
         * This backend produces a single real-valued field.
         *
         * @return false
         */
        bool hasSpareField() const
        {
          return false;
        }

        /**
         * @brief Set entry based on pair of random numbers
         *
         * This function takes two normally distributed random numbers
         * and stores them in the backend, after multiplying them with
         * a scalar value, which is the square root of one of the
         * eigenvalues of the extended covariance matrix. The two values
         * are interpreted as the Hermitian part of the noise of the
         * original circulant embedding method, and therefore have to be
         * scaled with the inverse square root of two. Together with the
         * anti-Hermitian half they would then generate the original noise.
         * Cells that lie at the beginning or exactly in the middle of the
         * array in each dimension need special treatment, since they
         * are their own mirror image when reflecting at the origin.
         *
         * @param index  index of extended field cell to fill
         * @param lambda square root of covariance matrix eigenvalue
         * @param rand1  normally distributed random number
         * @param rand2  second normally distributed random number
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
         *
         * This function multiplies a complex entry of the array with
         * a scalar. This is used when a random field should be multiplied
         * with the covariance matrix, or its inverse, etc.
         *
         * @param index  index of extended field cell to scale
         * @param lambda scalar factor
         */
        void mult(Index index, RF lambda)
        {
          fieldData[index][0] *= lambda;
          fieldData[index][1] *= lambda;
        }

        /**
         * @brief Embed a random field in the extended domain
         *
         * This function maps a random field onto the extended domain,
         * filling any cells that are not part of the original domain
         * with zero values.
         *
         * @param field random field to embed in larger domain
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
         *
         * This function restricts an extended random field and cuts out the
         * part that lies on the original domain. The optional argument can
         * be used to select between the two fields that are stored in the
         * real and imaginary part of the extended random field.
         *
         * @param[out] field     random field to fill with restriction
         * @param      component dummy variable, backend produces single field
         *
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
         *
         * This function obtains the local offset and cells in the distributed
         * dimension as prescibed by FFTW.
         */
        void getR2CData()
        {
          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim-1; i++)
            n[i] = extendedCells[dim-1-i];
          n[dim-1] = extendedCells[0]/2+1;

          allocLocal = FFTW<RF>::mpi_local_size(dim, n, (*traits).comm, &localN0, &local0Start);
        }

        /**
         * @brief Calculate R2C cells from extended domain cells
         *
         * This function computes the number of cells per dimension for
         * the untransformed and transformed array. For the untransformed
         * array, this is simply the number of cells of the extended domain,
         * with some slight padding in the first dimension, and for the
         * transformed array, the first dimension is cut in half, because the
         * second half is redundant.
         */
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
