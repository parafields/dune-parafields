#ifndef DUNE_RANDOMFIELD_DFTMATRIXBACKEND_HH
#define DUNE_RANDOMFIELD_DFTMATRIXBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Matrix backend that uses discrete Fourier transform (DFT)
     *
     * This matrix backend implements the classical circulant embedding
     * method: the circulant extended covariance matrix is represented
     * as a scalar field on the extended domain, representing either
     * the first row / column of the matrix (on the original domain)
     * or the diagonal of its Fourier transform (in the frequency domain).
     * The backend stores one complex number per cell in the extended
     * domain, and uses the general discrete Fourier transform for its
     * operations.
     *
     * @tparam Traits traits class with data types and definitions
     */
    template<typename Traits>
      class DFTMatrixBackend
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        const std::shared_ptr<Traits> traits;

        int rank, commSize;

        ptrdiff_t allocLocal, localN0, local0Start;

        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Indices localExtendedOffset;
        Index   localExtendedDomainSize;

        mutable typename FFTW<RF>::complex* matrixData;

        bool transposed;

        public:

        /**
         * @brief Constructor
         *
         * Imports FFTW wisdom if configured to do so.
         *
         * @param traits_ traits object with parameters and communication
         */
        DFTMatrixBackend(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            matrixData(nullptr)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using DFTMatrixBackend" << std::endl;

          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            if ((*traits).rank == 0)
              FFTW<RF>::import_wisdom_from_filename("wisdom-DFTMatrix.ini");

            FFTW<RF>::mpi_broadcast_wisdom((*traits).comm);
          }
        }

        /**
         * @brief Destructor
         *
         * Cleans up allocated arrays and FFTW plans. Exports FFTW
         * wisdom if configured to do so.
         */
        ~DFTMatrixBackend()
        {
          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            FFTW<RF>::mpi_gather_wisdom((*traits).comm);

            if ((*traits).rank == 0)
              FFTW<RF>::export_wisdom_to_filename("wisdom-DFTMatrix.ini");
          }

          if (matrixData != nullptr)
          {
            FFTW<RF>::free(matrixData);
            matrixData = nullptr;
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

          extendedCells           = (*traits).extendedCells;
          extendedDomainSize      = (*traits).extendedDomainSize;
          localExtendedCells      = (*traits).localExtendedCells;
          localExtendedOffset     = (*traits).localExtendedOffset;
          localExtendedDomainSize = (*traits).localExtendedDomainSize;
          transposed              = (*traits).transposed;

          getDFTData();

          if (matrixData != nullptr)
          {
            FFTW<RF>::free(matrixData);
            matrixData = nullptr;
          }
        }

        /**
         * @brief Check whether matrix has already been created
         *
         * @return true if the matrix data is present, else false
         */
        bool valid() const
        {
          return (matrixData != nullptr);
        }

        /**
         * @brief Number of matrix entries stored on this processor
         *
         * This is the size of the extended domain, or its local
         * subset in the case of parallel data distribution.
         *
         * @return number of local degrees of freedom
         */
        Index localMatrixSize() const
        {
          return localExtendedDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         *
         * This is the number of cells per dimension of the
         * extended domain, or the number of cells per dimension
         * for the local part of the extended domain in the case of
         * parallel data distribution.
         *
         * @return tuple of local cells per dimension
         */
        const Indices& localMatrixCells() const
        {
          return localExtendedCells;
        }

        /**
         * @brief Offset between local indices and global indices per dim
         *
         * This is the tuple of offsets, one per dimension, between the
         * start of the local array and the start of the global array
         * spanning all processors.
         *
         * @return tuple of offsets
         */
        const Indices& localMatrixOffset() const
        {
          return localExtendedOffset;
        }

        /**
         * @brief Number of logical entries per dim on this processor
         *
         * This is the number of entries that the local array represents.
         * For the given backend, this is identical with localMatrixCells,
         * but it can differ for other backends, which make use of
         * redundancy in the matrix array.
         *
         * @return tuple of local cells per dimension
         */
        const Indices& localEvalMatrixCells() const
        {
          return localExtendedCells;
        }

        /**
         * @brief Reserve memory before storing any matrix entries
         *
         * Explicitly request the matrix backend to reserve storage for the
         * multidimensional array. This ensures that the backend doesn't
         * waste memory when it won't be used.
         */
        void allocate()
        {
          if (matrixData == nullptr)
            matrixData = FFTW<RF>::alloc_complex(allocLocal);
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
            localExtendedOffset[dim-1] = localExtendedCells[dim-1] * rank;
          }
        }

        /**
         * @brief Transform into Fourier (i.e., frequency) space
         *
         * Perform a forward Fourier transform, mapping from the original
         * domain to the frequency domain. Uses a single FFTW DFT transform.
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

          typename FFTW<RF>::plan plan_forward = FFTW<RF>::mpi_plan_dft(dim,n,matrixData,
              matrixData,(*traits).comm,FFTW_FORWARD,flags);

          if (plan_forward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create forward plan");

          FFTW<RF>::execute(plan_forward);
          FFTW<RF>::destroy_plan(plan_forward);

          for (Index i = 0; i < allocLocal; i++)
          {
            matrixData[i][0] /= extendedDomainSize;
            matrixData[i][1] /= extendedDomainSize;
          }

          transposeIfNeeded();
        }

        /**
         * @brief Transform from Fourier (i.e., frequency) space
         *
         * Perform a backward Fourier transform, mapping from the frequency
         * domain back to the original domain. Uses a single FFTW DFT transform.
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

          typename FFTW<RF>::plan plan_backward = FFTW<RF>::mpi_plan_dft(dim,n,matrixData,
              matrixData,(*traits).comm,FFTW_BACKWARD,flags);

          if (plan_backward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create backward plan");

          FFTW<RF>::execute(plan_backward);
          FFTW<RF>::destroy_plan(plan_backward);
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         *
         * This function returns the matrix entry associated with the given
         * local index (i.e., index for the local part of the array for the
         * extended domain). This backend stores each entry explicitly, and
         * therefore this is just a direct array access.
         *
         * @param index flat index for the local array
         *
         * @return value associated with index
         */
        RF eval(Index index) const
        {
          // no translation necessary
          return get(index);
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         *
         * This function returns the matrix entry associated with the given
         * local indices (i.e., tuple of indices for the local part of the
         * array for the extended domain, taking possible offsets into account).
         * This backend stores each entry explicitly, and therefore this is just
         * a direct array access.
         *
         * @param indices tuple of local indices
         *
         * @return value associated with indices
         */
        RF eval(Indices indices) const
        {
          const Index& index = Traits::indicesToIndex(indices,localExtendedCells);
          return eval(index);
        }

        /**
         * @brief Get matrix entry (using the actual index)
         *
         * This function returns the entry of the array associated with the
         * given index. This backend stores each matrix value for the extended
         * domain, so this is the same as the eval method.
         *
         * @param index flat index for the local array
         *
         * @return value associated with index
         *
         * @see eval
         */
        RF get(Index index) const
        {
          return matrixData[index][0];
        }

        /**
         * @brief Set matrix entry (using the actual index)
         *
         * This function sets the entry of the array associated with the given
         * index. The argument is used for the real part, and the imaginary part
         * is set zero.
         *
         * @param index flat index for the local array
         * @param value value that should be associated with the index
         */
        void set(Index index, RF value)
        {
          matrixData[index][0] = value;
          matrixData[index][1] = 0.;
        }

        /**
         * @brief Dummy function, nothing to do after Fourier transform
         *
         * This function transforms the stored data in some way in the case of
         * other backends (saving memory, or enabling use in parallel field
         * generation), which makes it impossible to apply any transforms
         * after this method has been called. For the given backend, this
         * method does nothing, and transforms can still be used after it
         * has been called.
         */
        void finalize()
        {
          // nothing to do
        }

        private:

        /**
         * @brief Get the domain decomposition data of the Fourier transform
         *
         * This function obtains the local offset and cells in the distributed
         * dimension as prescibed by FFTW.
         */
        void getDFTData()
        {
          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            n[i] = extendedCells[dim-1-i];

          if (dim == 1)
          {
            ptrdiff_t localN02, local0Start2;
            allocLocal = FFTW<RF>::mpi_local_size_1d(n[0],(*traits).comm,FFTW_FORWARD,FFTW_ESTIMATE,
                &localN0,&local0Start,&localN02,&local0Start2);
            if (localN0 != localN02 || local0Start != local0Start2)
              DUNE_THROW(Dune::Exception,"1d size / offset results don't match");
          }
          else
            allocLocal = FFTW<RF>::mpi_local_size(dim,n,(*traits).comm,&localN0,&local0Start);
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_DFTMATRIXBACKEND_HH
