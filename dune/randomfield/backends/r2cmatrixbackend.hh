#ifndef DUNE_RANDOMFIELD_R2CMATRIXBACKEND_HH
#define DUNE_RANDOMFIELD_R2CMATRIXBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Matrix backend that uses real-input discrete Fourier transform (R2C)
     *
     * This matrix backend is an implementation of the circulant embedding
     * method that makes use of data redundancy to save a significant amount
     * of memory and time. By construction, the covariance function is an even
     * real function, which means that its Fourier transform is also even and
     * real. As a consequence, both the required memory and the number of
     * instructions needed for the transform can be reduced by roughly a factor
     * of four. However, FFTW doesn't offer general real even transforms for
     * higher dimensions, only tensor products of onedimensional transforms.
     * Therefore, a real-to-complex DFT is employed to create an Hermitian
     * complex transformed covariance matrix, and then the zero imaginary part
     * is eliminated. This needs more time than a true real-to-real transform
     * would, but the additional memory use can be masked, since the storage
     * for the field backend hasn't been allocated at this point in time.
     *
     * @tparam Traits traits class with data types and definitions
     */
    template<typename Traits>
      class R2CMatrixBackend
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        static_assert(dim != 1, "R2CMatrixBackend requires dim > 1");

        const std::shared_ptr<Traits> traits;

        int rank, commSize;

        ptrdiff_t allocLocal, localN0, local0Start;

        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Indices localExtendedOffset;
        Index   localExtendedDomainSize;

        Indices localR2CComplexCells;
        Index   localR2CComplexDomainSize;
        Indices localR2CRealCells;
        Index   localR2CRealDomainSize;

        mutable typename FFTW<RF>::complex* matrixData;
        mutable Indices indices;

        bool transformed = false;
        bool transposed, finalized;

        public:

        /**
         * @brief Constructor
         *
         * Imports FFTW wisdom if configured to do so.
         *
         * @param traits_ traits object with parameters and communication
         */
        R2CMatrixBackend(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            matrixData(nullptr),
            finalized(false)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using R2CMatrixBackend" << std::endl;

          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            if ((*traits).rank == 0)
              FFTW<RF>::import_wisdom_from_filename("wisdom-R2CMatrix.ini");

            FFTW<RF>::mpi_broadcast_wisdom((*traits).comm);
          }
        }

        /**
         * @brief Destructor
         *
         * Cleans up allocated arrays and FFTW plans. Exports FFTW
         * wisdom if configured to do so.
         */
        ~R2CMatrixBackend()
        {
          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            FFTW<RF>::mpi_gather_wisdom((*traits).comm);

            if ((*traits).rank == 0)
              FFTW<RF>::export_wisdom_to_filename("wisdom-R2CMatrix.ini");
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
          checkFinalized();

          rank     = (*traits).rank;
          commSize = (*traits).commSize;

          extendedCells           = (*traits).extendedCells;
          extendedDomainSize      = (*traits).extendedDomainSize;
          localExtendedCells      = (*traits).localExtendedCells;
          localExtendedOffset     = (*traits).localExtendedOffset;
          localExtendedDomainSize = (*traits).localExtendedDomainSize;
          transposed              = (*traits).transposed;

          if (dim == 2 && transposed)
            DUNE_THROW(Dune::Exception,
                "R2CMatrixBackend supports transposed output only for dim > 2");

          getR2CData();

          getR2CCells();

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
         * For this backend, the method may return two different values,
         * depending on whether the data has been transformed or not.
         * The untransformed array contains real numbers, and the size of
         * the array is the same as with DFTMatrixBackend, which is simply
         * the size of the extended domain, or its local subset in the
         * case of parallel data distribution. The transformed array
         * contains complex numbers, i.e., twice the data per index, but
         * has only half as many indices, because half the entries are
         * redundant due to symmetry.
         *
         * @return number of local degrees of freedom
         */
        Index localMatrixSize() const
        {
          if (transformed)
            return localR2CComplexDomainSize;
          else
            return localR2CRealDomainSize;
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
          return localR2CRealCells;
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
         * This is after transformation, so the total number of indices is
         * half that of the original array. The calling code has to take
         * this into account, because only the part that is actually stored
         * can be accessed directly, and the other half has to be reconstructed
         * by mapping to the associated other index.
         *
         * @return tuple of local cells per dimension
         */
        const Indices& localEvalMatrixCells() const
        {
          return localR2CComplexCells;
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

            getR2CCells();
          }

          transformed = !transformed;
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
          checkFinalized();

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

          typename FFTW<RF>::plan plan_forward = FFTW<RF>::mpi_plan_dft_r2c(dim,n,(RF*)matrixData,
              matrixData,(*traits).comm,flags);

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
         * domain back to the original domain. Uses a single FFTW complex-to-real
         * DFT transform: the input is a multidimensional Hermitian array of
         * complex numbers, with half the data not stored because of redundancy,
         * and the output is an array of real numbers.
         */
        void backwardTransform()
        {
          checkFinalized();
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

          typename FFTW<RF>::plan plan_backward = FFTW<RF>::mpi_plan_dft_c2r(dim,n,matrixData,
              (RF*)matrixData,(*traits).comm,flags);

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
         * extended domain). This is used after transform, so half the
         * entries are missing and have to be reconstructed by accessing the
         * corresponding other index. At this point, the data is purely real,
         * because the finalize method has been called beforehand.
         *
         * @param index flat index for the local array
         *
         * @return value associated with index
         */
        RF eval(Index index) const
        {
          return ((RF*)matrixData)[index];
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         *
         * This function returns the matrix entry associated with the given
         * local indices (i.e., tuple of indices for the local part of the
         * array for the extended domain, taking possible offsets into account).
         * This is used after transform, so half the entries are missing and
         * have to be reconstructed by accessing the corresponding other index.
         * The method takes care of this, and remaps indices that would access
         * parts of the array that are not actually present.
         *
         * @param indices tuple of local indices
         *
         * @return value associated with indices
         */
        RF eval(Indices indices) const
        {
          for (unsigned int i = 0; i < dim; i++)
            if (indices[i] >= localR2CComplexCells[i])
              indices[i] = localExtendedCells[i] - indices[i];

          const Index& index = Traits::indicesToIndex(indices,localR2CComplexCells);
          return eval(index);
        }

        /**
         * @brief Get matrix entry (using the actual index)
         *
         * This function returns the entry of the array associated with the
         * given index. It uses the actual index of the untransformed array,
         * and may not be used after the matrix backend has been finalized.
         *
         * @param index flat index for the local array
         *
         * @return value associated with index
         *
         * @see eval
         * @see finalize
         */
        RF get(Index index) const
        {
          checkFinalized();
          return ((RF*)matrixData)[index];
        }

        /**
         * @brief Set matrix entry (using the actual index)
         *
         * This function sets the entry of the array associated with the given
         * index. It uses the actual index of the untransformed array,
         * and may not be used after the matrix backend has been finalized.
         *
         * @param index flat index for the local array
         * @param value value that should be associated with the index
         *
         * @see finalize
         */
        void set(Index index, RF value)
        {
          checkFinalized();
          ((RF*)matrixData)[index] = value;
        }

        /**
         * @brief Remove zero imaginary part
         *
         * This function takes the transformed array, which is an Hermitian
         * array of complex numbers that has half the number of indices as
         * the original array, and deletes its imaginary part, since that is
         * zero. After this function has been called, the array will have half
         * the number of entries as the original array before the transform,
         * and the backend can no longer be modified.
         * */
        void finalize()
        {
          typename FFTW<RF>::complex* uncut = matrixData;
          matrixData = (typename FFTW<RF>::complex*)FFTW<RF>::alloc_real(allocLocal);
          for (Index i = 0; i < allocLocal; i++)
            ((RF*)matrixData)[i] = uncut[i][0];
          FFTW<RF>::free(uncut);

          finalized = true;
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

        /**
         * @brief Raise an exception if the field can no longer be modified
         *
         * This function raises an exception if any operation is attempted
         * that can no longer be used after the matrix backend has been
         * finalized, e.g., forward or backward transforms, or accessing the
         * raw data using get and set.
         */
        void checkFinalized() const
        {
          if (finalized)
            DUNE_THROW(Dune::Exception,
                "matrix is finalized, use DFTMatrixBackend if you need to modify the matrix");
        }
      };

  }
}

#endif // DUNE_RANDOMFIELD_R2CMATRIXBACKEND_HH
