// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_R2CMATRIXBACKEND_HH
#define	DUNE_RANDOMFIELD_R2CMATRIXBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Matrix backend that uses real-input discrete Fourier transform (R2C)
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
        Index localR2CComplexDomainSize;
        Indices localR2CRealCells;
        Index localR2CRealDomainSize;

        mutable fftw_complex* matrixData;
        mutable Indices indices;

        bool transposed;

        public:

        R2CMatrixBackend<Traits>(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            matrixData(nullptr)
        {
          if ((*traits).rank == 0)
            std::cout << "R2CMatrixBackend" << std::endl;
        }

        ~R2CMatrixBackend<Traits>()
        {
          if (matrixData != nullptr)
          {
            fftw_free(matrixData);
            matrixData = nullptr;
          }
        }

        /*
         * @brief Update internal data after creation or refinement
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

          if (dim == 2 && transposed)
            DUNE_THROW(Dune::Exception,
                "R2CMatrixBackend supports transposed output only for dim > 2");

          getR2CData();

          getR2CCells();

          if (matrixData != nullptr)
          {
            fftw_free(matrixData);
            matrixData = nullptr;
          }
        }

        /**
         * @brief Check whether matrix has already been created
         */
        bool valid() const
        {
          return (matrixData != nullptr);
        }

        /**
         * @brief Number of matrix entries stored on this processor
         */
        Index localMatrixSize() const
        {
          return localR2CRealDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         */
        const Indices& localMatrixCells() const
        {
          return localR2CRealCells;
        }

        /**
         * @brief Offset between local indices and global indices per dim
         */
        const Indices& localMatrixOffset() const
        {
          return localExtendedOffset;
        }

        /**
         * @brief Number of logical entries per dim on this processor
         */
        const Indices& localEvalMatrixCells() const
        {
          return localR2CComplexCells;
        }

        /**
         * @brief Reserve memory before storing any matrix entries
         */
        void allocate()
        {
          if (matrixData == nullptr)
            matrixData = fftw_alloc_complex(allocLocal);
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

          fftw_plan plan_forward = fftw_mpi_plan_dft_r2c(dim,n,(RF*)matrixData,
              matrixData,(*traits).comm,flags);

          if (plan_forward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create forward plan");

          fftw_execute(plan_forward);
          fftw_destroy_plan(plan_forward);

          for (Index i = 0; i < allocLocal; i++)
          {
            matrixData[i][0] /= extendedDomainSize;
            matrixData[i][1] /= extendedDomainSize;
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

          fftw_plan plan_backward = fftw_mpi_plan_dft_c2r(dim,n,matrixData,
              (RF*)matrixData,(*traits).comm,flags);

          if (plan_backward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create backward plan");

          fftw_execute(plan_backward);
          fftw_destroy_plan(plan_backward);
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         */
        const RF& eval(Index index) const
        {
          return matrixData[index][0];
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         */
        const RF eval(Indices indices) const
        {
          for (unsigned int i = 0; i < dim; i++)
            if (indices[i] >= localR2CComplexCells[i])
              indices[i] = localExtendedCells[i] - indices[i];

          const Index& index = Traits::indicesToIndex(indices,localR2CComplexCells);
          return eval(index);
        }

        /**
         * @brief Get matrix entry (using the actual index)
         */
        const RF& get(Index index) const
        {
          return ((RF*)matrixData)[index];
        }

        /**
         * @brief Get matrix entry (using the actual index)
         */
        void set(Index index, RF value)
        {
          ((RF*)matrixData)[index] = value;
        }

        /**
         * @brief Dummy function, nothing to do after Fourier transform
         */
        void finalize()
        {
          // nothing to do
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

          allocLocal = fftw_mpi_local_size(dim, n, (*traits).comm, &localN0, &local0Start);
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

#endif // DUNE_RANDOMFIELD_R2CMATRIXBACKEND_HH