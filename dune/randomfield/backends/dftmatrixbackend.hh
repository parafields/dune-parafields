// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_DFTMATRIXBACKEND_HH
#define	DUNE_RANDOMFIELD_DFTMATRIXBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Matrix backend that uses discrete Fourier transform (DFT)
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

        mutable fftw_complex* matrixData;

        bool transposed;

        public:

        DFTMatrixBackend<Traits>(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            matrixData(nullptr)
        {}

        ~DFTMatrixBackend<Traits>()
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

          getDFTData();

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
          return localExtendedDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         */
        const Indices& localMatrixCells() const
        {
          return localExtendedCells;
        }

        /**
         * @brief Offset between local indices and global indices per dim
         */
        const Indices& localMatrixOffset() const
        {
          return localExtendedOffset;
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
            localExtendedOffset[dim-1] = localExtendedCells[dim-1] * rank;
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

          fftw_plan plan_forward = fftw_mpi_plan_dft(dim,n,matrixData,
              matrixData,(*traits).comm,FFTW_FORWARD,flags);

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

          fftw_plan plan_backward = fftw_mpi_plan_dft(dim,n,matrixData,
              matrixData,(*traits).comm,FFTW_BACKWARD,flags);

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
          // no translation necessary
          return get(index);
        }

        /**
         * @brief Get matrix entry (using the actual index)
         */
        const RF& get(Index index) const
        {
          return matrixData[index][0];
        }

        /**
         * @brief Get matrix entry (using the actual index)
         */
        void set(Index index, RF value)
        {
          matrixData[index][0] = value;
          matrixData[index][1] = 0.;
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

#endif // DUNE_RANDOMFIELD_DFTMATRIXBACKEND_HH
