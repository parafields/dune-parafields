#ifndef DUNE_RANDOMFIELD_DCTMATRIXBACKEND_HH
#define DUNE_RANDOMFIELD_DCTMATRIXBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Matrix backend that uses discrete cosine transform (DCT)
     *
     * This matrix backend implements the circulant embedding method
     * for covariance functions with axial symmetries, i.e., functions
     * for which flipping the sign in one of the dimensions of the
     * argument doesn't change the function value. This is the case,
     * e.g., for isotropic, axially anisotropic, and separable
     * covariance functions, and combinations thereof. These additional
     * symmetries make it possible to reduce the number of cells by a
     * factor of two in each dimension. In cases where the minimum
     * embedding factor of two is sufficient, the data on the extended
     * domain can be represented on the original domain with one row
     * of cells per dimension as padding.
     *
     * @tparam Traits traits class with data types and definitions
     */
    template<typename Traits>
      class DCTMatrixBackend
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        static_assert(dim != 1, "DCTMatrixBackend requires dim > 1");

        const std::shared_ptr<Traits> traits;

        int rank, commSize;

        ptrdiff_t allocLocal, localN0, local0Start, localN0Trans, local0StartTrans;

        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Indices localExtendedOffset;

        Index   localDCTDomainSize;
        Index   dctDomainSize;
        Indices dctCells;
        Indices localDCTCells;
        Indices localDCTOffset;
        Indices evalCells;
        Indices localEvalCells;
        Indices localEvalOffset;

        mutable RF* matrixData;
        mutable Indices indices;

        bool transposed, finalized;

        enum {mirrorForward, mirrorBackward};

        public:

        /**
         * @brief Constructor
         *
         * Imports FFTW wisdom if configured to do so.
         *
         * @param traits_ traits object with parameters and communication
         */
        DCTMatrixBackend(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            matrixData(nullptr),
            finalized(false)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using DCTMatrixBackend" << std::endl;

          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            if ((*traits).rank == 0)
              FFTW<RF>::import_wisdom_from_filename("wisdom-DCTMatrix.ini");

            FFTW<RF>::mpi_broadcast_wisdom((*traits).comm);
          }
        }

        /**
         * @brief Destructor
         *
         * Cleans up allocated arrays and FFTW plans. Exports FFTW
         * wisdom if configured to do so.
         */
        ~DCTMatrixBackend()
        {
          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            FFTW<RF>::mpi_gather_wisdom((*traits).comm);

            if ((*traits).rank == 0)
              FFTW<RF>::export_wisdom_to_filename("wisdom-DCTMatrix.ini");
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

          extendedCells       = (*traits).extendedCells;
          extendedDomainSize  = (*traits).extendedDomainSize;
          localExtendedCells  = (*traits).localExtendedCells;
          localExtendedOffset = (*traits).localExtendedOffset;
          transposed          = (*traits).transposed;

          getDCTData();

          getDCTCells(localN0,local0Start);

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
         * subset in the case of parallel data distribution. For this
         * backend, it is the size of the original domain plus a small
         * amount needed for padding.
         *
         * @return number of local degrees of freedom
         */
        Index localMatrixSize() const
        {
          return localDCTDomainSize;
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
          return localDCTCells;
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
          return localDCTOffset;
        }

        /**
         * @brief Number of logical entries per dim on this processor
         *
         * This is the number of entries that the local array represents.
         * For the given backend, this is approximately half the number
         * of cells of the extended domain in each dimension, except for
         * parallel runs, where the distributed dimension is mirrored to
         * make the entries locally available on the correct processors.
         */
        const Indices& localEvalMatrixCells() const
        {
          return localEvalCells;
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
            matrixData = FFTW<RF>::alloc_real(allocLocal);
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
         *
         * @param localN0     current number of cells in distributed dimension
         * @param local0Start current offset in distributed dimension
         */
        void transposeIfNeeded(ptrdiff_t& localN0, ptrdiff_t& local0Start)
        {
          if (transposed)
          {
            std::swap(extendedCells[dim-1],extendedCells[dim-2]);
            localExtendedCells[dim-1] = extendedCells[dim-1] / commSize;
            localExtendedCells[dim-2] = extendedCells[dim-2];
            localExtendedOffset[dim-1] = localExtendedCells[dim-1] * rank;

            getDCTCells(localN0,local0Start);
          }
        }

        /**
         * @brief Transform into Fourier (i.e., frequency) space
         *
         * Perform a forward Fourier transform, mapping from the original
         * domain to the frequency domain. Uses a single FFTW real-to-real
         * DFT transform.
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
          typename FFTW<RF>::r2r_kind k[dim];
          for (unsigned int i = 0; i < dim; i++)
          {
            n[i] = extendedCells[dim-1-i]/2 + 1;
            k[i] = FFTW_REDFT00;
          }

          typename FFTW<RF>::plan plan_forward = FFTW<RF>::mpi_plan_r2r(dim,n,matrixData,
              matrixData,(*traits).comm,k,flags);

          if (plan_forward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create forward plan");

          FFTW<RF>::execute(plan_forward);
          FFTW<RF>::destroy_plan(plan_forward);

          for (Index i = 0; i < allocLocal; i++)
            matrixData[i] /= extendedDomainSize;

          transposeIfNeeded(localN0Trans,local0StartTrans);
        }

        /**
         * @brief Transform from Fourier (i.e., frequency) space
         *
         * Perform a backward Fourier transform, mapping from the frequency
         * domain back to the original domain. Uses a single FFTW real-to-real
         * DFT transform.
         */
        void backwardTransform()
        {
          checkFinalized();
          transposeIfNeeded(localN0,local0Start);

          unsigned int flags;
          if ((*traits).config.template get<bool>("fftw.measure",false))
            flags = FFTW_MEASURE;
          else
            flags = FFTW_ESTIMATE;
          if (transposed)
            flags |= FFTW_MPI_TRANSPOSED_IN;

          ptrdiff_t n[dim];
          typename FFTW<RF>::r2r_kind k[dim];
          for (unsigned int i = 0; i < dim; i++)
          {
            n[i] = extendedCells[dim-1-i]/2 + 1;
            k[i] = FFTW_REDFT00;
          }

          typename FFTW<RF>::plan plan_backward = FFTW<RF>::mpi_plan_r2r(dim,n,matrixData,
              matrixData,(*traits).comm,k,flags);

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
         * extended domain). Half of the entries per dimension are not
         * actually stored, since they are redundant. Therefore, this
         * function has to convert its argument into the corresponding
         * index that actually has data associated with it.
         *
         * @param index flat index for the local array
         *
         * @return value associated with index
         */
        RF eval(Index index) const
        {
          Traits::indexToIndices(index,indices,localExtendedCells);
          return eval(indices);
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         *
         * This function returns the matrix entry associated with the given
         * local indices (i.e., tuple of indices for the local part of the
         * array for the extended domain, taking possible offsets into account).
         * In each dimension, half of the entries are not actually stored, since
         * they are redundant. This function takes care of this and maps any
         * such tuple of indices to the corresponding index tuple that actually
         * has data associated with it.
         *
         * @param indices tuple of local indices
         *
         * @return value associated with indices
         */
        RF eval(Indices indices) const
        {
          for (unsigned int i = 0; i < dim; i++)
            if (indices[i] + localEvalOffset[i] >= evalCells[i])
              indices[i] = extendedCells[i] - indices[i] - localEvalOffset[i];

          const Index& smallIndex = Traits::indicesToIndex(indices,localEvalCells);

          return matrixData[smallIndex];
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
          return matrixData[index];
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
          matrixData[index] = value;
        }

        /**
         * @brief Mirror matrix on domain boundary in parallel case
         *
         * This function takes the transformed array and mirrors the
         * distributed dimension on the domain boundary in the parallel
         * case, i.e., It reverts the savings gained by using the
         * underlying symmetry in this one dimension, and the size of
         * the array grows by roughly a factor of two. This is needed
         * for parallel field generation, because otherwise the data
         * for some of the cells would lie on another processor and
         * couldn't be accessed. After this function has been called,
         * the backend can no longer be modified.
         */
        void finalize()
        {
          // nothing to do in sequential case
          if (commSize == 1)
            return;

          std::vector<MPI_Request> request(4);

          RF* unmirrored = matrixData;
          unsigned int mirrorAllocLocal = localExtendedCells[dim-1];
          for (unsigned int i = 0; i < dim-1; i++)
            mirrorAllocLocal *= localDCTCells[i];
          matrixData = FFTW<RF>::alloc_real(mirrorAllocLocal);

          Index strideWidth = localExtendedCells[dim-1];
          Index sliceSize = 1;
          for (unsigned int i = 0; i < dim - 1; i++)
            sliceSize *= localDCTCells[i];

          std::array<unsigned int, 2> sendPartners;
          unsigned int sendSplit;
          // first
          if (rank == 0)
          {
            sendPartners[0] = 0;
            sendPartners[1] = 0;
            sendSplit = 0;
          }
          // normal
          else
          {
            sendPartners[0] = (rank     * (strideWidth/2+1)    ) / strideWidth;
            sendPartners[1] = ((rank+1) * (strideWidth/2+1) - 1) / strideWidth;
            sendSplit       = (rank     * (strideWidth/2+1)    ) % strideWidth;
          }

          Index sendSize = localDCTCells[dim-1];
          // skip last slice on way forward
          if ((rank+1) * (strideWidth/2+1) >= dctCells[dim-1]
              && rank * (strideWidth/2+1) < dctCells[dim-1] )
            sendSize--;
          if (sendPartners[0] == sendPartners[1] || strideWidth - sendSplit >= sendSize)
          {
            if (sendSize > 0)
            {
              const Index size = sendSize * sliceSize;

              MPI_Isend(&(unmirrored[0]), size, mpiType<RF>,
                  sendPartners[0], mirrorForward, (*traits).comm, &(request[0]));
              request[1] = MPI_REQUEST_NULL;
            }
            else
            {
              // nothing to send
              request[0] = MPI_REQUEST_NULL;
              request[1] = MPI_REQUEST_NULL;
            }
          }
          else
          {
            const Index size1 = (strideWidth - sendSplit)              * sliceSize;
            const Index size2 = (sendSize - (strideWidth - sendSplit)) * sliceSize;

            MPI_Isend(&(unmirrored[0]),     size1, mpiType<RF>,
                sendPartners[0], mirrorForward, (*traits).comm, &(request[0]));
            MPI_Isend(&(unmirrored[size1]), size2, mpiType<RF>,
                sendPartners[1], mirrorForward, (*traits).comm, &(request[1]));
          }

          // first
          if (rank == 0)
          {
            sendPartners[0] = commSize - 1;
            sendPartners[1] = commSize - 1;
            sendSplit = 0;
          }
          // normal
          else
          {
            sendPartners[0] = commSize - 1 - (rank     * (strideWidth/2+1) - 1) / strideWidth;
            sendPartners[1] = commSize - 1 - ((rank+1) * (strideWidth/2+1) - 2) / strideWidth;
            sendSplit       =                (rank     * (strideWidth/2+1) - 1) % strideWidth;
          }

          sendSize = localDCTCells[dim-1];
          // skip first slice on way back
          if (rank == 0)
            sendSize--;
          if (sendPartners[0] == sendPartners[1] || strideWidth - sendSplit >= sendSize)
          {
            if (sendSize > 0)
            {
              const Index size = sendSize * sliceSize;

              // skip first slice on way back
              MPI_Isend(&(unmirrored[(rank==0)?sliceSize:0]), size, mpiType<RF>,
                  sendPartners[0], mirrorBackward, (*traits).comm, &(request[2]));
              request[3] = MPI_REQUEST_NULL;
            }
            else
            {
              // nothing to send
              request[2] = MPI_REQUEST_NULL;
              request[3] = MPI_REQUEST_NULL;
            }
          }
          else
          {
            const Index size1 = (strideWidth - sendSplit)              * sliceSize;
            const Index size2 = (sendSize - (strideWidth - sendSplit)) * sliceSize;

            MPI_Isend(&(unmirrored[0]), size1, mpiType<RF>,
                sendPartners[0], mirrorBackward, (*traits).comm, &(request[2]));
            MPI_Isend(&(unmirrored[size1]), size2, mpiType<RF>,
                sendPartners[1], mirrorBackward, (*traits).comm, &(request[3]));
          }

          std::array<unsigned int, 3> recvPartners;
          unsigned int recvSplit;
          if (rank < commSize/2)
          {
            recvPartners[0] =   (rank     * strideWidth    ) / (strideWidth/2+1);
            recvPartners[2] =   ((rank+1) * strideWidth - 1) / (strideWidth/2+1);
            recvPartners[1] = (recvPartners[0] + recvPartners[2])/2;
            recvSplit       =   (rank     * strideWidth    ) % (strideWidth/2+1);

            const Index size1 = (strideWidth/2+1 - recvSplit) * sliceSize;

            MPI_Recv(&(matrixData[0]), size1, mpiType<RF>,
                recvPartners[0], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);

            if (recvPartners[1] != recvPartners[0] && recvPartners[1] != recvPartners[2])
            {
              const Index size2 = (strideWidth/2+1)                               * sliceSize;
              const Index size3 = (strideWidth - 2*(strideWidth/2+1) + recvSplit) * sliceSize;

              MPI_Recv(&(matrixData[size1]),       size2, mpiType<RF>,
                  recvPartners[1], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);
              MPI_Recv(&(matrixData[size1+size2]), size3, mpiType<RF>,
                  recvPartners[2], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);
            }
            else if (recvPartners[2] != recvPartners[0])
            {
              const Index size2 = (strideWidth - (strideWidth/2+1) + recvSplit) * sliceSize;

              MPI_Recv(&(matrixData[size1]), size2, mpiType<RF>,
                  recvPartners[2], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);
            }
          }
          else if (rank == commSize/2 && commSize % 2 != 0)
          {
            recvPartners[0] = (rank * strideWidth) / (strideWidth/2+1);
            recvPartners[1] = recvPartners[0] + 1;
            recvSplit       = (rank * strideWidth) % (strideWidth/2+1);

            if (strideWidth/2+1 - recvSplit >= strideWidth/2)
            {
              const Index size = (strideWidth/2) * sliceSize;

              MPI_Recv(&(matrixData[0]), size, mpiType<RF>,
                  recvPartners[0], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);
            }
            else
            {
              const Index size1 = ((strideWidth/2+1) - recvSplit)                 * sliceSize;
              const Index size2 = (strideWidth/2 - (strideWidth/2+1) + recvSplit) * sliceSize;

              MPI_Recv(&(matrixData[0]), size1, mpiType<RF>,
                  recvPartners[0], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);
              MPI_Recv(&(matrixData[size1]), size2, mpiType<RF>,
                  recvPartners[1], mirrorForward, (*traits).comm, MPI_STATUS_IGNORE);
            }

            recvPartners[0] = ((commSize - 1 - rank) * strideWidth + 1) / (strideWidth/2+1);
            recvPartners[1] = recvPartners[0] + 1;
            recvSplit       = ((commSize - 1 - rank) * strideWidth + 1) % (strideWidth/2+1);

            if (strideWidth/2+1 - recvSplit >= strideWidth/2)
            {
              const Index size = (strideWidth/2) * sliceSize;

              MPI_Recv(&(matrixData[strideWidth/2*sliceSize]), size, mpiType<RF>,
                  recvPartners[0], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);
            }
            else
            {
              const Index size1 = ((strideWidth/2+1) - recvSplit)                 * sliceSize;
              const Index size2 = (strideWidth/2 - (strideWidth/2+1) + recvSplit) * sliceSize;

              MPI_Recv(&(matrixData[strideWidth/2*sliceSize]),       size1, mpiType<RF>,
                  recvPartners[0], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);
              MPI_Recv(&(matrixData[strideWidth/2*sliceSize+size1]), size2, mpiType<RF>,
                  recvPartners[1], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);
            }
          }
          else
          {
            recvPartners[0] = ((commSize - 1 - rank    ) * strideWidth + 1) / (strideWidth/2+1);
            recvPartners[2] = ((commSize - 1 - rank + 1) * strideWidth    ) / (strideWidth/2+1);
            recvPartners[1] = (recvPartners[0] + recvPartners[2])/2;
            recvSplit       = ((commSize - 1 - rank    ) * strideWidth + 1) % (strideWidth/2+1);

            const Index size1 = ((strideWidth/2+1) - recvSplit) * sliceSize;

            MPI_Recv(&(matrixData[0]), size1, mpiType<RF>,
                recvPartners[0], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);

            if (recvPartners[1] != recvPartners[0] && recvPartners[1] != recvPartners[2])
            {
              const Index size2 = (strideWidth/2+1)                               * sliceSize;
              const Index size3 = (strideWidth - 2*(strideWidth/2+1) + recvSplit) * sliceSize;

              MPI_Recv(&(matrixData[size1]),       size2, mpiType<RF>,
                  recvPartners[1], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);
              MPI_Recv(&(matrixData[size1+size2]), size3, mpiType<RF>,
                  recvPartners[2], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);
            }
            else if (recvPartners[2] != recvPartners[0])
            {
              const Index size2 = (strideWidth - (strideWidth/2+1) + recvSplit) * sliceSize;

              MPI_Recv(&(matrixData[size1]), size2, mpiType<RF>,
                  recvPartners[2], mirrorBackward, (*traits).comm, MPI_STATUS_IGNORE);
            }
          }

          if (rank == commSize/2 && commSize % 2 != 0)
          {
            for (Index i = 0; i < strideWidth/4; i++)
              for (Index j = 0; j < sliceSize; j++)
                std::swap(matrixData[(strideWidth/2 + i) * sliceSize + j],
                    matrixData[(strideWidth - 1 - i) * sliceSize + j]);
          }
          else if (rank >= commSize/2)
          {
            for (Index i = 0; i < strideWidth/2; i++)
              for (Index j = 0; j < sliceSize; j++)
                std::swap(matrixData[i * sliceSize + j],
                    matrixData[(strideWidth - 1 - i) * sliceSize + j]);
          }

          MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);

          evalCells[dim-1]       = extendedCells[dim-1];
          localEvalCells[dim-1]  = localExtendedCells[dim-1];
          localEvalOffset[dim-1] = localExtendedOffset[dim-1];

          FFTW<RF>::free(unmirrored);
          unmirrored = nullptr;

          finalized = true;
        }

        private:

        /**
         * @brief Get the domain decomposition data of the Fourier transform
         *
         * This function obtains the local offset and cells in the distributed
         * dimension as prescibed by FFTW.
         */
        void getDCTData()
        {
          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            n[i] = extendedCells[dim-1-i]/2 + 1;

          allocLocal = FFTW<RF>::mpi_local_size_transposed(dim,n,(*traits).comm,
              &localN0,&local0Start,&localN0Trans,&local0StartTrans);
        }

        /**
         * @brief Calculate DCT cells from extended domain cells
         *
         * This function computes the number of cells per dimension,
         * which is half the number of cells in the extended domain
         * plus one, since the content of the other cells is known
         * due to the underlying symmetries.
         */
        void getDCTCells(ptrdiff_t localN0, ptrdiff_t local0Start)
        {
          localDCTDomainSize = 1;
          dctDomainSize = 1;
          for (unsigned int i = 0; i < dim; i++)
          {
            dctCells[i] = extendedCells[i]/2 + 1;
            if (i == dim-1)
            {
              localDCTCells[i] = localN0;
              localDCTOffset[i] = local0Start;
            }
            else
            {
              localDCTCells[i] = dctCells[i];
              localDCTOffset[i] = 0;
            }
            dctDomainSize *= dctCells[i];
            localDCTDomainSize *= localDCTCells[i];
          }

          evalCells       = dctCells;
          localEvalCells  = localDCTCells;
          localEvalOffset = localDCTOffset;
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

#endif // DUNE_RANDOMFIELD_DCTMATRIXBACKEND_HH
