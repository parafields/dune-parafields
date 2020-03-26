// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_DCTMATRIXBACKEND_HH
#define	DUNE_RANDOMFIELD_DCTMATRIXBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Matrix backend that uses discrete cosine transform (DCT)
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

        bool transposed;

        enum {mirrorForward, mirrorBackward};

        public:

        DCTMatrixBackend<Traits>(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            matrixData(nullptr)
        {
          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            if ((*traits).rank == 0)
              FFTW<RF>::import_wisdom_from_filename("wisdom-DCTMatrix.ini");

            FFTW<RF>::mpi_broadcast_wisdom((*traits).comm);
          }
        }

        ~DCTMatrixBackend<Traits>()
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
         */
        void update()
        {
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
          return localDCTDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         */
        const Indices& localMatrixCells() const
        {
          return localDCTCells;
        }

        /**
         * @brief Offset between local indices and global indices per dim
         */
        const Indices& localMatrixOffset() const
        {
          return localDCTOffset;
        }

        /**
         * @brief Number of logical entries per dim on this processor
         */
        const Indices& localEvalMatrixCells() const
        {
          return localExtendedCells;
        }

        /**
         * @brief Reserve memory before storing any matrix entries
         */
        void allocate()
        {
          if (matrixData == nullptr)
            matrixData = FFTW<RF>::alloc_real(allocLocal);
        }

        /**
         * @brief Switch last two dimensions (for transposed transforms)
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
         */
        void backwardTransform()
        {
          if (commSize > 1)
            DUNE_THROW(Dune::Exception,"DCT backward transforms are not implemented in the parallel case"
                " because the matrix has been redistributed");

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
         */
        const RF& eval(Index index) const
        {
          Traits::indexToIndices(index,indices,localExtendedCells);
          return eval(indices);
        }

        /**
         * @brief Evaluate matrix entry (in virtual, i.e., logical indices)
         */
        const RF& eval(Indices indices) const
        {
          for (unsigned int i = 0; i < dim; i++)
            if (indices[i] + localEvalOffset[i] >= evalCells[i])
              indices[i] = extendedCells[i] - indices[i] - localEvalOffset[i];

          const Index& smallIndex = Traits::indicesToIndex(indices,localEvalCells);

          return matrixData[smallIndex];
        }

        /**
         * @brief Get matrix entry (using the actual index)
         */
        const RF& get(Index index) const
        {
          return matrixData[index];
        }

        /**
         * @brief Get matrix entry (using the actual index)
         */
        void set(Index index, RF value)
        {
          matrixData[index] = value;
        }

        /**
         * @brief Mirror matrix on domain boundary in parallel case
         */
        void finalize() const
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

          FFTW<RF>::free(unmirrored);
          unmirrored = nullptr;
        }

        private:

        /**
         * @brief Get the domain decomposition data of the Fourier transform
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
          if (commSize != 1)
          {
            evalCells[dim-1]       = extendedCells[dim-1];
            localEvalCells[dim-1]  = localExtendedCells[dim-1];
            localEvalOffset[dim-1] = localExtendedOffset[dim-1];
          }
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_DCTMATRIXBACKEND_HH
