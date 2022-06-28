#ifndef DUNE_RANDOMFIELD_DCTDSTFIELDBACKEND_HH
#define DUNE_RANDOMFIELD_DCTDSTFIELDBACKEND_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Extended field backend that uses discrete cosine / sine transform (DCT/DST)
     */
    template<typename Traits>
      class DCTDSTFieldBackend
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        static_assert(dim != 1, "DCTDSTMatrixBackend requires dim > 1");

        const std::shared_ptr<Traits> traits;

        int rank, commSize;

        ptrdiff_t allocLocal, localN0, local0Start, localN0Trans, local0StartTrans;

        Indices localCells;
        Indices localOffset;
        Index   localDomainSize;
        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Indices localExtendedOffset;

        Index   localDCTDomainSize;
        Index   dctDomainSize;
        Indices dctCells;
        Indices localDCTCells;
        Indices localDCTOffset;
        Indices localDCTDSTCells;

        mutable RF* fieldData;
        mutable Indices indices;

        Index sliceSize;
        bool odd[dim];
        typename FFTW<RF>::r2r_kind k[dim];
        bool transposed;

        enum {toCompatible, fromCompatible, toExtended, fromExtended};

        public:

        /**
         * @brief Constructor
         *
         * Imports FFTW wisdom if configured to do so.
         *
         * @param traits_ traits object with parameters and communication
         */
        DCTDSTFieldBackend(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            fieldData(nullptr)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using DCTDSTFieldBackend" << std::endl;

          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            if ((*traits).rank == 0)
              FFTW<RF>::import_wisdom_from_filename("wisdom-DCTDSTField.ini");

            FFTW<RF>::mpi_broadcast_wisdom((*traits).comm);
          }
        }

        /**
         * @brief Destructor
         *
         * Cleans up allocated arrays and FFTW plans. Exports FFTW
         * wisdom if configured to do so.
         */
        ~DCTDSTFieldBackend()
        {
          if ((*traits).config.template get<bool>("fftw.useWisdom",false))
          {
            FFTW<RF>::mpi_gather_wisdom((*traits).comm);

            if ((*traits).rank == 0)
              FFTW<RF>::export_wisdom_to_filename("wisdom-DCTDSTField.ini");
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

          localCells          = (*traits).localCells;
          localOffset         = (*traits).localOffset;
          localDomainSize     = (*traits).localDomainSize;
          extendedCells       = (*traits).extendedCells;
          extendedDomainSize  = (*traits).extendedDomainSize;
          localExtendedCells  = (*traits).localExtendedCells;
          localExtendedOffset = (*traits).localExtendedOffset;
          transposed          = (*traits).transposed;

          getDCTData();

          getDCTCells(localN0,local0Start);

          setType(0);

          if (fieldData != nullptr)
          {
            FFTW<RF>::free(fieldData);
            fieldData = nullptr;
          }
        }

        /**
         * @brief Define even (DCT) and odd (DST) dimensions
         *
         * This function configures the backend to represent extended
         * random fields of a given symmetry in each dimension, either
         * even or odd. It configures the correct FFTW flags, and
         * computes the correct array size for FFTW, since the package
         * handles the zeros in odd dimensions implicitly, which means
         * the data has to be shifted and padding added / removed.
         *
         * @param type number that encodes evenness / oddness in binary
         */
        void setType(unsigned int type)
        {
          for (unsigned int i = 0; i < dim; i++)
          {
            if (type % 2 == 0)
            {
              odd[i] = false;
              k[dim-i-1] = FFTW_REDFT00;
              localDCTDSTCells[i] = localDCTCells[i];
            }
            else
            {
              odd[i] = true;
              k[dim-i-1] = FFTW_RODFT00;
              if (i != dim-1)
                localDCTDSTCells[i] = localDCTCells[i] - 2;
            }

            type /= 2;
          }
        }

        /**
         * @brief Number of extended field entries stored on this processor
         *
         * This is the number of extended field entries that are stored.
         * The backend treats each combination of even and odd symmetry
         * separately, and only ever stores the part that actually needs
         * to be stored. This function returns the number of entries that
         * are needed for this, which is the number of entries when all
         * the domain boundaries have even symmetry.
         *
         * @return number of local degrees of freedom
         */
        Index localFieldSize() const
        {
          return localDCTDomainSize;
        }

        /**
         * @brief Number of entries per dim on this processor
         *
         * This is the number of cells per dimension of the
         * extended domain, or the number of cells per dimension
         * for the local part of the extended domain in the case of
         * parallel data distribution. Only half the array per
         * dimension is stored, since the other half contains
         * the same values in reversed order, with the same sign in
         * the case of even symmetry, else with the opposite sign.
         * One additional entry per dimension is needed for the symmetry
         * axis.
         *
         * @return tuple of local cells per dimension
         */
        const Indices& localFieldCells() const
        {
          return localDCTCells;
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
            fieldData = FFTW<RF>::alloc_real(allocLocal);
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
        void transposeIfNeeded()
        {
          transposeIfNeeded(localN0Trans,local0StartTrans);
        }

        /**
         * @brief Transform into Fourier (i.e., frequency) space
         *
         * Perform a forward Fourier transform, mapping from the original
         * domain to the frequency domain. Uses a single FFTW real-to-real
         * DFT transform corresponding to the configured type of symmetry.
         */
        void forwardTransform()
        {
          toFFTWCompatible();

          unsigned int flags;
          if ((*traits).config.template get<bool>("fftw.measure",false))
            flags = FFTW_MEASURE;
          else
            flags = FFTW_ESTIMATE;
          if (transposed)
            flags |= FFTW_MPI_TRANSPOSED_OUT;

          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            if (k[i] == FFTW_RODFT00)
              n[i] = extendedCells[dim-1-i]/2 - 1;
            else
              n[i] = extendedCells[dim-1-i]/2 + 1;

          Index shiftIn = 0;
          if (odd[dim-1])
          {
            shiftIn = 1;
            for (unsigned int i = 0; i < dim - 1; i++)
              shiftIn *= localDCTDSTCells[i];
          }

          Index shiftOut = 0;
          if (!transposed)
            shiftOut = shiftIn;
          else if (odd[dim-2])
          {
            shiftOut = 1;
            for (unsigned int i = 0; i < dim - 2; i++)
              shiftOut *= localDCTDSTCells[i];
            if (odd[dim-1])
              shiftOut *= dctCells[dim-1] - 2;
            else
              shiftOut *= dctCells[dim-1];
          }

          ptrdiff_t blockSizeIn = (extendedCells[dim-1]/2 + 1)/commSize + 1;
          ptrdiff_t blockSizeOut;
          if (transposed)
            blockSizeOut = (extendedCells[dim-2]/2 + 1)/commSize + 1;
          else
            blockSizeOut = (extendedCells[dim-1]/2 + 1)/commSize + 1;

          typename FFTW<RF>::plan plan_forward = FFTW<RF>::mpi_plan_many_r2r(dim,n,1,blockSizeIn,
              blockSizeOut,fieldData + shiftIn,fieldData + shiftIn,(*traits).comm,k,flags);

          if (plan_forward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create forward plan");

          FFTW<RF>::execute(plan_forward);
          FFTW<RF>::destroy_plan(plan_forward);

          for (Index i = 0; i < allocLocal; i++)
            fieldData[i] /= extendedDomainSize;

          if (shiftIn > shiftOut)
          {
            const Index diff = shiftIn - shiftOut;
            for (Index index = 0; index < localDCTDomainSize; index++)
              fieldData[index] = fieldData[index + diff];
          }
          else if (shiftIn < shiftOut)
          {
            const Index diff = shiftOut - shiftIn;
            for (Index index = localDCTDomainSize - 1; index < localDCTDomainSize; index--)
              fieldData[index + diff] = fieldData[index];
          }

          transposeIfNeeded(localN0Trans,local0StartTrans);

          fromFFTWCompatible();
        }

        /**
         * @brief Transform from Fourier (i.e., frequency) space
         *
         * Perform a backward Fourier transform, mapping from the frequency
         * domain back to the original domain. Uses a single FFTW real-to-real
         * DFT transform corresponding to the configured type of symmetry.
         */
        void backwardTransform()
        {
          toFFTWCompatible();

          transposeIfNeeded(localN0,local0Start);

          unsigned int flags;
          if ((*traits).config.template get<bool>("fftw.measure",false))
            flags = FFTW_MEASURE;
          else
            flags = FFTW_ESTIMATE;
          if (transposed)
            flags |= FFTW_MPI_TRANSPOSED_IN;

          ptrdiff_t n[dim];
          for (unsigned int i = 0; i < dim; i++)
            if (k[i] == FFTW_RODFT00)
              n[i] = extendedCells[dim-1-i]/2 - 1;
            else
              n[i] = extendedCells[dim-1-i]/2 + 1;

          Index shiftOut = 0;
          if (odd[dim-1])
          {
            shiftOut = 1;
            for (unsigned int i = 0; i < dim - 1; i++)
              shiftOut *= localDCTDSTCells[i];
          }

          Index shiftIn = 0;
          if (!transposed)
            shiftIn = shiftOut;
          else if (odd[dim-2])
          {
            shiftIn = 1;
            for (unsigned int i = 0; i < dim - 2; i++)
              shiftIn *= localDCTDSTCells[i];
            if (odd[dim-1])
              shiftIn *= dctCells[dim-1] - 2;
            else
              shiftIn *= dctCells[dim-1];
          }

          ptrdiff_t blockSizeOut = (extendedCells[dim-1]/2 + 1)/commSize + 1;
          ptrdiff_t blockSizeIn;
          if (transposed)
            blockSizeIn = (extendedCells[dim-2]/2 + 1)/commSize + 1;
          else
            blockSizeIn = (extendedCells[dim-1]/2 + 1)/commSize + 1;

          typename FFTW<RF>::plan plan_backward = FFTW<RF>::mpi_plan_many_r2r(dim,n,1,blockSizeIn,
              blockSizeOut,fieldData + shiftIn,fieldData + shiftIn,(*traits).comm,k,flags);

          if (plan_backward == nullptr)
            DUNE_THROW(Dune::Exception, "failed to create backward plan");

          FFTW<RF>::execute(plan_backward);
          FFTW<RF>::destroy_plan(plan_backward);

          if (shiftIn > shiftOut)
          {
            const Index diff = shiftIn - shiftOut;
            for (Index index = 0; index < localDCTDomainSize; index++)
              fieldData[index] = fieldData[index + diff];
          }
          else if (shiftIn < shiftOut)
          {
            const Index diff = shiftOut - shiftIn;
            for (Index index = localDCTDomainSize - 1; index < localDCTDomainSize; index--)
              fieldData[index + diff] = fieldData[index];
          }

          fromFFTWCompatible();
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
         * @brief Set entry based on random number
         */
        void set(const Index index, const Indices& indices, RF lambda, RF rand)
        {
          unsigned int border = 0;
          for (unsigned int i = 0; i < dim; i++)
            if (indices[i] + localDCTOffset[i] == 0
                || indices[i] + localDCTOffset[i] == dctCells[i] - 1)
            {
              if (odd[i])
              {
                fieldData[index] = 0;
                return;
              }

              border++;
            }

          static const RF scale = 1./std::sqrt(1 << dim);

          fieldData[index] = scale * lambda * rand;

          if (border > 0)
            fieldData[index] *= std::sqrt(1 << border);
        }

        /**
         * @brief Set entry, but take even/odd component boundary conditions into account
         */
        void setComponent(const Index index, RF val)
        {
          Traits::indexToIndices(index,indices,localDCTCells);

          unsigned int border = 0;
          for (unsigned int i = 0; i < dim; i++)
            if (indices[i] + localDCTOffset[i] == 0
                || indices[i] + localDCTOffset[i] == dctCells[i] - 1)
            {
              if (odd[i])
              {
                fieldData[index] = 0;
                return;
              }

              border++;
            }

          fieldData[index] = val / (1 << (dim - border));
        }

        /**
         * @brief Return entry
         */
        RF get(const Index index) const
        {
          return fieldData[index];
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
          fieldData[index] *= lambda;
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
            fieldData = FFTW<RF>::alloc_real(allocLocal);

          for (Index index = 0; index < localDCTDomainSize; index++)
              fieldData[index] = 0.;

          if (commSize == 1)
          {
            Index index, dctIndex;
            metaForLoopToExtended(index,dctIndex,&(field[0]),fieldData);
          }
          else
          {
            Index fieldSize = localCells[dim-1];
            for (unsigned int i = 0; i < dim-1; i++)
              fieldSize *= localDCTCells[i];
            std::vector<RF> localCopy (fieldSize,0.);

            Index index, dctIndex;
            metaForLoopToExtended(index,dctIndex,&(field[0]),&(localCopy[0]));

            std::vector<MPI_Request> request(2);
            for (unsigned int i = 0; i < 2; i++)
              request[i] = MPI_REQUEST_NULL;

            unsigned int blockSize = (extendedCells[dim-1]/2 + 1)/commSize + 1;

            for (unsigned int i = 0; i < unsigned(commSize); i++)
            {
              if (localOffset[dim-1] >= i * blockSize && localOffset[dim-1] < (i+1) * blockSize)
              {
                if (localCells[dim-1] > (i+1) * blockSize - localOffset[dim-1])
                {
                  Index sendSize1 = (i+1) * blockSize - localOffset[dim-1];
                  for (unsigned int j = 0; j < dim-1; j++)
                    sendSize1 *= localDCTCells[j];

                  MPI_Isend(&(localCopy[0]), sendSize1, mpiType<RF>,
                    i, toExtended, (*traits).comm, &request[0]);

                  Index sendSize2 = localCells[dim-1] - ((i+1) * blockSize - localOffset[dim-1]);
                  for (unsigned int j = 0; j < dim-1; j++)
                    sendSize2 *= localDCTCells[j];

                  MPI_Isend(&(localCopy[0]) + sendSize1, sendSize2, mpiType<RF>,
                    i+1, toExtended, (*traits).comm, &request[1]);
                }
                else
                {
                  Index sendSize = localCells[dim-1];
                  for (unsigned int j = 0; j < dim-1; j++)
                    sendSize *= localDCTCells[j];

                  MPI_Isend(&(localCopy[0]), sendSize, mpiType<RF>,
                    i, toExtended, (*traits).comm, &request[0]);
                }
              }
            }

            std::vector<std::pair<Index,unsigned int>> borders;
            for (unsigned int i = 1; i < unsigned(commSize + 1); i++)
              if (i * localCells[dim-1] >= localDCTOffset[dim-1]
                  && i * localCells[dim-1] < localDCTOffset[dim-1] + localDCTCells[dim-1])
                borders.push_back({i * localCells[dim-1] - localDCTOffset[dim-1],i-1});

            if (!borders.empty())
            {
              Index sendSize1 = borders[0].first;
              for (unsigned int j = 0; j < dim-1; j++)
                sendSize1 *= localDCTCells[j];

              MPI_Recv(fieldData, sendSize1, mpiType<RF>,
                borders[0].second, toExtended, (*traits).comm, MPI_STATUS_IGNORE);

              Index offset = sendSize1;

              for (unsigned int i = 1; i < borders.size(); i++)
              {
                Index sendSize2 = borders[i].first - borders[i-1].first;
                for (unsigned int j = 0; j < dim-1; j++)
                  sendSize2 *= localDCTCells[j];

                MPI_Recv(fieldData + offset, sendSize2, mpiType<RF>,
                  borders[i].second, toExtended, (*traits).comm, MPI_STATUS_IGNORE);

                offset += sendSize2;
              }

              if (borders[borders.size()-1].second + 1 < unsigned(commSize))
              {
                Index sendSize3 = localDCTCells[dim-1] - borders[borders.size()-1].first;
                for (unsigned int j = 0; j < dim-1; j++)
                  sendSize3 *= localDCTCells[j];

                MPI_Recv(fieldData + offset, sendSize3, mpiType<RF>,
                  borders[borders.size()-1].second + 1, toExtended, (*traits).comm, MPI_STATUS_IGNORE);
              }
            }

            MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);
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
         * @param      additive  add to field if true, else replace it
         */
        void extendedFieldToField(
            std::vector<RF>& field,
            unsigned int component = 0,
            bool additive = false
            ) const
        {
          if (component != 0)
            DUNE_THROW(Dune::Exception,"tried to extract more than one field from DCTDSTFieldBackend");

          field.resize(localDomainSize);

          if (commSize == 1)
          {
            Index index, dctIndex;
            if (additive)
              metaForLoopFromExtendedAdditive(index,dctIndex,&(field[0]),fieldData);
            else
              metaForLoopFromExtended(index,dctIndex,&(field[0]),fieldData);
          }
          else
          {
            Index fieldSize = localCells[dim-1];
            for (unsigned int i = 0; i < dim-1; i++)
              fieldSize *= localDCTCells[i];

            std::vector<RF> localCopy(fieldSize);

            std::vector<std::pair<Index,unsigned int>> borders;
            for (unsigned int i = 1; i < unsigned(commSize + 1); i++)
              if (i * localCells[dim-1] >= localDCTOffset[dim-1]
                  && i * localCells[dim-1] < localDCTOffset[dim-1] + localDCTCells[dim-1])
                borders.push_back({i * localCells[dim-1] - localDCTOffset[dim-1],i-1});

            std::vector<MPI_Request> request(commSize);
            for (unsigned int i = 0; i < unsigned(commSize); i++)
              request[i] = MPI_REQUEST_NULL;

            if (!borders.empty())
            {
              Index sendSize1 = borders[0].first;
              for (unsigned int j = 0; j < dim-1; j++)
                sendSize1 *= localDCTCells[j];

              MPI_Isend(fieldData, sendSize1, mpiType<RF>,
                borders[0].second, fromExtended, (*traits).comm, &request[0]);

              Index offset = sendSize1;

              for (unsigned int i = 1; i < borders.size(); i++)
              {
                Index sendSize2 = borders[i].first - borders[i-1].first;
                for (unsigned int j = 0; j < dim-1; j++)
                  sendSize2 *= localDCTCells[j];

                MPI_Isend(fieldData + offset, sendSize2, mpiType<RF>,
                  borders[i].second, fromExtended, (*traits).comm, &request[i]);

                offset += sendSize2;
              }

              if (borders[borders.size()-1].second + 1 < unsigned(commSize))
              {
                Index sendSize3 = localDCTCells[dim-1] - borders[borders.size()-1].first;
                for (unsigned int j = 0; j < dim-1; j++)
                  sendSize3 *= localDCTCells[j];

                MPI_Isend(fieldData + offset, sendSize3, mpiType<RF>,
                  borders[borders.size()-1].second + 1, fromExtended, (*traits).comm, &request[commSize-1]);
              }
            }

            unsigned int blockSize = (extendedCells[dim-1]/2 + 1)/commSize + 1;

            for (unsigned int i = 0; i < unsigned(commSize); i++)
            {
              if (localOffset[dim-1] >= i * blockSize && localOffset[dim-1] < (i+1) * blockSize)
              {
                if (localCells[dim-1] > (i+1) * blockSize - localOffset[dim-1])
                {
                  Index sendSize1 = (i+1) * blockSize - localOffset[dim-1];
                  for (unsigned int j = 0; j < dim-1; j++)
                    sendSize1 *= localDCTCells[j];

                  MPI_Recv(&(localCopy[0]), sendSize1, mpiType<RF>,
                    i, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);

                  Index sendSize2 = localCells[dim-1] - ((i+1) * blockSize - localOffset[dim-1]);
                  for (unsigned int j = 0; j < dim-1; j++)
                    sendSize2 *= localDCTCells[j];

                  MPI_Recv(&(localCopy[0]) + sendSize1, sendSize2, mpiType<RF>,
                    i+1, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);
                }
                else
                {
                  Index sendSize = localCells[dim-1];
                  for (unsigned int j = 0; j < dim-1; j++)
                    sendSize *= localDCTCells[j];

                  MPI_Recv(&(localCopy[0]), sendSize, mpiType<RF>,
                    i, fromExtended, (*traits).comm, MPI_STATUS_IGNORE);
                }
              }
            }

            MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);

            Index index, dctIndex;
            if (additive)
              metaForLoopFromExtendedAdditive(index,dctIndex,&(field[0]),&(localCopy[0]));
            else
              metaForLoopFromExtended(index,dctIndex,&(field[0]),&(localCopy[0]));
          }
        }

        private:

        /**
         * @brief Get the domain decomposition data of the Fourier transform
         */
        void getDCTData()
        {
          allocLocal = 0;
 
          unsigned int blockSize      = (extendedCells[dim-1]/2 + 1)/commSize + 1;
          unsigned int blockSizeTrans = (extendedCells[dim-2]/2 + 1)/commSize + 1;
 
          // check all even/odd combinations, end with purely even
          for (unsigned int type = (1 << dim) - 1; type < (1 << dim); type--)
          {
            ptrdiff_t n[dim];
            unsigned int typeCopy = type;
            for (unsigned int i = 0; i < dim; i++)
            {
              if (typeCopy % 2 == 0)
                n[i] = extendedCells[dim-1-i]/2 + 1;
              else
                n[i] = extendedCells[dim-1-i]/2 - 1;
 
              typeCopy /= 2;
            }
 
            ptrdiff_t allocLocalComb = FFTW<RF>::mpi_local_size_many_transposed(dim,n,1,blockSize,
                blockSizeTrans,(*traits).comm,&localN0,&local0Start,&localN0Trans,&local0StartTrans);
 
            allocLocal = std::max(allocLocal,allocLocalComb);
          }
 
          // additional storage for shifted data in odd case
          unsigned int allocAdd = 1;
          for (unsigned int i = 0; i < dim - 2; i++)
            allocAdd *= extendedCells[i]/2 + 1;
          if (transposed)
            allocAdd *= std::max(extendedCells[dim-2],extendedCells[dim-1])/2 + 1;
          else
            allocAdd *= extendedCells[dim-2]/2 + 1;
 
          allocLocal += allocAdd;
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
        }

        /**
         * @brief Switch last two dimensions (internal version)
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

            std::swap(odd[dim-1],odd[dim-2]);
            if (odd[dim-2])
              localDCTDSTCells[dim-2] = localDCTCells[dim-2] - 2;
            else
              localDCTDSTCells[dim-2] = localDCTCells[dim-2];
            localDCTDSTCells[dim-1] = localN0;
          }
        }

        /**
         * @brief Recursive helper function for fieldToExtendedField
         */
        template<unsigned int currentDim = dim - 1>
          void metaForLoopToExtended(Index& index, Index& dctIndex, RF* data, RF* dctData)
          {
            if constexpr (currentDim > 0)
            {
              if constexpr (currentDim == dim - 1)
              {
                index = 0;
                dctIndex = 0;
              }

              for (Index i = 0; i < localCells[currentDim]; i++)
                metaForLoopToExtended<currentDim - 1>(index,dctIndex,data,dctData);

              metaSkip<currentDim>(dctIndex);
            }
            else
            {
              for (Index i = 0; i < localCells[currentDim]; i++, index++, dctIndex++)
                dctData[dctIndex] = data[index];

              metaSkip<currentDim>(dctIndex);
            }
          }

        /**
         * @brief Recursive helper function for extendedFieldToField, non-additive version
         */
        template<unsigned int currentDim = dim - 1>
          void metaForLoopFromExtended(Index& index, Index& dctIndex, RF* data, RF* dctData) const
          {
            if constexpr (currentDim > 0)
            {
              if constexpr (currentDim == dim - 1)
              {
                index = 0;
                dctIndex = 0;
              }

              for (Index i = 0; i < localCells[currentDim]; i++)
                metaForLoopFromExtended<currentDim - 1>(index,dctIndex,data,dctData);

              metaSkip<currentDim>(dctIndex);
            }
            else
            {
              for (Index i = 0; i < localCells[currentDim]; i++, index++, dctIndex++)
                data[index] += dctData[dctIndex];

              metaSkip<currentDim>(dctIndex);
            }
          }

        /**
         * @brief Recursive helper function for extendedFieldToField, additive version
         */
        template<unsigned int currentDim = dim - 1>
          void metaForLoopFromExtendedAdditive(Index& index, Index& dctIndex, RF* data, RF* dctData) const
          {
            if constexpr (currentDim > 0)
            {
              if constexpr (currentDim == dim - 1)
              {
                index = 0;
                dctIndex = 0;
              }

              for (Index i = 0; i < localCells[currentDim]; i++)
                metaForLoopFromExtendedAdditive<currentDim - 1>(index,dctIndex,data,dctData);

              metaSkip<currentDim>(dctIndex);
            }
            else
            {
              for (Index i = 0; i < localCells[currentDim]; i++, index++, dctIndex++)
                data[index] += dctData[dctIndex];

              metaSkip<currentDim>(dctIndex);
            }
          }

        /**
         * @brief Eliminate explicit zeros of DST dimensions
         */
        void toFFTWCompatible()
        {
          Index shift = 0;
          if (odd[dim-1] && commSize > 1)
          {
            shift = 1;
            for (unsigned int i = 0; i < dim - 1; i++)
              shift *= localDCTDSTCells[i];
          }

          // remove zeros in all dimensions but last one
          Index index, smallIndex;
          metaForLoopToFFTWCompatible(index,smallIndex);

          // shift last dimension by one cell in parallel case
          if (odd[dim-1] && commSize > 1)
          {
            MPI_Request request = MPI_REQUEST_NULL;

            if (rank > 0)
              MPI_Isend(fieldData, shift, mpiType<RF>,
                  rank-1, toCompatible, (*traits).comm, &request);

            Index localDCTDSTDomainSize = 1;
            for (unsigned int i = 0; i < dim; i++)
              localDCTDSTDomainSize *= localDCTDSTCells[i];

            if (rank < commSize - 1)
              MPI_Recv(fieldData + localDCTDSTDomainSize, shift, mpiType<RF>,
                  rank+1, toCompatible, (*traits).comm, MPI_STATUS_IGNORE);

            MPI_Wait(&request,MPI_STATUS_IGNORE);
          }
        }

        /**
         * @brief Helper function for toFFTWCompatible, iterating over variable number of dimensions
         */
        template<unsigned int currentDim = dim - 1>
        void metaForLoopToFFTWCompatible(Index& index, Index& smallIndex)
        {
          if constexpr (currentDim == dim - 1)
          {
            index = 0;
            smallIndex = 0;

            for (Index i = 0; i < localDCTDSTCells[currentDim]; i++)
              metaForLoopToFFTWCompatible<currentDim - 1>(index,smallIndex);
          }
          else if constexpr (currentDim > 0)
          {
            if (odd[currentDim])
              metaSkip<currentDim>(index);

            for (Index i = 0; i < localDCTDSTCells[currentDim]; i++)
              metaForLoopToFFTWCompatible<currentDim - 1>(index,smallIndex);

            if (odd[currentDim])
              metaSkip<currentDim>(index);
          }
          else
          {
            if (odd[currentDim])
              metaSkip<currentDim>(index);

            for (Index i = 0; i < localDCTDSTCells[currentDim]; i++, index++, smallIndex++)
              fieldData[smallIndex] = fieldData[index];

            if (odd[currentDim])
              metaSkip<currentDim>(index);
          }
        }

        /**
         * @brief Reinsert explicit zeros of DST dimensions
         */
        void fromFFTWCompatible()
        {
          Index shift = 0;
          if (odd[dim-1] && commSize > 1)
          {
            shift = 1;
            for (unsigned int i = 0; i < dim - 1; i++)
              shift *= localDCTDSTCells[i];
          }

          // shift last dimension by one cell in parallel case
          if (odd[dim-1] && commSize > 1)
          {
            MPI_Request request = MPI_REQUEST_NULL;

            Index localDCTDSTDomainSize = 1;
            for (unsigned int i = 0; i < dim; i++)
              localDCTDSTDomainSize *= localDCTDSTCells[i];

            if (rank < commSize - 1)
              MPI_Isend(fieldData + localDCTDSTDomainSize, shift, mpiType<RF>,
                  rank+1, fromCompatible, (*traits).comm, &request);

            if (rank > 0)
              MPI_Recv(fieldData, shift, mpiType<RF>,
                  rank-1, fromCompatible, (*traits).comm, MPI_STATUS_IGNORE);

            MPI_Wait(&request,MPI_STATUS_IGNORE);
          }

          // reinsert zeros in all dimensions but last one
          Index index, smallIndex;
          metaForLoopFromFFTWCompatible(index,smallIndex);

          Index sliceSize = 1;
          for (unsigned int i = 0; i < dim-1; i++)
            sliceSize *= dctCells[i];

          // reinsert zeros in last dimension if odd
          if (odd[dim-1])
          {
            if (rank == 0)
              for (Index index = 0; index < sliceSize; index++)
                fieldData[index] = 0.;
            if (rank == commSize - 1)
              for (Index index = localDCTDomainSize - sliceSize; index < localDCTDomainSize; index++)
                fieldData[index] = 0.;
          }
        }

        /**
         * @brief Helper function for fromFFTWCompatible, iterating over variable number of dimensions
         */
        template<unsigned int currentDim = dim - 1>
        void metaForLoopFromFFTWCompatible(Index& index, Index& smallIndex)
        {
          if constexpr (currentDim == dim - 1)
          {
            Index localDCTDSTDomainSize = 1;
            for (unsigned int i = 0; i < dim; i++)
              localDCTDSTDomainSize *= localDCTDSTCells[i];

            index = localDCTDomainSize - 1;
            smallIndex = localDCTDSTDomainSize - 1;

            for (Index i = localDCTDSTCells[currentDim] - 1; i < localDCTDSTCells[currentDim]; i--)
              metaForLoopFromFFTWCompatible<currentDim - 1>(index,smallIndex);
          }
          else if constexpr (currentDim > 0)
          {
            if (odd[currentDim])
              metaFill<currentDim>(index);

            for (Index i = localDCTDSTCells[currentDim] - 1; i < localDCTDSTCells[currentDim]; i--)
              metaForLoopFromFFTWCompatible<currentDim - 1>(index,smallIndex);

            if (odd[currentDim])
              metaFill<currentDim>(index);
          }
          else
          {
            if (odd[currentDim])
              metaFill<currentDim>(index);

            for (Index i = localDCTDSTCells[currentDim] - 1;
                i < localDCTDSTCells[currentDim];i--, index--, smallIndex--)
              fieldData[index] = fieldData[smallIndex];

            if (odd[currentDim])
              metaFill<currentDim>(index);
          }
        }

        /**
         * @brief Helper function, skip consecutive indices with irrelevant entries
         */
        template<unsigned int currentDim>
          void metaSkip(Index& index) const
          {
            Index skip = 1;
            for (unsigned int i = 0; i < currentDim; i++)
              skip *= localDCTCells[i];

            index += skip;
          }

        /**
         * @brief Helper function, insert consecutive explicit zero entries
         */
        template<unsigned int currentDim>
          void metaFill(Index& index)
          {
            Index skip = 1;
            for (unsigned int i = 0; i < currentDim; i++)
              skip *= localDCTCells[i];

            for (Index i = 0; i < skip; i++)
            {
              fieldData[index] = 0.;
              index--;
            }
          }

      };

  }
}

#endif // DUNE_RANDOMFIELD_DCTDSTFIELDBACKEND_HH
