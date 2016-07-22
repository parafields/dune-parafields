// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_MATRIX_HH
#define	DUNE_RANDOMFIELD_MATRIX_HH

namespace Dune {
  namespace RandomField {

    template<typename T>
      class RandomFieldMatrix
      {

        public:

          typedef T Traits;

        private:

          typedef typename Traits::RF RF;
          enum {dim = Traits::dimDomain};

          const Dune::shared_ptr<Traits> traits;
          const typename Traits::Covariance covariance;

          int rank, commSize;
          unsigned int              level;
          std::vector<RF>           meshsize;
          RF                        variance;
          std::vector<RF>           corrLength;
          unsigned int              cgIterations;

          ptrdiff_t allocLocal, localN0, local0Start;

          std::vector<unsigned int> localCells;
          unsigned int              localDomainSize;
          std::vector<unsigned int> extendedCells;
          unsigned int              extendedDomainSize;
          std::vector<unsigned int> localExtendedCells;
          std::vector<unsigned int> localExtendedOffset;
          unsigned int              localExtendedDomainSize;

          fftw_complex* fftTransformedMatrix;

        public:

          RandomFieldMatrix<Traits>(const Dune::shared_ptr<Traits>& traits_)
            : traits(traits_), covariance(), fftTransformedMatrix(NULL)
          {
            update();
          }

          ~RandomFieldMatrix<Traits>()
          {
            if (fftTransformedMatrix != NULL)
              fftw_free(fftTransformedMatrix);
          }

          /*
           * @brief Calculate internal data after creation or refinement
           */
          void update()
          {
            Dune::Timer timer(false);
            timer.start();

            rank                    = (*traits).rank;
            commSize                = (*traits).commSize;
            level                   = (*traits).level;
            meshsize                = (*traits).meshsize;
            variance                = (*traits).variance;
            corrLength              = (*traits).corrLength;
            cgIterations            = (*traits).cgIterations;
            allocLocal              = (*traits).allocLocal;
            localN0                 = (*traits).localN0;
            local0Start             = (*traits).local0Start;
            localCells              = (*traits).localCells;
            localDomainSize         = (*traits).localDomainSize;
            extendedCells           = (*traits).extendedCells;
            extendedDomainSize      = (*traits).extendedDomainSize;
            localExtendedCells      = (*traits).localExtendedCells;
            localExtendedOffset     = (*traits).localExtendedOffset;
            localExtendedDomainSize = (*traits).localExtendedDomainSize;

            if (fftTransformedMatrix != NULL)
              fftw_free(fftTransformedMatrix);
            fftTransformedMatrix = (fftw_complex*) fftw_malloc( allocLocal * sizeof(fftw_complex) );

            std::vector<RF> coord(dim,0.);
            std::vector<unsigned int> indices(dim,0);

            for (unsigned int index = 0; index < localExtendedDomainSize; index++)
            {
              (*traits).indexToIndices(index,indices,localExtendedCells);

              for (unsigned int i = 0; i < dim; i++)
                coord[i]  = std::min(indices[i] + localExtendedOffset[i], extendedCells[i] - indices[i] - localExtendedOffset[i]) * meshsize[i];

              fftTransformedMatrix[index][0] = covariance(variance, coord, corrLength);
              fftTransformedMatrix[index][1] = 0.;
            }

            forwardTransform(fftTransformedMatrix);

            RF regFactor = (*traits).config.template get<RF>("randomField.regFactor",0.);

            unsigned int small = 0;
            unsigned int negative = 0;
            unsigned int smallNegative = 0;
            for (unsigned int index = 0; index < localExtendedDomainSize; index++)
            {
              if (fftTransformedMatrix[index][0] < 1e-6)
              {
                if (fftTransformedMatrix[index][0] < 1e-10)
                {
                  if (fftTransformedMatrix[index][0] > -1e-10)
                    smallNegative++;
                  else
                    negative++;
                }
                else
                  small++;
              }

              if (fftTransformedMatrix[index][0] < regFactor)
                fftTransformedMatrix[index][0] = regFactor;
            }

            if (rank == 0) std::cout << small << " small, " << smallNegative << " small negative and " << negative << " large negative eigenvalues in covariance matrix" << std::endl;

            timer.stop();
            if (rank == 0) std::cout << "Time for RandomMatrix update " << timer.lastElapsed() << std::endl;
          }

          /**
           * @brief Multiply random field with covariance matrix
           */
          StochasticPart<Traits> operator*(const StochasticPart<Traits>& input) const
          {
            StochasticPart<Traits> output(input);

            multiplyExtended(output.dataVector,output.dataVector);

            output.evalValid = false;

            return output;
          }

          /**
           * @brief Multiply random field with root of covariance matrix (up to boundary effects)
           */
          StochasticPart<Traits> multiplyRoot(const StochasticPart<Traits>& input) const
          {
            StochasticPart<Traits> output(input);

            multiplyRootExtended(output.dataVector,output.dataVector);

            output.evalValid = false;

            return output;
          }

          /**
           * @brief Multiply random field with inverse of covariance matrix
           */
          StochasticPart<Traits> multiplyInverse(const StochasticPart<Traits>& input) const
          {
            StochasticPart<Traits> output(input);

            bool fieldZero = true;
            for (unsigned int i = 0; i < localDomainSize; i++)
              if (std::abs(input.dataVector[i]) > 1e-10)
                fieldZero = false;

            if (!fieldZero)
            {
              multiplyInverseExtended(output.dataVector,output.dataVector);

              innerCG(output.dataVector,input.dataVector);
              output.evalValid = false;
            }

            return output;
          }

          /**
           * @brief Generate random field based on covariance matrix
           */
          void generateField(StochasticPart<Traits>& stochasticPart) const
          {
            // initialize pseudo-random generator
            unsigned int seed = (unsigned int) clock(); // create seed out of the current time
            seed += rank;                               // different seed for each processor
            seed += static_cast<int>(reinterpret_cast<uintptr_t>(&stochasticPart));  // different seeds for different fields
            std::default_random_engine generator(seed);
            std::normal_distribution<RF> normalDist(0.,1.);

            fftw_complex *extendedField;
            extendedField = (fftw_complex*) fftw_malloc(allocLocal * sizeof (fftw_complex));

            RF lambda = 0.;

            for (unsigned int index = 0; index < localExtendedDomainSize; index++)
            {
              lambda = std::sqrt(std::abs(fftTransformedMatrix[index][0]) / extendedDomainSize);

              extendedField[index][0] = lambda * normalDist(generator);
              extendedField[index][1] = lambda * normalDist(generator);
            }

            forwardTransform(extendedField);

            extendedFieldToField(stochasticPart.dataVector,extendedField);
            stochasticPart.evalValid = false;

            fftw_free(extendedField);
          }

          /**
           * @brief Generate uncorrelated random field (i.e. noise)
           */
          void generateUncorrelatedField(StochasticPart<Traits>& stochasticPart) const
          {
            // initialize pseudo-random generator
            unsigned int seed = (unsigned int) clock(); // create seed out of the current time
            seed += rank;                               // different seed for each processor
            seed += static_cast<int>(reinterpret_cast<uintptr_t>(&stochasticPart));  // different seeds for different fields
            std::default_random_engine generator(seed);
            std::normal_distribution<RF> normalDist(0.,1.);

            for (unsigned int index = 0; index < localDomainSize; index++)
              stochasticPart.dataVector[index] = normalDist(generator);

            stochasticPart.evalValid = false;
          }

          /**
           * @brief Create field that represents the local variance
           */
          void setVarianceAsField(StochasticPart<Traits>& stochasticPart) const
          {
            for (unsigned int index = 0; index < localDomainSize; index++)
              stochasticPart.dataVector[index] = variance;

            stochasticPart.evalValid = false;
          }

        private:

          /**
           * @brief Perform a forward Fourier tranform of a vector
           */
          template<typename V>
            void forwardTransform(V& vector) const
            {
              fftw_plan plan_forward;

              if (dim == 3)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1],(ptrdiff_t)extendedCells[2]};
                plan_forward = fftw_mpi_plan_dft_3d(n[2], n[1], n[0], vector, vector, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

              }
              else
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1]};
                plan_forward = fftw_mpi_plan_dft_2d(n[1], n[0], vector, vector, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

              }

              fftw_execute(plan_forward);
              fftw_destroy_plan(plan_forward);
            }

          /**
           * @brief Perform a backward Fourier transform of a vector
           */
          template<typename V>
            void backwardTransform(V& vector) const
            {
              fftw_plan plan_backward;

              if (dim == 3)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1],(ptrdiff_t)extendedCells[2]};
                plan_backward = fftw_mpi_plan_dft_3d(n[2], n[1], n[0], vector, vector, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
              }
              else
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1]};
                plan_backward = fftw_mpi_plan_dft_2d(n[1], n[0], vector, vector, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
              }

              fftw_execute(plan_backward);
              fftw_destroy_plan(plan_backward);

              for (unsigned int i = 0; i < localExtendedDomainSize; i++)
              {
                vector[i][0] /= extendedDomainSize;
                vector[i][1] /= extendedDomainSize;
              }
            }

          /**
           * @brief Inner Conjugate Gradients method for multiplication with inverse
           */
          void innerCG(std::vector<RF>& iter, const std::vector<RF>& solution, bool precondition = true) const
          {
            Dune::Timer timer(false);
            timer.start();

            std::vector<RF> tempSolution = solution;
            std::vector<RF> matrixTimesSolution(iter.size());
            std::vector<RF> matrixTimesIter(iter.size());
            std::vector<RF> residual(iter.size());
            std::vector<RF> precResidual(iter.size());
            std::vector<RF> direction(iter.size());
            std::vector<RF> matrixTimesDirection(iter.size());
            RF scalarProd, scalarProd2, myScalarProd, alphaDenominator, myAlphaDenominator, alpha, beta;

            multiplyExtended(tempSolution,matrixTimesSolution);

            multiplyExtended(iter,matrixTimesIter);

            for (unsigned int i = 0; i < residual.size(); i++)
            {
              residual[i] = solution[i] - matrixTimesIter[i];
            }

            if (precondition)
              multiplyInverseExtended(residual,precResidual);
            else
              precResidual = residual;

            direction = precResidual;

            bool converged = false;
            scalarProd = 0.;
            myScalarProd = 0.;
            for (unsigned int i = 0; i < residual.size(); i++)
              myScalarProd += precResidual[i] * residual[i];
            MPI_Allreduce(&myScalarProd,&scalarProd,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

            scalarProd2 = 0.;
            myScalarProd = 0.;
            for (unsigned int i = 0; i < residual.size(); i++)
              myScalarProd += residual[i] * residual[i];
            MPI_Allreduce(&myScalarProd,&scalarProd2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

            if (std::sqrt(std::abs(scalarProd2)) < 1e-6)
              converged = true;

            RF firstValue = 0., myFirstVal = 0.;
            for (unsigned int i = 0; i < iter.size(); i++)
              myFirstVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
            MPI_Allreduce(&myFirstVal,&firstValue,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

            unsigned int count = 0;
            while(!converged && count < cgIterations)
            {
              multiplyExtended(direction,matrixTimesDirection);

              alphaDenominator = 0., myAlphaDenominator = 0.;
              for (unsigned int i = 0; i < direction.size(); i++)
                myAlphaDenominator += direction[i] * matrixTimesDirection[i];

              MPI_Allreduce(&myAlphaDenominator,&alphaDenominator,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
              alpha = scalarProd / alphaDenominator;

              RF oldValue = 0., myOldVal = 0.;
              for (unsigned int i = 0; i < iter.size(); i++)
                myOldVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
              MPI_Allreduce(&myOldVal,&oldValue,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

              for (unsigned int i = 0; i < iter.size(); i++)
              {
                iter[i]            += alpha * direction[i];
                matrixTimesIter[i] += alpha * matrixTimesDirection[i];
                //residual[i]        -= alpha * matrixTimesDirection[i];
              }

              RF value = 0., myVal = 0.;
              for (unsigned int i = 0; i < iter.size(); i++)
                myVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
              MPI_Allreduce(&myVal,&value,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

              for (unsigned int i = 0; i < residual.size(); i++)
                residual[i] = solution[i] - matrixTimesIter[i];

              if (precondition)
                multiplyInverseExtended(residual,precResidual);
              else
                precResidual = residual;

              beta = 1./scalarProd;
              scalarProd = 0.;
              myScalarProd = 0.;
              for (unsigned int i = 0; i < residual.size(); i++)
                myScalarProd += precResidual[i] * residual[i];

              MPI_Allreduce(&myScalarProd,&scalarProd,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
              beta *= scalarProd;

              for (unsigned int i = 0; i < direction.size(); i++)
                direction[i] = precResidual[i] + beta * direction[i];

              if (value != firstValue)
              {
                if (std::abs(value - oldValue)/std::abs(value - firstValue) < 1e-16)
                  converged = true;
              }

              count++;
            }

            if (rank == 0) std::cout << count << " iterations" << std::endl;

            timer.stop();
            if (rank == 0) std::cout << "Time for RandomMatrix innerCG " << timer.lastElapsed() << std::endl;
          }

          /**
           * @brief Embed a random field in the extended domain
           */
          void fieldToExtendedField(std::vector<RF>& field, fftw_complex *extendedField) const
          {
            for(unsigned int i = 0; i < localExtendedDomainSize; i++)
            {
              extendedField[i][0] = 0.;
              extendedField[i][1] = 0.;
            }

            if (commSize == 1)
            {
              std::vector<unsigned int> indices(dim);
              for (unsigned int index = 0; index < localDomainSize; index++)
              {
                (*traits).indexToIndices(index,indices,localCells);
                const unsigned int extIndex = (*traits).indicesToIndex(indices,localExtendedCells);

                extendedField[extIndex][0] = field[index];
              }
            }
            else
            {
              const int embeddingFactor = (*traits).embeddingFactor;
              MPI_Request request;

              MPI_Isend(&(field[0]), localDomainSize, MPI_DOUBLE, rank/embeddingFactor, 0, MPI_COMM_WORLD, &request);

              if (rank*embeddingFactor < commSize)
              {
                MPI_Status status;
                std::vector<RF> localCopy(localDomainSize);
                std::vector<unsigned int> indices(dim);

                unsigned int receiveSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
                for (unsigned int i = 0; i < receiveSize; i++)
                {
                  MPI_Recv(&(localCopy[0]), localDomainSize, MPI_DOUBLE, rank*embeddingFactor + i,   0, MPI_COMM_WORLD, &status);

                  for (unsigned int index = 0; index < localDomainSize; index++)
                  {
                    (*traits).indexToIndices(index,indices,localCells);
                    const unsigned int offset =  i * localExtendedDomainSize/embeddingFactor;
                    const unsigned int extIndex = (*traits).indicesToIndex(indices,localExtendedCells) + offset;

                    extendedField[extIndex][0] = localCopy[index];
                  }
                }
              }

              MPI_Barrier(MPI_COMM_WORLD);
            }
          }

          /**
           * @brief Restrict an extended random field to the original domain
           */
          void extendedFieldToField(std::vector<RF>& field, fftw_complex *extendedField) const
          {
            for (unsigned int i = 0; i < localDomainSize; i++)
            {
              field[i] = 0.;
            }

            if (commSize == 1)
            {
              std::vector<unsigned int> indices(dim);
              for (unsigned int index = 0; index < localDomainSize; index++)
              {
                (*traits).indexToIndices(index,indices,localCells);
                const unsigned int extIndex = (*traits).indicesToIndex(indices,localExtendedCells);

                field[index] = extendedField[extIndex][0];
              }
            }
            else
            {
              const int embeddingFactor = (*traits).embeddingFactor;
              MPI_Status status;
              std::vector<std::vector<RF> > localCopy;
              std::vector<MPI_Request>      request;

              if (rank*embeddingFactor < commSize)
              {
                unsigned int sendSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
                localCopy.resize(sendSize);
                request.resize(sendSize);
                std::vector<unsigned int> indices(dim);

                for (unsigned int i = 0; i < sendSize; i++)
                {
                  localCopy[i].resize(localDomainSize);
                  for (unsigned int index = 0; index < localDomainSize; index++)
                  {
                    (*traits).indexToIndices(index,indices,localCells);
                    const unsigned int offset =  i * localExtendedDomainSize/embeddingFactor;
                    const unsigned int extIndex = (*traits).indicesToIndex(indices,localExtendedCells);

                    localCopy[i][index] = extendedField[extIndex + offset][0];
                  }

                  MPI_Isend(&(localCopy[i][0]), localDomainSize, MPI_DOUBLE, rank*embeddingFactor + i, 0, MPI_COMM_WORLD, &(request[i]));
                }

                MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE, rank/embeddingFactor, 0, MPI_COMM_WORLD, &status);
              }
              else
              {
                MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE, rank/embeddingFactor, 0, MPI_COMM_WORLD, &status);
              }

              MPI_Barrier(MPI_COMM_WORLD);
            }
          }

          /**
           * @brief Multiply an extended random field with covariance matrix
           */
          void multiplyExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            fftw_complex *extendedField;
            extendedField = (fftw_complex*) fftw_malloc(allocLocal * sizeof (fftw_complex));

            fieldToExtendedField(input,extendedField);
            forwardTransform(extendedField);

            for (unsigned int i = 0; i < localExtendedDomainSize; i++)
            {
              extendedField[i][0] *= fftTransformedMatrix[i][0];
              extendedField[i][1] *= fftTransformedMatrix[i][0];
            }

            backwardTransform(extendedField);
            extendedFieldToField(output,extendedField);

            fftw_free(extendedField);
          }

          /**
           * @brief Multiply an extended random field with root of covariance matrix
           */
          void multiplyRootExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            fftw_complex *extendedField;
            extendedField = (fftw_complex*) fftw_malloc(allocLocal * sizeof (fftw_complex));

            fieldToExtendedField(input,extendedField);
            forwardTransform(extendedField);

            for (unsigned int i = 0; i < localExtendedDomainSize; i++)
            {
              extendedField[i][0] *= std::sqrt(fftTransformedMatrix[i][0]);
              extendedField[i][1] *= std::sqrt(fftTransformedMatrix[i][0]);
            }

            backwardTransform(extendedField);
            extendedFieldToField(output,extendedField);

            fftw_free(extendedField);
          }

          /**
           * @brief Multiply an extended random field with inverse of covariance matrix
           */
          void multiplyInverseExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            fftw_complex *extendedField;
            extendedField = (fftw_complex*) fftw_malloc(allocLocal * sizeof (fftw_complex));

            fieldToExtendedField(input,extendedField);
            forwardTransform(extendedField);

            for (unsigned int i = 0; i < localExtendedDomainSize; i++)
            {
              extendedField[i][0] /= fftTransformedMatrix[i][0];
              extendedField[i][1] /= fftTransformedMatrix[i][0];
            }

            backwardTransform(extendedField);
            extendedFieldToField(output,extendedField);

            fftw_free(extendedField);
          }

      };

  }
}

#endif // DUNE_RANDOMFIELD_MATRIX_HH
