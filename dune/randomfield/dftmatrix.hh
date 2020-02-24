// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_DFTMATRIX_HH
#define	DUNE_RANDOMFIELD_DFTMATRIX_HH

#include<string>
#include<vector>
#include<array>

#include <fftw3.h>
#include <fftw3-mpi.h>

#if HAVE_GSL
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#else
#include<random>
#endif // HAVE_GSL

#include"dune/randomfield/covariance.hh"

namespace Dune {
  namespace RandomField {

    template<typename Traits>
      class DFTMatrix
      {
        using StochasticPartType = StochasticPart<Traits>;

        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        const std::shared_ptr<Traits> traits;

        int rank, commSize;
        std::array<RF,dim> extensions;
        unsigned int       level;
        std::array<RF,dim> meshsize;
        RF                 variance;
        std::string        covariance;
        unsigned int       cgIterations;

        ptrdiff_t allocLocal, localN0, local0Start;

        Indices localCells;
        Index   localDomainSize;
        Indices extendedCells;
        Index   extendedDomainSize;
        Indices localExtendedCells;
        Indices localExtendedOffset;
        Index   localExtendedDomainSize;

        mutable fftw_complex* fftTransformedMatrix;

        public:

        DFTMatrix<Traits>(const std::shared_ptr<Traits>& traits_)
          :
            traits(traits_),
            covariance(),
            fftTransformedMatrix(nullptr)
        {
          update();
        }

        ~DFTMatrix<Traits>()
        {
          if (fftTransformedMatrix != nullptr)
            fftw_free(fftTransformedMatrix);
        }

        /*
         * @brief Calculate internal data after creation or refinement
         */
        void update()
        {
          rank                    = (*traits).rank;
          commSize                = (*traits).commSize;
          extensions              = (*traits).extensions;
          level                   = (*traits).level;
          meshsize                = (*traits).meshsize;
          variance                = (*traits).variance;
          covariance              = (*traits).covariance;
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

          if (fftTransformedMatrix != nullptr)
            fillTransformedMatrix();
        }

        /**
         * @brief Multiply random field with covariance matrix
         */
        StochasticPartType operator*(const StochasticPartType& input) const
        {
          StochasticPartType output(input);

          multiplyExtended(output.dataVector,output.dataVector);

          output.evalValid = false;

          return output;
        }

        /**
         * @brief Multiply random field with root of covariance matrix (up to boundary effects)
         */
        StochasticPartType multiplyRoot(const StochasticPartType& input) const
        {
          StochasticPartType output(input);

          multiplyRootExtended(output.dataVector,output.dataVector);

          output.evalValid = false;

          return output;
        }

        /**
         * @brief Multiply random field with inverse of covariance matrix
         */
        StochasticPartType multiplyInverse(const StochasticPartType& input) const
        {
          StochasticPartType output(input);

          bool fieldZero = true;
          for (Index i = 0; i < localDomainSize; i++)
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
        void generateField(unsigned int seed, StochasticPartType& stochasticPart) const
        {
          if (fftTransformedMatrix == nullptr)
            fillTransformedMatrix();

          // initialize pseudo-random generator
          seed += rank; // different seed for each processor
#if HAVE_GSL
          gsl_rng* gslRng = gsl_rng_alloc(gsl_rng_mt19937);
          gsl_rng_set(gslRng,seed);
#else
          std::default_random_engine generator(seed);
          std::normal_distribution<RF> normalDist(0.,1.);
#endif // HAVE_GSL

          fftw_complex* extendedField = fftw_alloc_complex(allocLocal);

          RF lambda = 0.;

          for (Index index = 0; index < localExtendedDomainSize; index++)
          {
            lambda = std::sqrt(std::abs(fftTransformedMatrix[index][0]) / extendedDomainSize);

#if HAVE_GSL
            extendedField[index][0] = lambda * gsl_ran_gaussian_ziggurat(gslRng,1.);
            extendedField[index][1] = lambda * gsl_ran_gaussian_ziggurat(gslRng,1.);
#else
            extendedField[index][0] = lambda * normalDist(generator);
            extendedField[index][1] = lambda * normalDist(generator);
#endif // HAVE_GSL
          }

          forwardTransform(extendedField);

          extendedFieldToField(stochasticPart.dataVector,extendedField);
          stochasticPart.evalValid = false;

          fftw_free(extendedField);
        }

        /**
         * @brief Generate uncorrelated random field (i.e. noise)
         */
        void generateUncorrelatedField(unsigned int seed, StochasticPartType& stochasticPart) const
        {
          // initialize pseudo-random generator
          seed += rank; // different seed for each processor
          std::default_random_engine generator(seed);
          std::normal_distribution<RF> normalDist(0.,1.);

          for (Index index = 0; index < localDomainSize; index++)
            stochasticPart.dataVector[index] = normalDist(generator);

          stochasticPart.evalValid = false;
        }

        /**
         * @brief Create field that represents the local variance
         */
        void setVarianceAsField(StochasticPartType& stochasticPart) const
        {
          for (Index index = 0; index < localDomainSize; index++)
            stochasticPart.dataVector[index] = variance;

          stochasticPart.evalValid = false;
        }

        private:

        /**
         * @brief Compute entries of Fourier-transformed covariance matrix
         */
        void fillTransformedMatrix() const
        {
          if (fftTransformedMatrix != nullptr)
            fftw_free(fftTransformedMatrix);
          fftTransformedMatrix = fftw_alloc_complex(allocLocal);

          if (covariance == "exponential")
            fillCovarianceMatrix<ExponentialCovariance>();
          else if (covariance == "gaussian")
            fillCovarianceMatrix<GaussianCovariance>();
          else if (covariance == "spherical")
            fillCovarianceMatrix<SphericalCovariance>();
          else if (covariance == "separableExponential")
            fillCovarianceMatrix<SeparableExponentialCovariance>();
          else if (covariance == "matern32")
            fillCovarianceMatrix<Matern32Covariance>();
          else if (covariance == "matern52")
            fillCovarianceMatrix<Matern52Covariance>();
          else if (covariance == "dampedOscillation")
            fillCovarianceMatrix<DampedOscillationCovariance>();
          else if (covariance == "cauchy")
            fillCovarianceMatrix<CauchyCovariance>();
          else if (covariance == "cubic")
            fillCovarianceMatrix<CubicCovariance>();
          else if (covariance == "whiteNoise")
            fillCovarianceMatrix<WhiteNoiseCovariance>();
          else
            DUNE_THROW(Dune::Exception,"covariance structure " + covariance + " not known");

          forwardTransform(fftTransformedMatrix);

          unsigned int mySmall = 0;
          unsigned int myNegative = 0;
          unsigned int mySmallNegative = 0;
          RF mySmallest = std::numeric_limits<RF>::max();
          for (Index index = 0; index < localExtendedDomainSize; index++)
          {
            if (fftTransformedMatrix[index][0] < mySmallest)
              mySmallest = fftTransformedMatrix[index][0];

            if (fftTransformedMatrix[index][0] < 1e-6)
            {
              if (fftTransformedMatrix[index][0] < 1e-10)
              {
                if (fftTransformedMatrix[index][0] > -1e-10)
                  mySmallNegative++;
                else
                  myNegative++;
              }
              else
                mySmall++;
            }

            if (fftTransformedMatrix[index][0] < 0.)
              fftTransformedMatrix[index][0] = 0.;
          }
          int small, negative, smallNegative;
          RF smallest;
          MPI_Allreduce(&mySmall,        &small,        1,MPI_INT,MPI_SUM,(*traits).comm);
          MPI_Allreduce(&myNegative,     &negative,     1,MPI_INT,MPI_SUM,(*traits).comm);
          MPI_Allreduce(&mySmallNegative,&smallNegative,1,MPI_INT,MPI_SUM,(*traits).comm);
          MPI_Allreduce(&mySmallest,     &smallest,     1,MPI_DOUBLE,MPI_MIN,(*traits).comm);

          if ((*traits).verbose && rank == 0)
            std::cout << small << " small, " << smallNegative << " small negative and "
              << negative << " large negative eigenvalues in covariance matrix, smallest "
              << smallest << std::endl;

          if (negative > 0 && !(*traits).approximate)
          {
            if (rank == 0)
              std::cerr << "negative eigenvalues in covariance matrix, "
                << "consider increasing embeddingFactor, or alternatively "
                << "allow generation of approximate samples" << std::endl;
            DUNE_THROW(Dune::Exception,"negative eigenvalues in covariance matrix");
          }
        }

        template<typename Covariance>
          void fillCovarianceMatrix() const
          {
            const std::string& anisotropy = (*traits).config.template get<std::string>("stochastic.anisotropy","none");
            if (anisotropy == "none")
              computeCovarianceMatrixEntries<Covariance,ScaledIdentityMatrix<RF,dim>>();
            else if (anisotropy == "axiparallel")
              computeCovarianceMatrixEntries<Covariance,DiagonalMatrix<RF,dim>>();
            else if (anisotropy == "geometric")
              computeCovarianceMatrixEntries<Covariance,GeneralMatrix<RF,dim>>();
            else
              DUNE_THROW(Dune::Exception,
                  "stochastic.anisotropy must be \"none\", \"axiparallel\" or \"geometric\"");
          }

        template<typename Covariance, typename GeometryMatrix>
          void computeCovarianceMatrixEntries() const
          {
            GeometryMatrix matrix((*traits).config);

            const Covariance   covariance;
            std::array<RF,dim> coord;
            std::array<RF,dim> transCoord;
            Indices            indices;

            for (Index index = 0; index < localExtendedDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localExtendedCells);

              for (unsigned int i = 0; i < dim; i++)
              {
                coord[i] = (indices[i] + localExtendedOffset[i]) * meshsize[i];
                if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                  coord[i] -= extensions[i] * (*traits).embeddingFactor;
              }

              matrix.transform(coord,transCoord);

              fftTransformedMatrix[index][0] = covariance(variance, transCoord);
              fftTransformedMatrix[index][1] = 0.;
            }
          }

        /**
         * @brief Perform a forward Fourier tranform of a vector
         */
        template<typename V>
          void forwardTransform(V& vector) const
          {
            fftw_plan plan_forward;

            ptrdiff_t n[dim];
            for (unsigned int i = 0; i < dim; i++)
              n[i] = extendedCells[dim-1-i];

            plan_forward = fftw_mpi_plan_dft(dim,n,vector,vector,
                (*traits).comm, FFTW_FORWARD, FFTW_ESTIMATE);

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

            ptrdiff_t n[dim];
            for (unsigned int i = 0; i < dim; i++)
              n[i] = extendedCells[dim-1-i];

            plan_backward = fftw_mpi_plan_dft(dim,n,vector,vector,
                (*traits).comm, FFTW_BACKWARD, FFTW_ESTIMATE);

            fftw_execute(plan_backward);
            fftw_destroy_plan(plan_backward);

            for (Index i = 0; i < localExtendedDomainSize; i++)
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
          MPI_Allreduce(&myScalarProd,&scalarProd,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);

          scalarProd2 = 0.;
          myScalarProd = 0.;
          for (unsigned int i = 0; i < residual.size(); i++)
            myScalarProd += residual[i] * residual[i];
          MPI_Allreduce(&myScalarProd,&scalarProd2,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);

          if (std::sqrt(std::abs(scalarProd2)) < 1e-6)
            converged = true;

          RF firstValue = 0., myFirstVal = 0.;
          for (unsigned int i = 0; i < iter.size(); i++)
            myFirstVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
          MPI_Allreduce(&myFirstVal,&firstValue,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);

          unsigned int count = 0;
          while(!converged && count < cgIterations)
          {
            multiplyExtended(direction,matrixTimesDirection);

            alphaDenominator = 0., myAlphaDenominator = 0.;
            for (unsigned int i = 0; i < direction.size(); i++)
              myAlphaDenominator += direction[i] * matrixTimesDirection[i];

            MPI_Allreduce(&myAlphaDenominator,&alphaDenominator,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);
            alpha = scalarProd / alphaDenominator;

            RF oldValue = 0., myOldVal = 0.;
            for (unsigned int i = 0; i < iter.size(); i++)
              myOldVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
            MPI_Allreduce(&myOldVal,&oldValue,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);

            for (unsigned int i = 0; i < iter.size(); i++)
            {
              iter[i]            += alpha * direction[i];
              matrixTimesIter[i] += alpha * matrixTimesDirection[i];
              //residual[i]        -= alpha * matrixTimesDirection[i];
            }

            RF value = 0., myVal = 0.;
            for (unsigned int i = 0; i < iter.size(); i++)
              myVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
            MPI_Allreduce(&myVal,&value,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);

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

            MPI_Allreduce(&myScalarProd,&scalarProd,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);
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

          if ((*traits).verbose && rank == 0) std::cout << count << " iterations" << std::endl;
        }

        /**
         * @brief Embed a random field in the extended domain
         */
        void fieldToExtendedField(std::vector<RF>& field, fftw_complex* extendedField) const
        {
          for(Index i = 0; i < localExtendedDomainSize; i++)
          {
            extendedField[i][0] = 0.;
            extendedField[i][1] = 0.;
          }

          if (commSize == 1)
          {
            Indices indices;
            for (Index index = 0; index < localDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localCells);
              const Index extIndex = Traits::indicesToIndex(indices,localExtendedCells);

              extendedField[extIndex][0] = field[index];
            }
          }
          else
          {
            const int embeddingFactor = (*traits).embeddingFactor;
            MPI_Request request;

            MPI_Isend(&(field[0]), localDomainSize, MPI_DOUBLE,
                rank/embeddingFactor, 0, (*traits).comm, &request);

            if (rank*embeddingFactor < commSize)
            {
              MPI_Status status;
              std::vector<RF> localCopy(localDomainSize);
              Indices indices;

              unsigned int receiveSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
              for (unsigned int i = 0; i < receiveSize; i++)
              {
                MPI_Recv(&(localCopy[0]), localDomainSize, MPI_DOUBLE,
                    rank*embeddingFactor + i,   0, (*traits).comm, &status);

                for (Index index = 0; index < localDomainSize; index++)
                {
                  Traits::indexToIndices(index,indices,localCells);
                  const Index offset =  i * localExtendedDomainSize/embeddingFactor;
                  const Index extIndex
                    = Traits::indicesToIndex(indices,localExtendedCells) + offset;

                  extendedField[extIndex][0] = localCopy[index];
                }
              }
            }

            MPI_Barrier((*traits).comm);
          }
        }

        /**
         * @brief Restrict an extended random field to the original domain
         */
        void extendedFieldToField(std::vector<RF>& field, fftw_complex* extendedField) const
        {
          for (Index i = 0; i < localDomainSize; i++)
            field[i] = 0.;

          if (commSize == 1)
          {
            Indices indices;
            for (Index index = 0; index < localDomainSize; index++)
            {
              Traits::indexToIndices(index,indices,localCells);
              const Index extIndex = Traits::indicesToIndex(indices,localExtendedCells);

              field[index] = extendedField[extIndex][0];
            }
          }
          else
          {
            const int embeddingFactor = (*traits).embeddingFactor;
            MPI_Status status;
            std::vector<std::vector<RF>> localCopy;
            std::vector<MPI_Request>     request;

            if (rank*embeddingFactor < commSize)
            {
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
                  const Index offset =  i * localExtendedDomainSize/embeddingFactor;
                  const Index extIndex = Traits::indicesToIndex(indices,localExtendedCells);

                  localCopy[i][index] = extendedField[extIndex + offset][0];
                }

                MPI_Isend(&(localCopy[i][0]), localDomainSize, MPI_DOUBLE,
                    rank*embeddingFactor + i, 0, (*traits).comm, &(request[i]));
              }

              MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE,
                  rank/embeddingFactor, 0, (*traits).comm, &status);
            }
            else
            {
              MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE,
                  rank/embeddingFactor, 0, (*traits).comm, &status);
            }

            MPI_Barrier((*traits).comm);
          }
        }

        /**
         * @brief Multiply an extended random field with covariance matrix
         */
        void multiplyExtended(std::vector<RF>& input, std::vector<RF>& output) const
        {
          if (fftTransformedMatrix == nullptr)
            fillTransformedMatrix();

          fftw_complex* extendedField = fftw_alloc_complex(allocLocal);

          fieldToExtendedField(input,extendedField);
          forwardTransform(extendedField);

          for (Index i = 0; i < localExtendedDomainSize; i++)
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
          if (fftTransformedMatrix == nullptr)
            fillTransformedMatrix();

          fftw_complex* extendedField = fftw_alloc_complex(allocLocal);

          fieldToExtendedField(input,extendedField);
          forwardTransform(extendedField);

          for (Index i = 0; i < localExtendedDomainSize; i++)
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
          if (fftTransformedMatrix == nullptr)
            fillTransformedMatrix();

          fftw_complex* extendedField = fftw_alloc_complex(allocLocal);

          fieldToExtendedField(input,extendedField);
          forwardTransform(extendedField);

          for (Index i = 0; i < localExtendedDomainSize; i++)
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

#endif // DUNE_RANDOMFIELD_DFTMATRIX_HH
