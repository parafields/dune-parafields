// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_MATRIX_HH
#define	DUNE_RANDOMFIELD_MATRIX_HH

#include<vector>
#include<array>
#include<random>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Geometry transformation that is only scaling
     */
    template<typename RF, unsigned int dim>
      class ScaledIdentityMatrix
      {
        RF value;

        public:

        ScaledIdentityMatrix(const Dune::ParameterTree& config)
          : value(config.get<RF>("stochastic.corrLength"))
        {
          value = 1./value;
        }

        void transform(const std::array<RF,dim>& x, std::array<RF,dim>& xTrans)
        {
          for (unsigned int i = 0; i < dim; i++)
            xTrans[i] = value * x[i];
        }
      };

    /**
     * @brief Geometry transformation with different scaling per dimension
     */
    template<typename RF, unsigned int dim>
      class DiagonalMatrix
      {
        std::array<RF,dim> diagonalValues;

        public:

        DiagonalMatrix(const Dune::ParameterTree& config)
          : diagonalValues(config.get<std::array<RF,dim> >("stochastic.corrLength"))
        {
          for (unsigned int i = 0; i < dim; i++)
            diagonalValues[i] = 1./diagonalValues[i];
        }

        void transform(const std::array<RF,dim>& x, std::array<RF,dim>& xTrans)
        {
          for (unsigned int i = 0; i < dim; i++)
            xTrans[i] = diagonalValues[i] * x[i];
        }
      };

    /**
     * @brief General geometry transformation
     */
    template<typename RF, unsigned int dim>
      class GeneralMatrix
      {
        std::array<RF,dim*dim> matrixValues;

        public:

        GeneralMatrix(const Dune::ParameterTree& config)
          : matrixValues(config.get<std::array<RF,dim*dim> >("stochastic.corrLength"))
        {
          std::array<RF,dim*dim> copy(matrixValues);
          if (dim == 3)
          {
            matrixValues[0] = copy[4]*copy[8] - copy[5]*copy[7];
            matrixValues[1] = copy[2]*copy[7] - copy[1]*copy[8];
            matrixValues[2] = copy[1]*copy[5] - copy[2]*copy[4];
            matrixValues[3] = copy[5]*copy[6] - copy[3]*copy[8];
            matrixValues[4] = copy[0]*copy[8] - copy[2]*copy[6];
            matrixValues[5] = copy[2]*copy[3] - copy[0]*copy[5];
            matrixValues[6] = copy[3]*copy[7] - copy[4]*copy[6];
            matrixValues[7] = copy[1]*copy[6] - copy[0]*copy[7];
            matrixValues[8] = copy[0]*copy[4] - copy[1]*copy[3];

            const RF det = copy[0]*(copy[4]*copy[8] - copy[5]*copy[7])
              - copy[1]*(copy[3]*copy[8] - copy[5]*copy[6])
              + copy[2]*(copy[3]*copy[7] - copy[4]*copy[6]);
            for (unsigned int i = 0; i < dim*dim; i++)
              matrixValues[i] /= det;
          }
          else if (dim == 2)
          {
            matrixValues[0] =   copy[3];
            matrixValues[1] = - copy[1];
            matrixValues[2] = - copy[2];
            matrixValues[3] =   copy[0];

            const RF det = copy[0]*copy[3] - copy[1]*copy[2];
            for (unsigned int i = 0; i < dim*dim; i++)
              matrixValues[i] /= det;
          }
          else if (dim == 1)
          {
            matrixValues[0] = 1./matrixValues[0];
          }
          else
            DUNE_THROW(Dune::Exception,"not implemented");
        }

        void transform(const std::array<RF,dim>& x, std::array<RF,dim>& xTrans)
        {
          for (unsigned int i = 0; i < dim; i++)
          {
            xTrans[i] = 0.;
            for (unsigned int j = 0; j < dim; j++)
              xTrans[i] += matrixValues[i*dim+j] * x[j];
          }
        }
      };

    /**
     * @brief Spherical covariance function
     */
    class SphericalCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            if (dim == 3)
            {
              if (h_eff > 1.)
                return 0.;
              else
                return variance * (1. - 1.5 * h_eff + 0.5 * std::pow(h_eff, 3));
            }
            else if (dim == 2)
            {
              if (h_eff > 1.)
                return 0.;
              else
                return variance * (1. - 2./M_PI*(h_eff*std::sqrt(1.-std::pow(h_eff,2)) + std::asin(h_eff)));
            }
            else if (dim == 1)
            {
              if (h_eff > 1.)
                return 0.;
              else
                return variance * (1. - h_eff);
            }
            else
              DUNE_THROW(Dune::Exception,"spherical covariance only defined for 1D, 2D and 3D");
          }
    };

    /**
     * @brief Exponential covariance function
     */
    class ExponentialCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * std::exp(-h_eff);
          }
    };

    /**
     * @brief Gaussian covariance function
     */
    class GaussianCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * std::exp(-h_eff * h_eff);
          }
    };

    /**
     * @brief Separable exponential covariance function
     */
    class SeparableExponentialCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += std::abs(x[i]);
            RF h_eff = sum;

            return variance * std::exp(-h_eff);
          }
    };

    /**
     * @brief Matern covariance function with nu = 3/2
     */
    class Matern32Covariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * (1. + std::sqrt(3.)*h_eff) * std::exp(-std::sqrt(3.) * h_eff);
          }
    };

    /**
     * @brief Matern covariance function with nu = 5/2
     */
    class Matern52Covariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * (1. + std::sqrt(5.)*h_eff + 5./3.*h_eff*h_eff) * std::exp(-std::sqrt(5.) * h_eff);
          }
    };

    /**
     * @brief Damped oscillation covariance function
     */
    class DampedOscillationCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            if (dim == 3)
              return variance * std::exp(-h_eff) * std::cos(h_eff/std::sqrt(3.));
            else
              return variance * std::exp(-h_eff) * std::cos(h_eff);
          }
    };

    /**
     * @brief Cauchy covariance function
     */
    class CauchyCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * std::pow(1. + std::pow(h_eff,2),-3);
          }
    };

    /**
     * @brief Cubic covariance function
     */
    class CubicCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            if (dim == 2 || dim == 1)
            {
              if (h_eff > 1.)
                return 0.;
              else
                return variance * (1. - 7. * std::pow(h_eff,2) + 8.75 * std::pow(h_eff,3) - 3.5 * std::pow(h_eff,5) + 0.75 * std::pow(h_eff,7));
            }
            else
              DUNE_THROW(Dune::Exception,"cubic covariance only applicable in 1D or 2D");
          }
    };

    /**
     * @brief White noise covariance function
     */
    class WhiteNoiseCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            for(unsigned int i = 0; i < dim; i++)
            {
              if (std::abs(x[i]) > 1e-10)
                return 0.;
            }

            return variance;
          }
    };

    template<typename T>
      class RandomFieldMatrix
      {

        public:

          typedef T Traits;

        private:

          typedef typename Traits::RF RF;
          enum {dim = Traits::dimDomain};

          const Dune::shared_ptr<Traits> traits;

          int rank, commSize;
          std::array<RF,dim>        extensions;
          unsigned int              level;
          std::array<RF,dim>        meshsize;
          RF                        variance;
          std::vector<RF>           corrLength;
          std::string               covariance;
          unsigned int              cgIterations;

          ptrdiff_t allocLocal, localN0, local0Start;

          std::array<unsigned int,dim> localCells;
          unsigned int                 localDomainSize;
          std::array<unsigned int,dim> extendedCells;
          unsigned int                 extendedDomainSize;
          std::array<unsigned int,dim> localExtendedCells;
          std::array<unsigned int,dim> localExtendedOffset;
          unsigned int                 localExtendedDomainSize;

          mutable fftw_complex* fftTransformedMatrix;

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

            if (fftTransformedMatrix != NULL)
              fillTransformedMatrix();
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
            if (fftTransformedMatrix == NULL)
              fillTransformedMatrix();

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
           * @brief Compute entries of Fourier-transformed covariance matrix
           */
          void fillTransformedMatrix() const
          {
            if (fftTransformedMatrix != NULL)
              fftw_free(fftTransformedMatrix);
            fftTransformedMatrix = (fftw_complex*) fftw_malloc( allocLocal * sizeof(fftw_complex) );

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

            unsigned int small = 0;
            unsigned int negative = 0;
            unsigned int smallNegative = 0;
            RF smallest = std::numeric_limits<RF>::max();
            for (unsigned int index = 0; index < localExtendedDomainSize; index++)
            {
              if (fftTransformedMatrix[index][0] < smallest)
                smallest = fftTransformedMatrix[index][0];

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

              if (fftTransformedMatrix[index][0] < 0.)
                fftTransformedMatrix[index][0] = 0.;
            }

            if ((*traits).verbose && rank == 0) std::cout << small << " small, " << smallNegative << " small negative and " << negative << " large negative eigenvalues in covariance matrix, smallest " << smallest << std::endl;
          }

          template<typename Covariance>
            void fillCovarianceMatrix() const
            {
              if ((*traits).config.template get<std::string>("stochastic.anisotropy","none") == "none")
                computeCovarianceMatrixEntries<Covariance,ScaledIdentityMatrix<RF,dim> >();
              else if ((*traits).config.template get<std::string>("stochastic.anisotropy") == "axiparallel")
                computeCovarianceMatrixEntries<Covariance,DiagonalMatrix<RF,dim> >();
              else if ((*traits).config.template get<std::string>("stochastic.anisotropy") == "geometric")
                computeCovarianceMatrixEntries<Covariance,GeneralMatrix<RF, dim> >();
              else
                DUNE_THROW(Dune::Exception,"stochastic.anisotropy must be \"none\", \"axiparallel\" or \"geometric\"");
            }

          template<typename Covariance, typename GeometryMatrix>
            void computeCovarianceMatrixEntries() const
            {
              GeometryMatrix matrix((*traits).config);

              const Covariance             covariance;
              std::array<RF,dim>           coord;
              std::array<RF,dim>           transCoord;
              std::array<unsigned int,dim> indices;

              for (unsigned int index = 0; index < localExtendedDomainSize; index++)
              {
                (*traits).indexToIndices(index,indices,localExtendedCells);

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i]  = (indices[i] + localExtendedOffset[i]) * meshsize[i];
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

              if (dim == 3)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1],(ptrdiff_t)extendedCells[2]};
                plan_forward = fftw_mpi_plan_dft_3d(n[2], n[1], n[0], vector, vector, (*traits).comm, FFTW_FORWARD, FFTW_ESTIMATE);
              }
              else if (dim == 2)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1]};
                plan_forward = fftw_mpi_plan_dft_2d(n[1], n[0], vector, vector, (*traits).comm, FFTW_FORWARD, FFTW_ESTIMATE);
              }
              else if (dim == 1)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0]};
                plan_forward = fftw_mpi_plan_dft_1d(n[0], vector, vector, (*traits).comm, FFTW_FORWARD, FFTW_ESTIMATE);
              }
              else
                DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

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
                plan_backward = fftw_mpi_plan_dft_3d(n[2], n[1], n[0], vector, vector, (*traits).comm, FFTW_BACKWARD, FFTW_ESTIMATE);
              }
              else if (dim == 2)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1]};
                plan_backward = fftw_mpi_plan_dft_2d(n[1], n[0], vector, vector, (*traits).comm, FFTW_BACKWARD, FFTW_ESTIMATE);
              }
              else if (dim == 1)
              {
                ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0]};
                plan_backward = fftw_mpi_plan_dft_1d(n[0], vector, vector, (*traits).comm, FFTW_BACKWARD, FFTW_ESTIMATE);
              }
              else
                DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

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
            for(unsigned int i = 0; i < localExtendedDomainSize; i++)
            {
              extendedField[i][0] = 0.;
              extendedField[i][1] = 0.;
            }

            if (commSize == 1)
            {
              std::array<unsigned int,dim> indices;
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

              MPI_Isend(&(field[0]), localDomainSize, MPI_DOUBLE, rank/embeddingFactor, 0, (*traits).comm, &request);

              if (rank*embeddingFactor < commSize)
              {
                MPI_Status status;
                std::vector<RF> localCopy(localDomainSize);
                std::array<unsigned int,dim> indices;

                unsigned int receiveSize = std::min(embeddingFactor, commSize - rank*embeddingFactor);
                for (unsigned int i = 0; i < receiveSize; i++)
                {
                  MPI_Recv(&(localCopy[0]), localDomainSize, MPI_DOUBLE, rank*embeddingFactor + i,   0, (*traits).comm, &status);

                  for (unsigned int index = 0; index < localDomainSize; index++)
                  {
                    (*traits).indexToIndices(index,indices,localCells);
                    const unsigned int offset =  i * localExtendedDomainSize/embeddingFactor;
                    const unsigned int extIndex = (*traits).indicesToIndex(indices,localExtendedCells) + offset;

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
            for (unsigned int i = 0; i < localDomainSize; i++)
            {
              field[i] = 0.;
            }

            if (commSize == 1)
            {
              std::array<unsigned int,dim> indices;
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
                std::array<unsigned int,dim> indices;

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

                  MPI_Isend(&(localCopy[i][0]), localDomainSize, MPI_DOUBLE, rank*embeddingFactor + i, 0, (*traits).comm, &(request[i]));
                }

                MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE, rank/embeddingFactor, 0, (*traits).comm, &status);
              }
              else
              {
                MPI_Recv(&(field[0]), localDomainSize, MPI_DOUBLE, rank/embeddingFactor, 0, (*traits).comm, &status);
              }

              MPI_Barrier((*traits).comm);
            }
          }

          /**
           * @brief Multiply an extended random field with covariance matrix
           */
          void multiplyExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            if (fftTransformedMatrix == NULL)
              fillTransformedMatrix();

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
            if (fftTransformedMatrix == NULL)
              fillTransformedMatrix();

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
            if (fftTransformedMatrix == NULL)
              fillTransformedMatrix();

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
