// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_MATRIX_HH
#define	DUNE_RANDOMFIELD_MATRIX_HH

#include<string>
#include<vector>
#include<array>

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "dune/randomfield/covariance.hh"

#include "dune/randomfield/backends/fftwwrapper.hh"

#include "dune/randomfield/backends/dftmatrixbackend.hh"
#include "dune/randomfield/backends/r2cmatrixbackend.hh"
#include "dune/randomfield/backends/dctmatrixbackend.hh"

#include "dune/randomfield/backends/dftfieldbackend.hh"
#include "dune/randomfield/backends/r2cfieldbackend.hh"
#include "dune/randomfield/backends/dctdstfieldbackend.hh"

#include "dune/randomfield/backends/cpprngbackend.hh"
#include "dune/randomfield/backends/gslrngbackend.hh"

namespace Dune {
  namespace RandomField {

    // forward declarations
    template<long unsigned int> class DefaultMatrixBackend;
    template<long unsigned int> class DefaultFieldBackend;
    template<long unsigned int> class DefaultRNGBackend;

    /**
     * @brief Covariance matrix for stationary Gaussian random fields
     */
    template<typename Traits,
      template<typename> class MatrixBackend = DefaultMatrixBackend<Traits::dim>::template Type,
      template<typename> class FieldBackend = DefaultFieldBackend<Traits::dim>::template Type,
      template<typename> class RNGBackend = DefaultRNGBackend<Traits::dim>::template Type>
        class Matrix
        {
          public:

          using MatrixBackendType = MatrixBackend<Traits>;
          using FieldBackendType  = FieldBackend<Traits>;
          using RNGBackendType    = RNGBackend<Traits>;

          using StochasticPartType = StochasticPart<Traits>;

          private:

          using RF      = typename Traits::RF;
          using Index   = typename Traits::Index;
          using Indices = typename Traits::Indices;

          enum {dim = Traits::dim};

          const std::shared_ptr<Traits> traits;

          int rank, commSize;
          std::array<RF,dim> extensions;
          std::array<RF,dim> meshsize;
          RF                 variance;
          std::string        covariance;
          unsigned int       cgIterations;

          mutable MatrixBackend<Traits> matrixBackend;
          mutable FieldBackend<Traits>  fieldBackend;
          mutable RNGBackend<Traits>    rngBackend;

          mutable std::vector<RF>* spareField;

          public:

          Matrix<Traits,MatrixBackend,FieldBackend,RNGBackend>(const std::shared_ptr<Traits>& traits_)
            :
              traits(traits_),
              matrixBackend(traits),
              fieldBackend(traits),
              rngBackend(traits),
              spareField(nullptr)
          {
            update();
          }

          ~Matrix<Traits,MatrixBackend,FieldBackend,RNGBackend>()
          {
            if (spareField != nullptr)
              delete spareField;
          }

          /*
           * @brief Update internal data after creation or refinement
           */
          void update()
          {
            matrixBackend.update();
            fieldBackend.update();

            rank         = (*traits).rank;
            commSize     = (*traits).commSize;
            extensions   = (*traits).extensions;
            meshsize     = (*traits).meshsize;
            variance     = (*traits).variance;
            covariance   = (*traits).covariance;
            cgIterations = (*traits).cgIterations;
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
            for (Index i = 0; i < input.localDomainSize; i++)
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
           * @brief Compute entries of Fourier-transformed covariance matrix
           */
          template<typename Covariance = void>
          void fillTransformedMatrix() const
          {
            if constexpr (std::is_same<Covariance,void>::value)
            {
              if (covariance == "custom-iso" || covariance == "custom-aniso")
                DUNE_THROW(Dune::Exception,
                    "you need to call fillMatrix with your covariance class as parameter");

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
            }
            else
            {
              if (covariance != "custom-iso" && covariance != "custom-aniso")
                DUNE_THROW(Dune::Exception,
                    "you can't fill the matrix explicitly if a default covariance was chosen");

              computeCovarianceMatrixEntries<Covariance,ScaledIdentityMatrix<RF,dim>>();
            }

            matrixBackend.forwardTransform();

            unsigned int mySmall = 0;
            unsigned int myNegative = 0;
            unsigned int myZero = 0;
            RF mySmallest = std::numeric_limits<RF>::max();
            for (Index index = 0; index < matrixBackend.localMatrixSize(); index++)
            {
              const RF value = matrixBackend.get(index);
              if (value < mySmallest)
                mySmallest = value;

              if (value < 1e-6)
              {
                if (value < 1e-10)
                {
                  if (value > -1e-10)
                    myZero++;
                  else
                    myNegative++;
                }
                else
                  mySmall++;
              }

              if (value < 0.)
                matrixBackend.set(index,0.);
            }

            int small, negative, zero;
            RF smallest;
            MPI_Allreduce(&mySmall,   &small,   1,MPI_INT,MPI_SUM,(*traits).comm);
            MPI_Allreduce(&myNegative,&negative,1,MPI_INT,MPI_SUM,(*traits).comm);
            MPI_Allreduce(&myZero,    &zero,    1,MPI_INT,MPI_SUM,(*traits).comm);
            MPI_Allreduce(&mySmallest,&smallest,1,mpiType<RF>,MPI_MIN,(*traits).comm);

            if ((*traits).verbose && rank == 0)
              std::cout << small << " small, " << zero << " approx. zero and "
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

            matrixBackend.finalize();
          }

          /**
           * @brief Generate random field based on covariance matrix
           */
          void generateField(unsigned int seed, StochasticPartType& stochasticPart) const
          {
            if (!matrixBackend.valid())
              fillTransformedMatrix();

            if (!spareField)
            {
              fieldBackend.allocate();

              // initialize pseudo-random generator
              seed += rank; // different seed for each processor
              rngBackend.seed(seed);

              RF lambda = 0.;

              // special version for DCT/DST field backend
              if constexpr(std::is_same<FieldBackend<Traits>,DCTDSTFieldBackend<Traits>>::value)
              {
                static_assert(std::is_same<MatrixBackend<Traits>,DCTMatrixBackend<Traits>>::value,
                    "DCTDSTFieldBackend requires DCTMatrixBackend");

                Indices indices;
                for (unsigned int type = 0; type < (1 << dim); type++)
                {
                  fieldBackend.setType(type);

                  fieldBackend.transposeIfNeeded();

                  for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                  {
                    Traits::indexToIndices(index,indices,fieldBackend.localFieldCells());
                    lambda = std::sqrt(matrixBackend.eval(indices));

                    const RF& rand = rngBackend.sample();

                    fieldBackend.set(index,indices,lambda,rand);
                  }

                  fieldBackend.backwardTransform();

                  fieldBackend.extendedFieldToField(stochasticPart.dataVector,0,type != 0);
                }

                stochasticPart.evalValid = false;
              }
              // general version
              else
              {
                fieldBackend.transposeIfNeeded();

                // raw (flat) index can be used
                if (sameLayout())
                {
                  for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                  {
                    lambda = std::sqrt(matrixBackend.eval(index));

                    const RF& rand1 = rngBackend.sample();
                    const RF& rand2 = rngBackend.sample();

                    fieldBackend.set(index,lambda,rand1,rand2);
                  }
                }
                // matrix and field layout differ, conversion needed
                else
                {
                  Indices indices;
                  for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                  {
                    Traits::indexToIndices(index,indices,fieldBackend.localFieldCells());

                    lambda = std::sqrt(matrixBackend.eval(indices));

                    const RF& rand1 = rngBackend.sample();
                    const RF& rand2 = rngBackend.sample();

                    fieldBackend.set(index,lambda,rand1,rand2);
                  }
                }

                fieldBackend.backwardTransform();

                fieldBackend.extendedFieldToField(stochasticPart.dataVector,0);
                stochasticPart.evalValid = false;

                if (fieldBackend.hasSpareField())
                {
                  spareField = new std::vector<RF>(stochasticPart.dataVector.size());
                  fieldBackend.extendedFieldToField(*spareField,1);
                }

              }
            }
            else
            {
              stochasticPart.dataVector = *spareField;
              delete spareField;
              spareField = nullptr;
            }
          }

          /**
           * @brief Generate uncorrelated random field (i.e. noise)
           */
          void generateUncorrelatedField(
              unsigned int seed,
              StochasticPartType& stochasticPart
              ) const
          {
            // initialize pseudo-random generator
            seed += rank; // different seed for each processor
            std::default_random_engine generator(seed);
            std::normal_distribution<RF> normalDist(0.,1.);

            for (Index index = 0; index < stochasticPart.localDomainSize; index++)
              stochasticPart.dataVector[index] = normalDist(generator);

            stochasticPart.evalValid = false;
          }

          /**
           * @brief Create field that represents the local variance
           */
          void setVarianceAsField(StochasticPartType& stochasticPart) const
          {
            for (Index index = 0; index < stochasticPart.localDomainSize; index++)
              stochasticPart.dataVector[index] = variance;

            stochasticPart.evalValid = false;
          }

          private:

          /**
           * @brief Compute entries of covariance matrix
           */
          template<typename Covariance>
            void fillCovarianceMatrix() const
            {
              const std::string& anisotropy
                = (*traits).config.template get<std::string>("stochastic.anisotropy","none");

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

          /**
           * @brief Evaluate isotropic covariance matrix in (potentially) transformed space
           */
          template<typename Covariance, typename GeometryMatrix>
            void computeCovarianceMatrixEntries() const
            {
              matrixBackend.allocate();

              GeometryMatrix matrix((*traits).config);

              const Covariance   covariance;
              std::array<RF,dim> coord;
              std::array<RF,dim> transCoord;
              Indices            indices;

              for (Index index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];
                  if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                    coord[i] -= extensions[i] * (*traits).embeddingFactor;
                }

                matrix.transform(coord,transCoord);

                matrixBackend.set(index,covariance(variance,transCoord));
              }
            }

          /**
           * @brief Whether matrix backend and field backend have the same local cell layout
           */
          bool sameLayout() const
          {
            for (unsigned int i = 0; i < dim; i++)
              if (matrixBackend.localEvalMatrixCells()[i] != fieldBackend.localFieldCells()[i])
                return false;

            return true;
          }

          /**
           * @brief Inner Conjugate Gradients method for multiplication with inverse
           */
          void innerCG(
              std::vector<RF>& iter,
              const std::vector<RF>& solution,
              bool precondition = true
              ) const
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
            MPI_Allreduce(&myScalarProd,&scalarProd,1,mpiType<RF>,MPI_SUM,(*traits).comm);

            scalarProd2 = 0.;
            myScalarProd = 0.;
            for (unsigned int i = 0; i < residual.size(); i++)
              myScalarProd += residual[i] * residual[i];
            MPI_Allreduce(&myScalarProd,&scalarProd2,1,mpiType<RF>,MPI_SUM,(*traits).comm);

            if (std::sqrt(std::abs(scalarProd2)) < 1e-6)
              converged = true;

            RF firstValue = 0., myFirstVal = 0.;
            for (unsigned int i = 0; i < iter.size(); i++)
              myFirstVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
            MPI_Allreduce(&myFirstVal,&firstValue,1,mpiType<RF>,MPI_SUM,(*traits).comm);

            unsigned int count = 0;
            while(!converged && count < cgIterations)
            {
              multiplyExtended(direction,matrixTimesDirection);

              alphaDenominator = 0., myAlphaDenominator = 0.;
              for (unsigned int i = 0; i < direction.size(); i++)
                myAlphaDenominator += direction[i] * matrixTimesDirection[i];

              MPI_Allreduce(&myAlphaDenominator,&alphaDenominator,1,mpiType<RF>,MPI_SUM,(*traits).comm);
              alpha = scalarProd / alphaDenominator;

              RF oldValue = 0., myOldVal = 0.;
              for (unsigned int i = 0; i < iter.size(); i++)
                myOldVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
              MPI_Allreduce(&myOldVal,&oldValue,1,mpiType<RF>,MPI_SUM,(*traits).comm);

              for (unsigned int i = 0; i < iter.size(); i++)
              {
                iter[i]            += alpha * direction[i];
                matrixTimesIter[i] += alpha * matrixTimesDirection[i];
                //residual[i]        -= alpha * matrixTimesDirection[i];
              }

              RF value = 0., myVal = 0.;
              for (unsigned int i = 0; i < iter.size(); i++)
                myVal += iter[i]*(0.5*matrixTimesIter[i] - solution[i]);
              MPI_Allreduce(&myVal,&value,1,mpiType<RF>,MPI_SUM,(*traits).comm);

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

              MPI_Allreduce(&myScalarProd,&scalarProd,1,mpiType<RF>,MPI_SUM,(*traits).comm);
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
           * @brief Multiply an extended random field with covariance matrix
           */
          void multiplyExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            if (!matrixBackend.valid())
              fillTransformedMatrix();

            fieldBackend.fieldToExtendedField(input);

            // special version for DCT/DST field backend
            if constexpr(std::is_same<FieldBackend<Traits>,DCTDSTFieldBackend<Traits>>::value)
            {
              static_assert(std::is_same<MatrixBackend<Traits>,DCTMatrixBackend<Traits>>::value,
                  "DCTDSTFieldBackend requires DCTMatrixBackend");

              FieldBackend<Traits> component(traits);
              component.allocate();

              for (unsigned int type = 0; type < (1 << dim); type++)
              {
                component.setType(type);

                for (Index index = 0; index < component.localFieldSize(); index++)
                  component.setComponent(index,fieldBackend.get(index));

                component.forwardTransform();

                for (Index index = 0; index < component.localFieldSize(); index++)
                  component.mult(index,matrixBackend.get(index));

                component.backwardTransform();

                component.extendedFieldToField(output,0,type != 0);
              }
            }
            // general version
            else
            {
              fieldBackend.forwardTransform();

              // raw (flat) index can be used
              if (sameLayout())
              {
                for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                  fieldBackend.mult(index,matrixBackend.eval(index));
              }
              // matrix and field layout differ, conversion needed
              else
              {
                Indices indices;
                for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                {
                  Traits::indexToIndices(index,indices,fieldBackend.localFieldCells());

                  fieldBackend.mult(index,matrixBackend.eval(indices));
                }
              }

              fieldBackend.backwardTransform();

              fieldBackend.extendedFieldToField(output);
            }
          }

          /**
           * @brief Multiply an extended random field with root of covariance matrix
           */
          void multiplyRootExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            if (!matrixBackend.valid())
              fillTransformedMatrix();

            fieldBackend.fieldToExtendedField(input);

            // special version for DCT/DST field backend
            if constexpr(std::is_same<FieldBackend<Traits>,DCTDSTFieldBackend<Traits>>::value)
            {
              static_assert(std::is_same<MatrixBackend<Traits>,DCTMatrixBackend<Traits>>::value,
                  "DCTDSTFieldBackend requires DCTMatrixBackend");

              FieldBackend<Traits> component(traits);
              component.allocate();

              for (unsigned int type = 0; type < (1 << dim); type++)
              {
                component.setType(type);

                for (Index index = 0; index < component.localFieldSize(); index++)
                  component.setComponent(index,fieldBackend.get(index));

                component.forwardTransform();

                for (Index index = 0; index < component.localFieldSize(); index++)
                  component.mult(index,std::sqrt(matrixBackend.get(index)));

                component.backwardTransform();

                component.extendedFieldToField(output,0,type != 0);
              }
            }
            // general version
            else
            {
              fieldBackend.forwardTransform();

              // raw (flat) index can be used
              if (sameLayout())
              {
                for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                  fieldBackend.mult(index,std::sqrt(matrixBackend.eval(index)));
              }
              // matrix and field layout differ, conversion needed
              else
              {
                Indices indices;
                for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                {
                  Traits::indexToIndices(index,indices,fieldBackend.localFieldCells());

                  fieldBackend.mult(index,std::sqrt(matrixBackend.eval(indices)));
                }
              }

              fieldBackend.backwardTransform();

              fieldBackend.extendedFieldToField(output);
            }
          }

          /**
           * @brief Multiply an extended random field with inverse of covariance matrix
           */
          void multiplyInverseExtended(std::vector<RF>& input, std::vector<RF>& output) const
          {
            if (!matrixBackend.valid())
              fillTransformedMatrix();

            fieldBackend.fieldToExtendedField(input);

            // special version for DCT/DST field backend
            if constexpr(std::is_same<FieldBackend<Traits>,DCTDSTFieldBackend<Traits>>::value)
            {
              static_assert(std::is_same<MatrixBackend<Traits>,DCTMatrixBackend<Traits>>::value,
                  "DCTDSTFieldBackend requires DCTMatrixBackend");

              FieldBackend<Traits> component(traits);
              component.allocate();

              for (unsigned int type = 0; type < (1 << dim); type++)
              {
                component.setType(type);

                for (Index index = 0; index < component.localFieldSize(); index++)
                  component.setComponent(index,fieldBackend.get(index));

                component.forwardTransform();

                for (Index index = 0; index < component.localFieldSize(); index++)
                  component.mult(index,1./matrixBackend.get(index));

                component.backwardTransform();

                component.extendedFieldToField(output,0,type != 0);
              }
            }
            // general version
            else
            {
              fieldBackend.forwardTransform();

              // raw (flat) index can be used
              if (sameLayout())
              {
                for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                  fieldBackend.mult(index,1./matrixBackend.eval(index));
              }
              // matrix and field layout differ, conversion needed
              else
              {
                Indices indices;
                for (Index index = 0; index < fieldBackend.localFieldSize(); index++)
                {
                  Traits::indexToIndices(index,indices,fieldBackend.localFieldCells());

                  fieldBackend.mult(index,1./matrixBackend.eval(indices));
                }
              }

              fieldBackend.backwardTransform();

              fieldBackend.extendedFieldToField(output);
            }
          }

        };

    /**
     * @brief Default matrix backend for dim > 1
     */
    template<long unsigned int dim>
      class DefaultMatrixBackend
      {
        public:

          template<typename T>
            using Type = R2CMatrixBackend<T>;
      };

    /**
     * @brief Default matrix backend for dim == 1
     */
    template<>
      class DefaultMatrixBackend<1>
      {
        public:

          template<typename T>
            using Type = DFTMatrixBackend<T>;
      };

    /**
     * @brief Default field backend for dim > 1
     */
    template<long unsigned int dim>
      class DefaultFieldBackend
      {
        public:

          template<typename T>
            using Type = R2CFieldBackend<T>;
      };

    /**
     * @brief Default field backend for dim == 1
     */
    template<>
      class DefaultFieldBackend<1>
      {
        public:

          template<typename T>
            using Type = DFTFieldBackend<T>;
      };

    /**
     * @brief Default RNG backend: GSL when available, std::random as fallback
     */
    template<long unsigned int dim>
      class DefaultRNGBackend
      {
        public:

#ifdef HAVE_GSL
          template<typename T>
            using Type = GSLRNGBackend<T>;
#else // HAVE_GSL
          template<typename T>
            using Type = CppRNGBackend<T>;
#endif // HAVE_GSL
      };

    /**
     * @brief Default isotropic matrix selector for nD, n > 1: DCTMatrix
     */
    template<long unsigned int dim>
      class DefaultIsoMatrix
      {
        public:

          template<typename T>
            using Type = Matrix<T,DCTMatrixBackend,DCTDSTFieldBackend>;
      };

    /**
     * @brief Default isotropic matrix selector for 1D: DFTMatrix
     */
    template<>
      class DefaultIsoMatrix<1>
      {
        public:

          template<typename T>
            using Type = Matrix<T,DFTMatrixBackend>;
      };

    /**
     * @brief Default anisotropic matrix selector for nD, n > 1: R2CMatrix
     */
    template<long unsigned int dim>
      class DefaultAnisoMatrix
      {
        public:

          template<typename T>
            using Type = Matrix<T,R2CMatrixBackend>;
      };

    /**
     * @brief Default anisotropic matrix selector for 1D: DFTMatrix
     */
    template<>
      class DefaultAnisoMatrix<1>
      {
        public:

          template<typename T>
            using Type = Matrix<T,DFTMatrixBackend>;
      };
  }
}

#endif // DUNE_RANDOMFIELD_MATRIX_HH
