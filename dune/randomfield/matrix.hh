#ifndef DUNE_RANDOMFIELD_MATRIX_HH
#define DUNE_RANDOMFIELD_MATRIX_HH

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

#if HAVE_GSL
#include<gsl/gsl_integration.h>
#endif // HAVE_GSL

#if HAVE_DUNE_NONLINOPT
#include "dune/randomfield/optproblem.hh"
#endif // HAVE_DUNE_NONLINOPT

namespace Dune {
  namespace RandomField {

    // forward declarations
    template<long unsigned int> class DefaultMatrixBackend;
    template<long unsigned int> class DefaultFieldBackend;
    template<long unsigned int> class DefaultRNGBackend;

    struct EmbeddingFailed : public Dune::Exception {};

    /**
     * @brief Covariance matrix for stationary Gaussian random fields
     *
     * This class represents the covariance matrix of a stationary Gaussian
     * random field on a structured grid of arbitrary dimension. Internally,
     * the matrix is stored as the Fourier transform of the covariance matrix
     * belonging to an extended domain, which is a diagonal matrix that can
     * be stored as a vector. Different backends are available for the way
     * the matrix is stored, the way extended fields are represented, and the
     * way random numbers are generated. Custom backends can be used by
     * providing new template template parameters.
     *
     * @tparam MatrixBackend representation of extended covariance matrix
     * @tparam FieldBackend  representation of extended random field
     * @tparam RNGBackend    class that produces normally distributed random numbers
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

          /**
           * @brief Constructor
           *
           * @param traits_ shared pointer to traits class containing parameters
           */
          Matrix(const std::shared_ptr<Traits>& traits_)
            :
              traits(traits_),
              matrixBackend(traits),
              fieldBackend(traits),
              rngBackend(traits),
              spareField(nullptr)
          {
            update();
          }

          /**
           * @brief Destructor
           */
          ~Matrix()
          {
            if (spareField != nullptr)
              delete spareField;
          }

          /*
           * @brief Update internal data after creation or refinement
           *
           * This function forwards the update call to the backends.
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
           *
           * @param input field that should be used as multiplicand
           *
           * @return matrix-vector product with argument
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
           *
           * This function extends the fields with zeros to embed it in the extended
           * domain, multiplies with the square root of the extended covariance matrix,
           * and restricts the result to the original domain. Note that this does
           * produce a multiplication with the square root of the original matrix, only
           * an approximation.
           *
           * @param input field that should be used as multiplicand
           *
           * @return approximate matrix-vector product with matrix root
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
           *
           * This function multiplies a given vector with the inverse of the
           * covariance matrix. This is done via an internal CG method that
           * iteratively solves the corresponding linear system.
           *
           * @param input field that should be used as multiplicand
           *
           * @return matrix-vector product with inverse matrix
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
           *
           * This function fills the extended covariance matrix and then
           * transforms it into Fourier space. If the optional template parameter
           * is not supplied, then the covariance function is selected from an
           * internal list based on the given configuration. A non-void template
           * parameter is used as covariance function, as long as the desired
           * type of covariance has been set as "custom-iso" or "custom-aniso"
           * in the configuration. The user has to ensure that the covariance
           * function is truly symmetrical in each dimension if "custom-iso" is
           * chosen.
           *
           * @tparam Covariance type of custom covariance class, if desired
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
              else if (covariance == "matern")
                fillCovarianceMatrix<MaternCovariance>();
              else if (covariance == "matern32")
                fillCovarianceMatrix<Matern32Covariance>();
              else if (covariance == "matern52")
                fillCovarianceMatrix<Matern52Covariance>();
              else if (covariance == "dampedOscillation")
                fillCovarianceMatrix<DampedOscillationCovariance>();
              else if (covariance == "gammaExponential")
                fillCovarianceMatrix<GammaExponentialCovariance>();
              else if (covariance == "cauchy")
                fillCovarianceMatrix<CauchyCovariance>();
              else if (covariance == "generalizedCauchy")
                fillCovarianceMatrix<GeneralizedCauchyCovariance>();
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

            const RF threshold = (*traits).config.template get<RF>("embedding.threshold",1e-14);
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
                if (value < threshold)
                {
                  if (value > -threshold)
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
              DUNE_THROW(EmbeddingFailed,"negative eigenvalues in covariance matrix");
            }

            matrixBackend.finalize();
          }

          /**
           * @brief Generate random field based on covariance matrix
           *
           * This function creates a random field from noise, using the circulant
           * embedding technique. The extended covariance matrix is created automatically
           * if this is the first time the function is called.
           *
           * @param      seed           seed value for random number generation
           * @param[out] stochasticPart resulting random field
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
           * @brief Generate uncorrelated random field (i.e., noise)
           *
           * This is a convenience function that generates white noise,
           * i.e., uncorrelated normal random variables distributed
           * across the grid.
           *
           * @param seed           seed value for random number generation
           * @param stochasticPart resulting white noise
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
           *
           * This is a convenience function that returns a constant field,
           * with each entry being the variance. This is helpful, e.g., when
           * computing the posterior variance after Bayesian inversion.
           *
           * @param stochasticPart vector for constant field
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
           *
           * This function fills the extended covarance matrix, scaling the
           * covariance function with the correlation length if the field is
           * isotropic, applying a diagonal scaling matrix to the coordinates if
           * it is axiparallel anisotropic, or applying a general linear coordinate
           * transformation if it is geometrically anisotropic.
           *
           * @tparam Covariance prescribed covariance function
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
           *
           * This function evaluates one of the built-in isotropic covariance functions,
           * using the transformed coordinates, and stores the result in the
           * multidimensional array. Based on config options, it either employs classical
           * circulant embedding, or one of several choices of smooth embedding.
           * Optionally, optimization can be used to search for a positive semidefinite
           * conforming embedding.
           *
           * @tparam Covariance     prescribed covariance function
           * @tparam GeometryMatrix transformation matrix (based on correlation lengths)
           */
          template<typename Covariance, typename GeometryMatrix>
            void computeCovarianceMatrixEntries() const
            {
              matrixBackend.allocate();

              const std::string& periodization
                = (*traits).config.template get<std::string>("embedding.periodization","classical");
              const std::string& sigmoid
                = (*traits).config.template get<std::string>("embedding.sigmoid","smoothstep");

              if (periodization == "classical")
              {
                if ((*traits).verbose && rank == 0) std::cout << "classical circulant embedding" << std::endl;

                computeMatrixEntriesWithMirroring<Covariance,GeometryMatrix>();
              }
              else if (periodization == "merge")
              {
                if ((*traits).verbose && rank == 0) std::cout << "merge periodization" << std::endl;

                if (sigmoid == "smooth")
                  computeMatrixEntriesWithMerge<Covariance,GeometryMatrix,SmoothSigmoid>();
                else if (sigmoid == "smoothstep")
                  computeMatrixEntriesWithMerge<Covariance,GeometryMatrix,SmoothstepSigmoid>();
                else
                  DUNE_THROW(Dune::Exception,"embedding.sigmoid must be \"smooth\" or \"smoothstep\"");
              }
              else if (periodization == "fold")
              {
                if ((*traits).verbose && rank == 0) std::cout << "fold periodization" << std::endl;

                if (sigmoid == "fold_smooth")
                  computeMatrixEntriesWithFold<Covariance,GeometryMatrix,SmoothSigmoid>();
                else if (sigmoid == "fold_smoothstep")
                  computeMatrixEntriesWithFold<Covariance,GeometryMatrix,SmoothstepSigmoid>();
                else
                  DUNE_THROW(Dune::Exception,"embedding.sigmoid must be \"smooth\" or \"smoothstep\"");
              }
              else if (periodization == "cofold")
              {
                if ((*traits).verbose && rank == 0) std::cout << "cofold periodization" << std::endl;

                computeMatrixEntriesWithMirroring<Covariance,GeometryMatrix>();

                if (sigmoid == "smooth")
                  modifyMatrixEntriesWithCofold<SmoothSigmoid>();
                else if (sigmoid == "smoothstep")
                  modifyMatrixEntriesWithCofold<SmoothstepSigmoid>();
                else
                  DUNE_THROW(Dune::Exception,"embedding.sigmoid must be \"smooth\" or \"smoothstep\"");
              }
              else
                DUNE_THROW(Dune::Exception,
                    "stochastic.periodization must be \"classical\", \"merge\", \"fold\", or \"cofold\"");

              const std::string& optim
                = (*traits).config.template get<std::string>("embedding.optim","none");

              if (optim == "none")
                return;
#if HAVE_DUNE_NONLINOPT
              else if (optim == "coneopt")
                optimizeMatrixEntriesWithConeOptimization<Covariance,GeometryMatrix>();
              else if (optim == "dualopt")
                optimizeMatrixEntriesWithDualOptimization<Covariance,GeometryMatrix>();
              else
                DUNE_THROW(Dune::Exception,"stochastic.optim must be \"coneopt\" or \"dualopt\"");
#else // HAVE_DUNE_NONLINOPT
              else
                DUNE_THROW(Dune::Exception,"optimization requires dune-nonlinopt");
#endif // HAVE_DUNE_NONLINOPT
            }

          /**
           * @brief Classical circulant embedding
           *
           * This function periodizes the covariance function by simply taking
           * the distance to the closest copy of the origin. In effect, this
           * mirrors the covariance across half the domain in each dimension.
           *
           *
           * @tparam Covariance     prescribed covariance function
           * @tparam GeometryMatrix transformation matrix (based on correlation lengths)
           */
          template<typename Covariance, typename GeometryMatrix>
            void computeMatrixEntriesWithMirroring() const
            {
              GeometryMatrix matrix((*traits).config);

              const Covariance   covariance((*traits).config);
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
           * @brief Smooth embedding based on sigmoid merging
           *
           * This function periodizes the covariance function by merging
           * several copies of it, each tapering off using a sigmoid cutoff
           * function. This ensures that each copy has limited
           * support and doesn't modify points that have a predefined value.
           *
           * @tparam Covariance     prescribed covariance function
           * @tparam GeometryMatrix transformation matrix (based on correlation lengths)
           * @tparam Sigmoid        sigmoid function used for merging
           */
          template<typename Covariance, typename GeometryMatrix, typename Sigmoid>
            void computeMatrixEntriesWithMerge() const
            {
              GeometryMatrix matrix((*traits).config);

              const Covariance covariance((*traits).config);
              Sigmoid sigmoid;

              bool radial;
              const std::string& type = (*traits).config.template get<std::string>("stochastic.sigmoidCombine","radialPlus");
              if (type == "radial" || type == "radialPlus")
                radial = true;
              else if (type == "tensor")
                radial = false;
              else
                DUNE_THROW(Dune::Exception,"sigmoidCombine must be tensor, radial or radialPlus");

              std::array<RF,dim>           trueCoord;
              std::array<RF,dim>           mirrorCoord;
              std::array<RF,dim>           transCoord;
              std::array<unsigned int,dim> indices;

              const RF sigmoidStart = (*traits).config.template get<RF>("embedding.sigmoidStart",1.);
              const RF sigmoidEnd   = (*traits).config.template get<RF>("embedding.sigmoidEnd",(*traits).embeddingFactor - 1.);
              unsigned int recursions = (*traits).config.template get<unsigned int>("embedding.mergeRecursions",99);

              if(recursions == 99)
              {
                if ((*traits).covariance == "matern")
                   recursions = std::floor((*traits).config.template get<RF>("stochastic.maternNu") * 2. + RF(dim)/2.);
                 else if ((*traits).covariance == "gammaExponential")
                   recursions = std::floor((*traits).config.template get<RF>("stochastic.expGamma") + RF(dim)/2.);
                 else if ((*traits).covariance == "generalizedCauchy")
                   recursions = std::floor((*traits).config.template get<RF>("stochastic.cauchyAlpha") + RF(dim)/2.);
                 else if ((*traits).covariance == "exponential")
                   recursions = 2;
                 else if ((*traits).covariance == "matern32")
                   recursions = 4;
                 else if ((*traits).covariance == "matern52")
                   recursions = 6;
                 else if ((*traits).covariance == "cauchy")
                   recursions = 3;
                 else
                   DUNE_THROW(Dune::Exception,"magic recursions value 99 not implemented for choice of covariance");
              }

              RF constValue = 0.;
              if (type == "radialPlus")
              {
                for (unsigned int i = 0; i < dim; i++)
                  trueCoord[i] = extensions[i] * (*traits).embeddingFactor/2.;

                matrix.transform(trueCoord,transCoord);

                constValue = covariance(variance,transCoord);
              }

              for (unsigned int index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                  trueCoord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];

                matrixBackend.set(index,0.);

                RF remainder = 1.;

                for (unsigned int j = 0; j < (1 << dim); j++)
                {
                  mirrorCoord = trueCoord;
                  for (unsigned int i = 0; i < dim; i++)
                    // flip dimension if i-th bit is set
                    if (j & (1 << i))
                      mirrorCoord[i] = extensions[i] * (*traits).embeddingFactor - trueCoord[i];

                  RF dampening = 1.;
                  const RF eps = 1e-10;
                  static const RF sqrtDim = std::sqrt(RF(dim));
                  const unsigned int factor = (*traits).embeddingFactor;
                  if (radial && sqrtDim < factor/2.)
                  {
                    RF radius = 0.;
                    for (unsigned int i = 0; i < dim; i++)
                      radius += (mirrorCoord[i]*mirrorCoord[i]) / (extensions[i]*extensions[i]);
                    radius = std::sqrt(radius);

                    dampening = sigmoid(sigmoidStart * sqrtDim - eps,
                        factor - (factor - sigmoidEnd) * sqrtDim + eps,
                        radius, recursions);
                  }
                  else
                  {
                    for (unsigned int i = 0; i < dim; i++)
                      dampening *= sigmoid(extensions[i] * sigmoidStart - eps,
                          extensions[i] * sigmoidEnd + eps,
                          mirrorCoord[i], recursions);
                  }

                  if (dampening < eps)
                    continue;

                  remainder -= dampening;

                  matrix.transform(mirrorCoord,transCoord);
                  RF value = covariance(variance, transCoord);
                  matrixBackend.set(index,matrixBackend.get(index) + value * dampening);
                }

                matrixBackend.set(index,matrixBackend.get(index) + constValue * remainder);
              }
            }

          /**
           * @brief Smooth embedding based on domain folding
           *
           * This function periodizes the covariance function by restricting
           * the extent of the effective domain: beyond the original domain,
           * coordinates are moved towards the origin, until the become
           * effectively constant in a certain distance. As a consequence, the
           * covariance function itself becomes constant and therefore smooth
           * in those parts of the domain.
           *
           * @tparam Covariance     prescribed covariance function
           * @tparam GeometryMatrix transformation matrix (based on correlation lengths)
           * @tparam Sigmoid        sigmoid function used for smooth max function
           */
          template<typename Covariance, typename GeometryMatrix, typename Sigmoid>
             void computeMatrixEntriesWithFold() const
             {
              GeometryMatrix matrix((*traits).config);

              const Covariance   covariance((*traits).config);

#if HAVE_GSL
              std::array<RF,dim>           coord;
              std::array<RF,dim>           transCoord;
              std::array<unsigned int,dim> indices;

              const RF maxFactor      = (*traits).config.template get<RF>("embedding.foldMaxFactor",1.);
              unsigned int recursions = (*traits).config.template get<unsigned int>("embedding.foldRecursions",99);

              if(recursions == 99)
              {
                if ((*traits).covariance == "matern")
                   recursions = std::floor((*traits).config.template get<RF>("stochastic.maternNu") * 2. + RF(dim)/2.);
                 else if ((*traits).covariance == "gammaExponential")
                   recursions = std::floor((*traits).config.template get<RF>("stochastic.expGamma") + RF(dim)/2.);
                 else if ((*traits).covariance == "generalizedCauchy")
                   recursions = std::floor((*traits).config.template get<RF>("stochastic.cauchyAlpha") + RF(dim)/2.);
                 else if ((*traits).covariance == "exponential")
                   recursions = 2;
                 else if ((*traits).covariance == "matern32")
                   recursions = 4;
                 else if ((*traits).covariance == "matern52")
                   recursions = 6;
                 else if ((*traits).covariance == "cauchy")
                   recursions = 3;
                 else
                   DUNE_THROW(Dune::Exception,"magic recursions value 99 not implemented for choice of covariance");
              }

              RF params[3];
              params[0] = 1.;
              params[1] = maxFactor * 0.5*(*traits).embeddingFactor/std::sqrt(RF(dim));
              params[2] = recursions;

              auto func = [](RF x, void* params){
                const double alpha = ((double*)params)[0];
                const double beta  = ((double*)params)[1];
                const double gamma = ((double*)params)[2];
                Sigmoid sigmoid;
                return sigmoid(alpha,beta,x,unsigned(gamma));
              };

              gsl_function gslFunc;
              gslFunc.function = func;
              gslFunc.params = &params;
              RF error;
              std::size_t evals;

              gsl_error_handler_t* handler = gsl_set_error_handler_off();

              for (unsigned int index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];
                  if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                    coord[i] -= extensions[i] * (*traits).embeddingFactor;
                }

                RF norm = 0.;
                for (unsigned int i = 0; i < dim; i++)
                  norm += (coord[i]*coord[i]) / (extensions[i]*extensions[i]);
                norm = std::sqrt(norm/RF(dim));

                if (norm > 1.)
                {
                  RF val;
                  gsl_integration_qng (&gslFunc, params[0], norm, 1e-6, 1e-3, &val, &error, &evals);
                  val += 1.;

                  for (unsigned int i = 0; i < dim; i++)
                    coord[i] *= val/norm;
                }

                matrix.transform(coord,transCoord);

                matrixBackend.set(index,covariance(variance,transCoord));
              }

              gsl_set_error_handler(handler);
#else // HAVE_GSL
              DUNE_THROW(Dune::Exception,"Fold embedding requires GSL library");
#endif // HAVE_GSL
            }

          /**
           * @brief Smooth embedding based on codomain folding ("cofolding")
           *
           * This function periodizes the covariance function by restricting
           * the extent of the effective codomain: the lower part of the
           * codomain is smoothly moved upwards, so that the covariance
           * function becomes constant, and therefore smooth, in the regions
           * with the lowest function values. All function values encountered
           * on the original domain are preserved, only values below that
           * interval are moved upwards in a smooth fashion.
           *
           * @tparam Sigmoid sigmoid function used for smooth max function
           */
          template<typename Sigmoid>
            void modifyMatrixEntriesWithCofold() const
            {
#if HAVE_GSL
              std::array<RF,dim>           coord;
              std::array<unsigned int,dim> indices;

              RF minConstrained = std::numeric_limits<RF>::max();
              RF maxBorder = - std::numeric_limits<RF>::max();

              for (unsigned int index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];
                  if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                    coord[i] -= extensions[i] * (*traits).embeddingFactor;
                }

                RF norm = 0.;
                for (unsigned int i = 0; i < dim; i++)
                  if (std::abs(coord[i])/extensions[i] > norm)
                    norm = std::abs(coord[i])/extensions[i];

                const RF val = matrixBackend.get(index);

                if (norm <= 1.)
                {
                  if (val < minConstrained)
                    minConstrained = val;
                }
                else
                {
                  for (unsigned int i = 0; i < dim; i++)
                  {
                    if (val > maxBorder)
                      if (0.5 * extensions[i] * (*traits).embeddingFactor - std::abs(coord[i])
                        < extensions[i]/(*traits).extendedCells[i])
                        maxBorder = val;
                  }
                }
              }

              RF params[3];
              const RF maxFactor = (*traits).config.template get<RF>("embedding.cofoldMaxFactor",5.);
              const RF recursion = (*traits).config.template get<RF>("embedding.cofoldRecursions",1.);
              params[0] = maxBorder;
              params[1] = std::min(maxFactor * maxBorder, minConstrained);
              params[2] = recursion;

              auto func = [](RF x, void* params){
                const double alpha = ((double*)params)[0];
                const double beta  = ((double*)params)[1];
                const double gamma = ((double*)params)[2];
                Sigmoid sigmoid;
                return sigmoid(0.,beta-alpha,beta-x,unsigned(gamma));
              };

              gsl_error_handler_t* handler = gsl_set_error_handler_off();

              gsl_function gslFunc;
              gslFunc.function = func;
              gslFunc.params = &params;
              RF error, offset;
              std::size_t evals;

              gsl_integration_qng (&gslFunc, params[0], params[1], 1e-6, 1e-3, &offset, &error, &evals);
              offset = params[1] - offset;

              for (unsigned int index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];
                  if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                    coord[i] -= extensions[i] * (*traits).embeddingFactor;
                }

                RF norm = 0.;
                for (unsigned int i = 0; i < dim; i++)
                  if (std::abs(coord[i])/extensions[i] > norm)
                    norm = std::abs(coord[i])/extensions[i];

                if (norm > 1.)
                {
                  const RF x = matrixBackend.get(index);

                  RF val;
                  if (x < params[0])
                    val = offset;
                  else if (x < params[1])
                  {
                    gsl_integration_qng (&gslFunc, params[0], x, 1e-6, 1e-3, &val, &error, &evals);
                    val += offset;
                  }
                  else
                    val = x;

                  matrixBackend.set(index,val);
                }
              }

              gsl_set_error_handler(handler);
#else // HAVE_GSL
              DUNE_THROW(Dune::Exception,"Cofold embedding requires GSL library");
#endif // HAVE_GSL
            }

#if HAVE_DUNE_NONLINOPT
          /**
           * @brief Conic feasibility problem
           *
           * The set of covariance functions with correct values on the original
           * domain is an affine linear subspace, and the set of covariance
           * functions with non-negative Fourier modes is a cone (the non-negative
           * orthant in Fourier space). This function tries to find a point in
           * the intersection of the linear space and the cone, i.e., a
           * covariance function with prescribed values and non-negative transform.
           *
           * @tparam Covariance     prescribed covariance function
           * @tparam GeometryMatrix transformation matrix (based on correlation lengths)
           */
          template<typename Covariance, typename GeometryMatrix>
            void optimizeMatrixEntriesWithConeOptimization() const
            {
              if ((! std::is_same<MatrixBackend<Traits>,DFTMatrixBackend<Traits>>::value)
                  || (*traits).transposed)
                DUNE_THROW(Dune::Exception,"optimization requires untransposed DFTMatrixBackend");
              if ((*traits).commSize != 1)
                DUNE_THROW(Dune::Exception,"optimization is only implemented for sequential runs");

              bool radial;
              const std::string& type
                = (*traits).config.template get<std::string>("stochastic.sigmoidCombine");
              if (type == "radial")
                radial = true;
              else
                radial = false;

              GeometryMatrix matrix((*traits).config);

              const Covariance             covariance((*traits).config);
              std::array<RF,dim>           coord;
              std::array<RF,dim>           transCoord;
              std::array<unsigned int,dim> indices;
              std::vector<bool>            constrained(matrixBackend.localMatrixSize());

              for (Index index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];
                  if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                    coord[i] -= extensions[i] * (*traits).embeddingFactor;
                }

                RF norm = 0.;
                if (radial)
                {
                  for (unsigned int i = 0; i < dim; i++)
                    norm += (coord[i]*coord[i]) / (extensions[i]*extensions[i]);
                  norm = std::sqrt(norm/RF(dim));
                }
                else
                {
                  for (unsigned int i = 0; i < dim; i++)
                    if (std::abs(coord[i])/extensions[i] > norm)
                      norm = std::abs(coord[i])/extensions[i];
                }

                constrained[index] = (norm <= 1.);
              }

              using Problem = ConeOptimizationProblem<Traits>;

              VectorWrapper<Traits> iter(matrixBackend,(*traits).extendedCells,(*traits).extendedDomainSize,(*traits).comm);
              const RF shift     = (*traits).config.template get<RF>("embedding.projShift");
              const RF threshold = (*traits).config.template get<RF>("embedding.threshold",1e-14);
              Problem problem((*traits).config,iter,constrained,shift,threshold,(*traits).extendedDomainSize,
                  matrixBackend.localMatrixSize(),(*traits).extendedCells,(*traits).comm);

              using Solver = Dune::NonlinOpt::UnconstrainedOptimization<typename Problem::Real, typename Problem::Point>;

              RF absTol = (*traits).config.template get<RF>("embedding.optimAbsTol");
              unsigned int maxStep = (*traits).config.template get<unsigned int>("embedding.optimMaxStep");

              Solver solver((*traits).config);
              solver.template set_stoppingcriterion<Dune::NonlinOpt::GradientMaxNormCriterion>(absTol,1e-12);
              if ((*traits).config.template get<bool>("embedding.useCG"))
                solver.set_cg();
              else if ((*traits).config.template get<bool>("embedding.useGMRES"))
                solver.set_gmres();
              solver.report();
              solver.hard_reset(problem);

              unsigned int i = 0;
              const bool breakIfPositive = (*traits).config.template get<bool>("stochastic.breakIfPositive",true);
              while(i < maxStep)
              {
                const bool converged = solver.step(problem,iter);
                if (converged
                || (problem.negatives() == 0 && breakIfPositive))
                {
                  std::cout << "embeddingFactor: " << (*traits).embeddingFactor
                    << " iterations: " << problem.optimizationStep()
                    << " forward trans: " << problem.forwards()
                    << " backward trans: " << problem.backwards()
                    << " total trans: "  << problem.forwards() + problem.backwards()
                    << std::endl;
                  break;
                }
                i++;
              }

              for (Index index = 0; index < matrixBackend.localMatrixSize(); ++index)
                matrixBackend.set(index,iter.raw()[index][0]);

              for (Index index = 0; index < matrixBackend.localMatrixSize(); ++index)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                if (constrained[index])
                {
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
            }
#endif // HAVE_DUNE_NONLINOPT

#if HAVE_DUNE_NONLINOPT
          /**
           * @brief Conic projection via dual optimization problem
           *
           * The set of covariance functions with correct values on the original
           * domain is an affine linear subspace, and the set of covariance
           * functions with non-negative Fourier modes is a cone (the non-negative
           * orthant in Fourier space). This function tries to project a given
           * covariance function onty the intersection of the linear space and the cone,
           * i.e., find the closest covariance function with prescribed values and
           * non-negative transform. It does so by solving the dual problem, i.e.,
           * maximizing the dual over the set of Lagrange multipliers.
           *
           * @tparam Covariance     prescribed covariance function
           * @tparam GeometryMatrix transformation matrix (based on correlation lengths)
           */
        template<typename Covariance, typename GeometryMatrix>
            void optimizeMatrixEntriesWithDualOptimization() const
            {
              if ((! std::is_same<MatrixBackend<Traits>,DFTMatrixBackend<Traits>>::value)
                  || (*traits).transposed)
                DUNE_THROW(Dune::Exception,"optimization requires untransposed DFTMatrixBackend");
              if ((*traits).commSize != 1)
                DUNE_THROW(Dune::Exception,"optimization is only implemented for sequential runs");

              Covariance covariance((*traits).config);
              GeometryMatrix matrix((*traits).config);

              bool radial;
              const std::string& type = (*traits).config.template get<std::string>("stochastic.sigmoidCombine");
              if (type == "radial")
                radial = true;
              else
                radial = false;

              std::array<RF,dim>           coord;
              std::array<unsigned int,dim> indices;
              std::vector<bool>            constrained(matrixBackend.localMatrixSize());

              for (Index index = 0; index < matrixBackend.localMatrixSize(); index++)
              {
                Traits::indexToIndices(index,indices,matrixBackend.localMatrixCells());

                for (unsigned int i = 0; i < dim; i++)
                {
                  coord[i] = (indices[i] + matrixBackend.localMatrixOffset()[i]) * meshsize[i];
                  if (coord[i] > 0.5 * extensions[i] * (*traits).embeddingFactor)
                    coord[i] -= extensions[i] * (*traits).embeddingFactor;
                }

                RF norm = 0.;
                if (radial)
                {
                  for (unsigned int i = 0; i < dim; i++)
                    norm += (coord[i]*coord[i]) / (extensions[i]*extensions[i]);
                  norm = std::sqrt(norm/RF(dim));
                }
                else
                {
                  for (unsigned int i = 0; i < dim; i++)
                    if (std::abs(coord[i])/extensions[i] > norm)
                      norm = std::abs(coord[i])/extensions[i];
                }

                constrained[index] = (norm <= 1.);
              }

              using Problem = DualOptimizationProblem<Traits>;

              VectorWrapper<Traits> start(matrixBackend,(*traits).extendedCells,(*traits).extendedDomainSize,(*traits).comm);
              VectorWrapper<Traits> bound = start;

              const RF shift     = 0.;
              const RF threshold = (*traits).config.template get<RF>("embedding.threshold",1e-14);
              Problem problem((*traits).config,start,bound,constrained,shift,threshold,(*traits).extendedDomainSize,
                  matrixBackend.localMatrixSize(),(*traits).extendedCells,(*traits).comm);

              using Solver = Dune::NonlinOpt::UnconstrainedOptimization<typename Problem::Real, typename Problem::Point>;
              unsigned int maxStep = (*traits).config.template get<unsigned int>("embedding.optimMaxStep");

              VectorWrapper<Traits> iter = problem.zero();
              Solver solver((*traits).config);
              solver.template set_stoppingcriterion<Dune::NonlinOpt::GradientMaxNormCriterion>(threshold,1e-12);
              if ((*traits).config.template get<bool>("embedding.useCG"))
                solver.set_cg();
              else if ((*traits).config.template get<bool>("embedding.useGMRES"))
                solver.set_gmres();
              solver.report();
              solver.hard_reset(problem);

              unsigned int i = 0;
              while(i < maxStep)
              {
                bool converged = solver.step(problem,iter);
                if (converged)
                  break;

                i++;
              }

              std::cout << "embeddingFactor: " << (*traits).embeddingFactor
                << " iterations: " << problem.optimizationStep()
                << " forward trans: " << problem.forwards() + 1
                << " backward trans: " << problem.backwards() + 1
                << " total trans: "  << problem.forwards() + problem.backwards() + 2
                << std::endl;
            }
#endif // HAVE_DUNE_NONLINOPT

          /**
           * @brief Whether matrix backend and field backend have the same local cell layout
           *
           * This function returns true if the two backends have the same data distribution,
           * i.e., number of cells per dimension, else false. A flat index can be used if the
           * backends have the same layout, avoiding costly index transformations.
           *
           * @return Boolean value, true if layout is the same, else false
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
           *
           * This is a helper function used by multiplyInverse to provide matrix-vector
           * products with the inverse of the covariance matrix.
           *
           * @param[in,out] iter         initial guess, and result of product
           * @param         solution     input vector, righthand side of linear system
           * @param         precondition if true, use inverse of extended matrix as preconditioner
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
           *
           * Helper function that provides matrix-vector multiplication for extended
           * random fields and the extended covariance matrix.
           *
           * @param      input  vector that should be multiplied
           * @param[out] output resulting matrix-vector product
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
              component.update();
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
           *
           * Helper function that provides matrix-vector multiplication for extended
           * random fields and the square root of the extended covariance matrix. In
           * contrast to the multiplication on the original domain, this one is exact,
           * since the root of the extended covariance matrix is explicitly known.
           *
           * @param      input  vector that should be multiplied
           * @param[out] output resulting matrix-vector product
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
              component.update();
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
           *
           * Helper function that provides matrix-vector multiplication for extended
           * random fields and the inverse of the extended covariance matrix.  In
           * contrast to the multiplication on the original domain, this one is exact
           * and does not need any iterative solver, since the inverse of the extended
           * covariance matrix is explicitly known.
           *
           * @param      input  vector that should be multiplied
           * @param[out] output resulting matrix-vector product
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
              component.update();
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

#if HAVE_GSL
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
            using Type = Matrix<T,DCTMatrixBackend,R2CFieldBackend>;
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
