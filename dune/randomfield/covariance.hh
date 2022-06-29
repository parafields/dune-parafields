#ifndef DUNE_RANDOMFIELD_COVARIANCE_HH
#define DUNE_RANDOMFIELD_COVARIANCE_HH

#include<vector>
#include<array>
#include<random>

#ifdef HAVE_GSL
#include "gsl/gsl_sf_bessel.h"
#endif // HAVE_GSL

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

        /**
         * @brief Constructor reading correlation lengths from config
         *
         * This constructor expects exactly one parameter for the
         * correlation length (because it is isotropic and has the
         * same value in each direction).
         *
         * @param config parameter tree containing correlation lengths
         */
        ScaledIdentityMatrix(const Dune::ParameterTree& config)
          :
            value(config.get<RF>("stochastic.corrLength"))
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

        /**
         * @brief Constructor reading correlation lengths from config
         *
         * This constructor expects one correlation length per dimension.
         *
         * @param config parameter tree containing correlation lengths
         */
        DiagonalMatrix(const Dune::ParameterTree& config)
          :
            diagonalValues(config.get<std::array<RF,dim>>("stochastic.corrLength"))
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

        /**
         * @brief Constructor reading correlation lengths from config
         *
         * This parameter expects a regular \f$ d \times d \f$ matrix as
         * "correlation length", specified row after row and interpreted as a
         * general space transformation.
         *
         * @param config parameter tree containing correlation lengths
         */
        GeneralMatrix(const Dune::ParameterTree& config)
          :
            matrixValues(config.get<std::array<RF,dim*dim>>("stochastic.corrLength"))
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
            DUNE_THROW(Dune::NotImplemented,"dimension > 3 not implemented");
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
     * @brief Smooth (C^\infty) cutoff function
     *
     * This functor class provides a \f$ (C^\infty) \f$ interpolation
     * between a constant value of one on one side of the interval,
     * and a constant value of zero on the other.
     */
    class SmoothSigmoid
    {
      public:

        /**
         * @brief Evaluate the sigmoid function
         *
         * @param oneEdge    interval boundary with value one
         * @param zeroEdge   interval boundary with value zero
         * @param x          evaluation point
         * @param recursions unused dummy argument
         *
         * @return function value at x
         */
        template<typename RF>
          RF operator()(RF oneEdge, RF zeroEdge, RF x, unsigned int recursions = 0) const
          {
            if (std::abs(x) < oneEdge)
              return 1.;
            if (std::abs(x) > zeroEdge)
              return 0.;

            const RF x1 = (zeroEdge - std::abs(x))/(zeroEdge - oneEdge);
            const RF x2 = (std::abs(x) - oneEdge)/(zeroEdge - oneEdge);

            const RF output = theta(x1)/(theta(x1) + theta(x2));

            return output;
          }

      private:

        /**
         * @brief Helper function for sigmoid
         *
         * @param x evaluation point
         *
         * @return helper function value
         */
        template<typename RF>
          RF theta(RF x) const
          {
            if (x > 0.)
              return std::exp(-1./x);
            else
              return 0.;
          }
    };

    /**
     * @brief Sigmoid function based on clamping (linear interpolation)
     *
     * This functor class provides linear interpolation
     * between a constant value of one on one side of the interval,
     * and a constant value of zero on the other.
     */
    class ClampingSigmoid
    {
      public:

        /**
         * @brief Evaluate the sigmoid function
         *
         * @param oneEdge    interval boundary with value one
         * @param zeroEdge   interval boundary with value zero
         * @param x          evaluation point
         * @param recursions unused dummy argument
         *
         * @return function value at x
         */
        template<typename RF>
          RF operator()(RF oneEdge, RF zeroEdge, RF x, unsigned int recursions = 0) const
          {
            if (std::abs(x) < oneEdge)
              return 1.;
            else if (std::abs(x) > zeroEdge)
              return 0.;

            const RF output = (zeroEdge - std::abs(x))/(zeroEdge - oneEdge);

            // clamping is invariant under recursion
            return output;
          }
    };

    /**
     * @brief Sigmoid based on smoothstep function (using Hermite interpolation)
     *
     * This functor class provides \f$ C^k \f$, \f$ k \geq 0 \f$,
     * Hermite interpolation between a constant value of one on one
     * side of the interval, and a constant value of zero on the
     * other. The smoothness of the interpolation is determined through
     * the number of function call recursions, with the number of calls
     * equal to the constant \f$k\f$: this is the number of
     * derivatives matched by the Hermite interpolation.
     */
    class SmoothstepSigmoid
    {
      ClampingSigmoid clamping;

      public:

        /**
         * @brief Evaluate the sigmoid function
         *
         * @param oneEdge    interval boundary with value one
         * @param zeroEdge   interval boundary with value zero
         * @param x          evaluation point
         * @param recursions number of recursions (and smoothness k)
         *
         * @return function value at x
         */
      template<typename RF>
        RF operator()(RF oneEdge, RF zeroEdge, RF x, unsigned int recursions = 0) const
        {
          const RF x1 = clamping(oneEdge,zeroEdge,x);

          const RF x2 = 2.*x1 - 1.;
          RF normalization = 1.;
          RF addition = 1.;
          RF output = 1.;

          for (unsigned int i = 1; i <= recursions; i++)
          {
            output *= 2.*i;
            addition *= 1 - x2 * x2;
            output += addition;
            output /= 2.*i + 1.;
            normalization *= 2.*i/(2.*i+1.);
          }
          output *= x2 / normalization;
          output += 1.;
          output /= 2.;

          return output;
        }
    };

    /**
     * @brief Spherical covariance function
     *
     * The spherical covariance function represents the interaction
     * between two spheres in a certain distance, in terms of overlapping
     * volume of the two spheres. It has compact support.
     */
    class SphericalCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        SphericalCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
                return variance * (1. - 2./M_PI*(h_eff*std::sqrt(1.-std::pow(h_eff,2))
                      + std::asin(h_eff)));
            }
            else if (dim == 1)
            {
              if (h_eff > 1.)
                return 0.;
              else
                return variance * (1. - h_eff);
            }
            else
              DUNE_THROW(Dune::NotImplemented,
                  "spherical covariance only defined for 1D, 2D and 3D");
          }
    };

    /**
     * @brief Exponential covariance function
     *
     * The exponential covariance function is \f$ C(h) = \sigma^2 \exp(-h) \f$,
     * and produces comparatively rough sample paths. This is the Matérn
     * covariance function for \f$ \nu = 1/2 \f$, and the gamma-exponential
     * covariance function for \f$ \gamma = 1 \f$.
     */
    class ExponentialCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        ExponentialCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
     * @brief Gamma-exponential covariance function
     *
     * The gamma-exponential covariance function is
     * \f$ C(h) = \sigma^2 \exp(-h^\gamma) \f$, its sample
     * paths are relatively rough except for \f$ \gamma = 2 \f$
     * (Gaussian covariace function).
     */
    class GammaExponentialCovariance
    {
      const double gamma;

      public:

      /**
       * @brief Constructor
       *
       * @param config configuration used for gamma
       */
      GammaExponentialCovariance(const Dune::ParameterTree& config)
        : gamma(config.template get<double>("stochastic.expGamma"))
      {
        if (gamma < 0. || gamma > 2.)
          DUNE_THROW(Dune::Exception,"exponent gamma has to be between 0 and 2");
      }

      /**
       * @brief Evaluate the covariance function
       *
       * @tparam RF      type of values and coordinates
       * @tparam dim     dimension of domain
       *
       * @param variance covariance for lag zero
       * @param x        location, after scaling / trafo with correlation length
       *
       * @return resulting value
       */
      template<typename RF, long unsigned int dim>
        RF operator()(const RF variance, const std::array<RF,dim>& x) const
        {
          RF sum = 0.;
          for(unsigned int i = 0; i < dim; i++)
            sum += x[i] * x[i];
          RF h_eff = std::sqrt(sum);

          return variance * std::exp(-std::pow(h_eff,gamma));
        }
    };

    /**
     * @brief Gaussian covariance function
     *
     * The Gaussian, or square-exponential, covariance function is
     * \f$ C(h) = \sigma^2 \exp(-h^2) \$f, producing smooth sample paths.
     * This is the Matérn covariance function for \f$ \nu \to \infty \f$,
     * and the gamma-exponential covariance function for \f$ \gamma = 2 \f$.
     */
    class GaussianCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        GaussianCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
     *
     * The separable exponential covariance function is simply the
     * product of one-dimensional exponential covariance functions.
     */
    class SeparableExponentialCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        SeparableExponentialCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
     * @brief Matern covariance function
     *
     * The Matern covariance function family is very popular,
     * since it provides explicit control over the smoothness
     * of the resulting sample paths via its parameter
     * \f$ \nu \f$. The general family requires Bessel functions
     * provided by the GNU Scientific Library (GSL). The
     * special cases \f$ \nu = 1/2, 3/2, 5/2 \f$ are provided
     * separately, for use cases where the GSL is unavailable.
     */
      class MaternCovariance
      {
        const double nu;
        const double sqrtTwoNu;
        const double twoToOneMinusNu;
        const double gammaNu;

        public:

      /**
       * @brief Constructor
       *
       * @param config configuration used for nu
       */
        MaternCovariance(const Dune::ParameterTree& config)
          : nu(config.template get<double>("stochastic.maternNu")),
          sqrtTwoNu(std::sqrt(2.*nu)),
          twoToOneMinusNu(std::pow(2.,1.-nu)),
          gammaNu(std::tgamma(nu))
        {
          if (nu < 0.)
            DUNE_THROW(Dune::Exception,"matern nu has to be positive");
        }

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
        template<typename RF, long unsigned int dim>
        RF operator()(const RF variance, const std::array<RF,dim>& x) const
        {
#ifdef HAVE_GSL
          RF sum = 0.;
          for(unsigned int i = 0; i < dim; i++)
            sum += x[i] * x[i];
          RF h_eff = std::sqrt(sum);

          if (h_eff < 1e-10)
            return variance;
          else
            return variance * twoToOneMinusNu / gammaNu
              * std::pow(sqrtTwoNu * h_eff,nu) * gsl_sf_bessel_Knu(nu,sqrtTwoNu * h_eff);
#else
          DUNE_THROW(Dune::Exception,"general matern requires the GNU Scientific Library (gsl)");
#endif // HAVE_GSL
        }
      };

    /**
     * @brief Matern covariance function with nu = 3/2
     *
     * This is a special case of the Matérn covariance function for
     * \f$ \nu = 3/2 \f$, where the function simplifies to
     * \f$ C(h) = \sigma^2 (1 + \sqrt(3) h) \exp(-\sqrt(3) h) \f$.
     */
    class Matern32Covariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        Matern32Covariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * (1. + std::sqrt(3.)*h_eff)
              * std::exp(-std::sqrt(3.) * h_eff);
          }
    };

    /**
     * @brief Matern covariance function with nu = 5/2
     *
     * This is a special case of the Matérn covariance function for
     * \f$ \nu = 5/2 \f$, where the function simplifies to
     * \f$ C(h) = \sigma^2 (1 + \sqrt(5) h + \frac{5}{3} h^2) \exp(-\sqrt(5) h) \f$.
     */
    class Matern52Covariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        Matern52Covariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
              sum += x[i] * x[i];
            RF h_eff = std::sqrt(sum);

            return variance * (1. + std::sqrt(5.)*h_eff + 5./3.*h_eff*h_eff)
              * std::exp(-std::sqrt(5.) * h_eff);
          }
    };

    /**
     * @brief Damped oscillation covariance function
     *
     * This is a damped cosine that decays fast enough so that
     * it remains positive definite.
     */
    class DampedOscillationCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        DampedOscillationCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
     *
     * The Cauchy covariance function is \$f C(h) = {(1 + h^2)}^{-3} \f$.
     */
    class CauchyCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        CauchyCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
     * @brief Generalized Cauchy covariance function
     *
     * The generalized Cauchy covariance function family provides
     * covariance functions with low regularity, given by
     * \f$ C(h) = {(1 + h^\alpha)}^{-\beta} \f$.
     */
    class GeneralizedCauchyCovariance
    {
      const double alpha;
      const double beta;

      public:

      /**
       * @brief Constructor
       *
       * @param config configuration used for alpha and beta
       */
      GeneralizedCauchyCovariance(const Dune::ParameterTree& config)
        : alpha(config.template get<double>("stochastic.cauchyAlpha")),
        beta(config.template get<double>("stochastic.cauchyBeta"))
      {
        if (alpha <= 0. || alpha > 2.)
          DUNE_THROW(Dune::Exception,"generalized Cauchy alpha has to be in range (0,2]");
        if (beta <= 0.)
          DUNE_THROW(Dune::Exception,"generalized Cauchy beta has to be positive");
      }

      /**
       * @brief Evaluate the covariance function
       *
       * @tparam RF      type of values and coordinates
       * @tparam dim     dimension of domain
       *
       * @param variance covariance for lag zero
       * @param x        location, after scaling / trafo with correlation length
       *
       * @return resulting value
       */
      template<typename RF, long unsigned int dim>
        RF operator()(const RF variance, const std::array<RF,dim>& x) const
        {
          RF sum = 0.;
          for(unsigned int i = 0; i < dim; i++)
            sum += x[i] * x[i];
          RF h_eff = std::sqrt(sum);

          return variance * std::pow(1. + std::pow(h_eff,alpha),-beta);
        }
    };

    /**
     * @brief Cubic covariance function
     *
     * The cubic covariance function is the restriction of
     * \f$ C(h) = 1 - 7 h^2 + 8.75 h^3 - 3.5 h^5 + 0.75 h^7 \f$
     * to the interval $\f [0,1] \f$. It has compact support.
     */
    class CubicCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        CubicCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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
                return variance * (1. - 7. * std::pow(h_eff,2) + 8.75 * std::pow(h_eff,3)
                    - 3.5 * std::pow(h_eff,5) + 0.75 * std::pow(h_eff,7));
            }
            else
              DUNE_THROW(Dune::NotImplemented,
                  "cubic covariance only applicable in 1D or 2D");
          }
    };

    /**
     * @brief White noise covariance function
     *
     * This is the covariance for Gaussian white noise, i.e.,
     * no correlation at all between different locations. This
     * function is also known as "nugget" component in other
     * covariance functions. Consider using the specialized
     * generateUncorrelated methods instead if you simply want
     * white noise, since they are much cheaper if applicable.
     */
    class WhiteNoiseCovariance
    {
      public:

        /**
         * @brief Constructor
         *
         * @param config unused dummy argument
         */
        WhiteNoiseCovariance(const Dune::ParameterTree& config) {}

        /**
         * @brief Evaluate the covariance function
         *
         * @tparam RF      type of values and coordinates
         * @tparam dim     dimension of domain
         *
         * @param variance covariance for lag zero
         * @param x        location, after scaling / trafo with correlation length
         *
         * @return resulting value
         */
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

  }
}

#endif // DUNE_RANDOMFIELD_COVARIANCE_HH
