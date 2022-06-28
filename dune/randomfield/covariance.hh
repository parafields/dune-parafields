#ifndef DUNE_RANDOMFIELD_COVARIANCE_HH
#define DUNE_RANDOMFIELD_COVARIANCE_HH

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
     * @brief Spherical covariance function
     *
     * The spherical covariance function represents the interaction
     * between two spheres in a certain distance, in terms of overlapping
     * volume of the two spheres. It has compact support.
     */
    class SphericalCovariance
    {
      public:

        /*
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
     * covariance function for \f$ \nu = 1/2 \f$.
     */
    class ExponentialCovariance
    {
      public:

        /*
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
     * @brief Gaussian covariance function
     *
     * The Gaussian, or square-exponential, covariance function is
     * \f$ C(h) = \sigma^2 \exp(-h^2) \$f, producing smooth sample paths.
     * This is the Matérn covariance function for \f$ \nu \to \infty \f$.
     */
    class GaussianCovariance
    {
      public:

        /*
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

        /*
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
     * @brief Matern covariance function with nu = 3/2
     *
     * This is a special case of the Matérn covariance function for
     * \f$ \nu = 3/2 \f$, where the function simplifies to
     * \f$ C(h) = \sigma^2 (1 + \sqrt(3) h) \exp(-\sqrt(3) h) \f$.
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

        /*
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

        /*
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

        /*
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
     * @brief Cubic covariance function
     *
     * The cubic covariance function is the restriction of
     * \f$ C(h) = 1 - 7 h^2 + 8.75 h^3 - 3.5 h^5 + 0.75 h^7 \f$
     * to the interval $\f [0,1] \f$. It has compact support.
     */
    class CubicCovariance
    {
      public:

        /*
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

        /*
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
