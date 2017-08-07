// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_COVARIANCE_HH
#define	DUNE_RANDOMFIELD_COVARIANCE_HH

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
              DUNE_THROW(Dune::NotImplemented,"spherical covariance only defined for 1D, 2D and 3D");
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
                return variance * (1. - 7. * std::pow(h_eff,2) + 8.75 * std::pow(h_eff,3)
                    - 3.5 * std::pow(h_eff,5) + 0.75 * std::pow(h_eff,7));
            }
            else
              DUNE_THROW(Dune::NotImplemented,"cubic covariance only applicable in 1D or 2D");
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

  }
}

#endif // DUNE_RANDOMFIELD_COVARIANCE_HH
