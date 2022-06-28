#ifndef DUNE_RANDOMFIELD_TREND_HH
#define DUNE_RANDOMFIELD_TREND_HH

#include<fstream>
#include<memory>

#if HAVE_GSL
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#else
#include<random>
#endif // HAVE_GSL

#include<dune/common/parametertreeparser.hh>
#if HAVE_DUNE_PDELAB
#include<dune/pdelab/gridfunctionspace/localfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/lfsindexcache.hh>
#endif //HAVE_DUNE_PDELAB

#include<dune/randomfield/pngreader.hh>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Predefined types of trend component
     *
     * This is an enumeration of the types of trend component
     * that are supported by the trend part.
     */
    struct TrendComponentType
    {
      enum Type {Mean, Slope, Disk, Block, Image};

      /**
       * @brief Check whether component is mean component
       */
      static bool isMean(Type i)
      {
        return (i == Mean);
      }

      /**
       * @brief Check whether component is slope component
       */
      static bool isSlope(Type i)
      {
        return (i == Slope);
      }

      /**
       * @brief Check whether component is Gaussian component
       */
      static bool isDisk(Type i)
      {
        return (i == Disk);
      }

      /**
       * @brief Check whether component is rectangle component
       */
      static bool isBlock(Type i)
      {
        return (i == Block);
      }

      /**
       * @brief Check whether component is a PNG image component
       */
      static bool isImage(Type i)
      {
        return (i == Image);
      }

    };

    /**
     * @brief Component of random field representing deterministic structure
     *
     * This class represents a component of the trend part of
     * the random field, i.e., some deterministic structure with associated
     * uncertain coefficients. The trend part is basically a list of such
     * trend components, which are uncorrelated, independent model
     * components.
     *
     * @tparam Traits class containing types, definitions and parameters
     */
    template<typename Traits>
      class TrendComponent
      {

        protected:

          using RF = typename Traits::RF;

          enum {dim = Traits::dim};

          std::shared_ptr<Traits> traits;

          TrendComponentType::Type componentType;
          unsigned int componentCount;

          std::array<RF,dim> extensions;

          std::vector<RF> shiftVector;
          std::vector<RF> meanVector;
          std::vector<RF> varianceVector;

        public:

          /**
           * @brief Constructor
           *
           * @param traits_         instance of Traits class with configuration
           * @param trendVector     current values for trend coefficients
           * @param meanVector_     mean values for trend coefficients
           * @param varianceVector_ variance for trend coefficients
           * @param componentType_  type of this component
           * @param componentCount_ index if more than one component of this type exists
           */
          TrendComponent(
              const std::shared_ptr<Traits>& traits_,
              const std::vector<RF>& trendVector,
              const std::vector<RF>& meanVector_,
              const std::vector<RF>& varianceVector_,
              const TrendComponentType::Type& componentType_,
              unsigned int componentCount_ = 0
              )
            :
              traits(traits_),
              componentType(componentType_),
              componentCount(componentCount_),
              extensions((*traits).extensions),
              shiftVector(trendVector),
              meanVector(meanVector_),
              varianceVector(varianceVector_)
        {
          if (trendVector.size() != meanVector.size()
              || trendVector.size() != varianceVector.size())
            DUNE_THROW(Dune::Exception,"trend component size does not match");

          if (TrendComponentType::isMean(componentType) && trendVector.size() != 1)
            DUNE_THROW(Dune::Exception,
                "Trend mean component must only contain one parameter");

          if (TrendComponentType::isSlope(componentType) && trendVector.size() != dim)
            DUNE_THROW(Dune::Exception,
                "Trend slope component must contain dim parameters: slope in each dimension");

          if (TrendComponentType::isDisk(componentType) && trendVector.size() != dim+2)
            DUNE_THROW(Dune::Exception,
                "Trend disk component must contain dim+2 parameters: position, radius, value");

          if (TrendComponentType::isBlock(componentType) && trendVector.size() != (2*dim)+1)
            DUNE_THROW(Dune::Exception,
                "Trend block component must contain (2*dim)+1 parameters: center, extent, value");

          for (unsigned int i = 0; i < shiftVector.size(); i++)
            shiftVector[i] -= meanVector[i];
        }

#if HAVE_DUNE_PDELAB
          /**
           * @brief Construct trend component from PDELab solution vector
           *
           * This function creates a trend component from a PDELab
           * GridFunctionSpace and associated coefficient vector. A typical
           * PDE solution can't be represented by such a component, so it
           * has to be decided what this operation should do. The given
           * implementation interprets the PDE solution as the sensitivity of
           * some quantity with regard to the random field, and constructs a
           * trend component with coefficients that represent the sensitivity
           * with regard to the component, via the chain rule and difference
           * quotients.
           *
           * @tparam GFS   type of GridFunctionSpace
           * @tparam Field type of coefficient vector
           *
           * @param gfs    PDELab GridFunctionSpace
           * @param field  corresponding coefficient vector
           */
          template<typename GFS, typename Field>
            void construct(const GFS& gfs, const Field& field)
            {
              std::vector<RF> newShiftVector(shiftVector.size(),0.),
                myNewShiftVector(shiftVector.size(),0.);

              using LFS      = Dune::PDELab::LocalFunctionSpace<GFS>;
              using LFSCache = Dune::PDELab::LFSIndexCache<LFS>;
              LFS lfs(gfs);
              LFSCache lfsCache(lfs);
              typename Field::template ConstLocalView<LFSCache> localView(field);
              std::vector<RF> vLocal;

              for (const auto& elem : elements(gfs.gridView(),Dune::Partitions::interior))
              {
                lfs.bind(elem);
                vLocal.resize(lfs.size());
                lfsCache.update();
                localView.bind(lfsCache);
                localView.read(vLocal);

                typename Traits::RangeType shift, deltaShift;
                RF delta = 1e-2;
                const typename Traits::DomainType& x = elem.geometry().center();
                evaluate(x,shift);
                for (unsigned int i = 0; i < shiftVector.size(); i++)
                {
                  shiftVector[i] += delta;
                  evaluate(x,deltaShift);
                  shiftVector[i] -= delta;

                  myNewShiftVector[i] += (deltaShift[0] - shift[0]) / delta * vLocal[0];
                }
              }

              MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),
                  shiftVector.size(),mpiType<RF>,MPI_SUM,(*traits).comm);
              shiftVector = newShiftVector;

            }

          /**
           * @brief Construct trend component from PDELab DiscreteGridFunction
           *
           * This function creates a trend component from a PDELab
           * DiscreteGridFunction object. A typical grid function can't be
           * represented by such a component, so it has to be decided what this
           * operation should do. The given implementation interprets the function
           * as the sensitivity of some quantity with regard to the random field,
           * and constructs a trend component with coefficients that represent the
           * sensitivity with regard to the component, via the chain rule and
           * difference quotients.
           *
           * @tparam DGF type of DiscreteGridFunction
           *
           * @param dgf DiscreteGridFunction containing sensitivities
           */
          template<typename DGF>
            void construct(const DGF& dgf)
            {
              std::vector<RF> newShiftVector(shiftVector.size(),0.),
                myNewShiftVector(shiftVector.size(),0.);

              for (const auto& elem : elements(dgf.getGridView(),Dune::Partitions::interior))
              {
                typename Traits::RangeType shift, deltaShift;
                RF delta = 1e-2;
                const typename Traits::DomainType& x = elem.geometry().center();
                evaluate(x,shift);
                const auto& center = referenceElement(elem.geometry()).position(0,0);
                Dune::FieldVector<RF,1> value;
                dgf.evaluate(elem,center,value);
                for (unsigned int i = 0; i < shiftVector.size(); i++)
                {
                  shiftVector[i] += delta;
                  evaluate(x,deltaShift);
                  shiftVector[i] -= delta;

                  myNewShiftVector[i] += (deltaShift[0] - shift[0]) / delta * value[0];
                }
              }

              MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),
                  shiftVector.size(),mpiType<RF>,MPI_SUM,(*traits).comm);
              shiftVector = newShiftVector;

            }
#endif // HAVE_DUNE_PDELAB

          /**
           * @brief Type of this trend component
           *
           * @return TrendComponentType object
           */
          TrendComponentType::Type type() const
          {
            return componentType;
          }

          /**
           * @brief Name of type of this trend component
           *
           * This returns a unique name for the component. If more than one
           * component of the same type is present, an index is added to the
           * name.
           *
           * @return string containing name
           */
          std::string name() const
          {
            if (TrendComponentType::isMean(componentType))
              return "mean";
            else if (TrendComponentType::isSlope(componentType))
              return "slope";
            else if (TrendComponentType::isDisk(componentType))
              return std::string("disk") + std::to_string(componentCount);
            else if (TrendComponentType::isBlock(componentType))
              return std::string("block") + std::to_string(componentCount);
            else
              DUNE_THROW(Dune::Exception,"Trend component type not found!");
          }

          /**
           * @brief Number of degrees of freedom
           *
           * @return degrees of freedom of this trend component
           */
          unsigned int dofs() const
          {
            if (TrendComponentType::isMean(componentType))
              return 1;
            else if (TrendComponentType::isSlope(componentType))
              return dim;
            else if (TrendComponentType::isDisk(componentType))
              return dim + 2;
            else if (TrendComponentType::isBlock(componentType))
              return 2*dim + 1;
            else
              DUNE_THROW(Dune::Exception,"Trend component type not found!");
          }

          /**
           * @brief Generate trend component coefficients with correct variance
           *
           * This function creates random coefficients with the assigned
           * mean and variance. Each component uses its own number generator,
           * and therefore requires a distict seed value.
           *
           * @param seed seed value for the random number generator
           */
          void generate(unsigned int seed)
          {
            std::vector<RF> newShiftVector(shiftVector.size(),0.),
              myNewShiftVector(shiftVector.size(),0.);

            if ((*traits).rank == 0)
            {
#if HAVE_GSL
              gsl_rng* gslRng = gsl_rng_alloc(gsl_rng_mt19937);
              gsl_rng_set(gslRng,seed);
#else
              std::default_random_engine generator(seed);
              std::normal_distribution<RF> normalDist(0.,1.);
#endif // HAVE_GSL

              for (unsigned int i = 0; i < shiftVector.size(); i++)
              {
#if HAVE_GSL
                myNewShiftVector[i] = gsl_ran_gaussian_ziggurat(gslRng,1.)
                  * std::sqrt(varianceVector[i]);
#else
                myNewShiftVector[i] = normalDist(generator) * std::sqrt(varianceVector[i]);
#endif // HAVE_GSL
              }
            }

            MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),
                shiftVector.size(),mpiType<RF>,MPI_SUM,(*traits).comm);
            shiftVector = newShiftVector;
          }

          /**
           * @brief Generate trend component coefficients that are noise
           *
           * This function generates white noise. Trend component coefficients
           * are generally uncorrelated, so the only difference to the generate
           * method is a variance of one.
           *
           * @param seed seed value for the random number generator
           *
           * @see generate
           */
          void generateUncorrelated(unsigned int seed)
          {
            std::vector<RF> newShiftVector(shiftVector.size(),0.),
              myNewShiftVector(shiftVector.size(),0.);

            if ((*traits).rank == 0)
            {
#if HAVE_GSL
              gsl_rng* gslRng = gsl_rng_alloc(gsl_rng_mt19937);
              gsl_rng_set(gslRng,seed);
#else
              std::default_random_engine generator(seed);
              std::normal_distribution<RF> normalDist(0.,1.);
#endif // HAVE_GSL

              for (unsigned int i = 0; i < shiftVector.size(); i++)
              {
#if HAVE_GSL
                myNewShiftVector[i] = gsl_ran_gaussian_ziggurat(gslRng,1.);
#else
                myNewShiftVector[i] = normalDist(generator);
#endif // HAVE_GSL
              }
            }

            MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),
                shiftVector.size(),mpiType<RF>,MPI_SUM,(*traits).comm);
            shiftVector = newShiftVector;
          }

          /**
           * @brief Multiply the trend coefficients with their variances
           *
           * This function performs multiplication with the covariance
           * matrix. Trend coefficients are uncorrelated, so this is just
           * multiplication with their variances.
           */
          void timesMatrix()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= varianceVector[i];
          }

          /**
           * @brief Divide the trend coefficients by their variances
           *
           * This function performs multiplication with the inverse of the
           * covariance matrix.
           *
           * @see timesMatrix
           */
          void timesInverseMatrix()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= 1./varianceVector[i];
          }

          /**
           * @brief Multiply the trend coefficients with their standard deviations
           *
           * This function performs multiplication with the square root of
           * the covariance matrix, which is multiplication with the standard
           * deviations in the case of uncorrelated trend component coefficients.
           *
           * @see timesMatrix
           */
          void timesMatrixRoot()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= std::sqrt(varianceVector[i]);
          }

          /**
           * @brief Divide the trend coefficients by their standard deviations
           *
           * This function performs multiplication with the inverse square root
           * of the covariance matrix.
           *
           * @see timesMatrixRoot
           */
          void timesInvMatRoot()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= 1./std::sqrt(varianceVector[i]);
          }

          /**
           * @brief Addition assignment operator
           *
           * @param other other trend component to add
           *
           * @return reference to updated trend component
           */
          TrendComponent<Traits>& operator+=(const TrendComponent<Traits>& other)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] += other.shiftVector[i];

            return *this;
          }

          /**
           * @brief Subtraction assignment operator
           *
           * @param other other trend component to subtract
           *
           * @return reference to updated trend component
           */
          TrendComponent<Traits>& operator-=(const TrendComponent<Traits>& other)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] -= other.shiftVector[i];

            return *this;
          }

          /**
           * @brief Multiplication with scalar
           *
           * @param alpha scale factor
           *
           * @return reference to updated trend component
           */
          TrendComponent<Traits>& operator*=(const RF alpha)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= alpha;

            return *this;
          }

          /**
           * @brief AXPY scaled addition
           *
           * Adds a multiple of another trend component.
           *
           * @param other other trend component to add
           * @param alpha scale factor
           *
           * @return reference to updated trend component
           */
          TrendComponent<Traits>& axpy(const TrendComponent<Traits>& other, const RF alpha)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] += other.shiftVector[i] * alpha;

            return *this;
          }

          /**
           * @brief Set trend component coefficients to zero
           *
           * This function sets the coefficients to zero. The
           * coefficients are always represented as deviations from
           * the mean, so evaluating the trend component will produce
           * non-zero results, namely the trend the coefficient mean
           * would produce.
           */
          void zero()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] = 0.;
          }

          /**
           * @brief Scalar product
           *
           * Sum of componentwise product across two trend components.
           *
           * @param other other trend component to multiply with
           *
           * @return resulting scalar value
           */
          RF operator*(const TrendComponent<Traits>& other) const
          {
            RF output = 0;

            for (unsigned int i = 0; i < shiftVector.size(); i++)
              output += shiftVector[i] * other.shiftVector[i];

            return output;
          }

          /**
           * @brief Equality operator
           *
           * @param other other trend component to compare to
           *
           * @return true if all coefficients are the same, else false
           */
          bool operator==(const TrendComponent<Traits>& other) const
          {
            bool same = true;

            for (unsigned int i = 0; i < shiftVector.size(); i++)
              if (shiftVector[i] != other.shiftVector[i])
              {
                same = false;
                break;
              }

            return same;
          }

          /**
           * @brief Inequality operator
           *
           * @param other other trend component to compare to
           *
           * @return true if operator== would return false, else false
           *
           * @see operator==
           */
          bool operator!=(const TrendComponent<Traits>& other) const
          {
            return !operator==(other);
          }

          /**
           * @brief Evaluate the trend component at a given location
           *
           * This function evaluates the trend component using its
           * current coefficients. The result is the value of the trend
           * function at the given location.
           *
           * @param      location coordinates where function should be evaluated
           * @param[out] output   value of trend component at given coordinates
           */
          void evaluate(
              const typename Traits::DomainType& location,
              typename Traits::RangeType& output
              ) const
          {
            if (TrendComponentType::isMean(componentType))
            {
              output[0] = meanVector[0] + shiftVector[0];
            }
            else if (TrendComponentType::isSlope(componentType))
            {
              output[0] = 0.;

              for (unsigned int i = 0; i < dim; i++)
                output[0] += (meanVector[i] + shiftVector[i])
                  * (location[i] - extensions[i]/2.);
            }
            else if (TrendComponentType::isDisk(componentType))
            {
              output[0] = 0.;

              RF distSquared = 0.;
              for (unsigned int i = 0; i < dim; i++)
                distSquared += std::pow(location[i] - (meanVector[i] + shiftVector[i]),2);

              output[0] = std::exp(- distSquared
                  / std::pow(meanVector[dim] + shiftVector[dim],2)
                  * (meanVector[dim+1] + shiftVector[dim+1]));
            }
            else if (TrendComponentType::isBlock(componentType))
            {
              output[0] = 0.;

              for (unsigned int i = 0; i < dim; i++)
              {
                if (std::abs(location[i] - (meanVector[i] + shiftVector[i]))
                    > 0.5 * (meanVector[dim+i] + shiftVector[dim+i]))
                  return;
              }

              output[0] = meanVector[2*dim] + shiftVector[2*dim];
            }
            else
              DUNE_THROW(Dune::Exception,"Trend component type not found!");
          }

          /**
           * @brief Sum of abs. values of component
           *
           * @return resulting value
           */
          RF oneNorm() const
          {
            RF output = 0.;
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              output += std::abs(shiftVector[i]);
            return output;
          }

          /**
           * @brief Maximum abs. value of component
           *
           * @return resulting value
           */
          RF infNorm() const
          {
            RF output = 0.;
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              output = std::max(output, std::abs(shiftVector[i]));
            return output;
          }

          /**
           * @brief Write the trend component to hard disk
           *
           * This function writes the trend component to a file, using
           * the same file format as the ParameterTree. The resulting
           * file can then be read in again at a later point in time.
           *
           * @param file file object to write component to
           */
          void writeToFile(std::ofstream& file) const
          {
            if ((*traits).rank == 0)
            {
              if (TrendComponentType::isMean(componentType))
                file << "mean =";
              else if (TrendComponentType::isSlope(componentType))
                file << "slope =";
              else if (TrendComponentType::isDisk(componentType))
                file << "disk" << componentCount << " =";
              else if (TrendComponentType::isBlock(componentType))
                file << "block" << componentCount << " =";
              else
                DUNE_THROW(Dune::Exception,"Trend component type not found!");

              for (unsigned int i = 0; i < shiftVector.size(); i++)
                file << " " << meanVector[i] + shiftVector[i];

              file << std::endl;
            }
          }

      };

    /**
     * @brief Component of random field based on pixel image
     *
     * This is a special version of the trend component class, representing
     * a PNG image file. The component can be used to read in a user-defined
     * field that contains up to 256 discrete values between zero and one.
     * It supports a single trend coefficient, which is the scale factor for
     * the resulting field.
     *
     * @tparam Traits class containing types, definitions and parameters
     */
    template<typename Traits>
      class ImageComponent
      : public TrendComponent<Traits>
      {
        using RF = typename TrendComponent<Traits>::RF;

        enum {dim = TrendComponent<Traits>::dim};

        const std::string imageFile;
        const PNGReader   pngReader;

        std::array<RF,dim> extensions;

        public:

        /**
         * @brief Constructor reading image file
         *
         * @see TrendComponent
         */
        ImageComponent(
            const std::shared_ptr<Traits>& traits,
            const std::vector<RF>& trendVector,
            const std::vector<RF>& meanVector,
            const std::vector<RF>& varianceVector,
            const std::string& imageFile_
            )
          :
            TrendComponent<Traits>(traits,trendVector,meanVector,varianceVector,
                TrendComponentType::Image),
            imageFile(imageFile_),
            pngReader(imageFile),
            extensions((*traits).extensions)
        {
          if (trendVector.size() != meanVector.size()
              || trendVector.size() != varianceVector.size())
            DUNE_THROW(Dune::Exception,"trend component size does not match");

          if (trendVector.size() != 1)
            DUNE_THROW(Dune::Exception,
                "Image component must only contain one parameter");

          if (dim != 2)
            DUNE_THROW(Dune::Exception,"image trend components require dim == 2");
        }

#if HAVE_DUNE_PDELAB
        /**
         * @brief Constructor based on PDELab solution
         *
         * Same as the general TrendComponent version, but copies over
         * the PNG reader and PNG file.
         *
         * @param other other ImageComponent to copy image from
         * @param gfs   GridFunctionSpace object
         * @param field corresponding coefficient vector
         *
         * @see TrendComponent
         */
        template<typename GFS, typename Field>
          ImageComponent(
              const ImageComponent<Traits>& other,
              const GFS& gfs,
              const Field& field
              )
          :
            TrendComponent<Traits>(other,gfs,field),
            imageFile(other.imageFile),
            pngReader(other.pngReader),
            extensions(other.extensions)
        {}

        /**
         * @brief Constructor based on PDELab DiscreteGridFunction
         *
         * Same as the general TrendComponent version, but copies over
         * the PNG reader and PNG file.
         *
         * @param other other ImageComponent to copy image from
         * @param dgf   DiscreteGridFunction object containing sensitivities
         *
         * @see TrendComponent
         */
        template<typename DGF>
          ImageComponent(const ImageComponent<Traits>& other, const DGF& dgf)
          :
            TrendComponent<Traits>(other,dgf),
            imageFile(other.imageFile),
            pngReader(other.pngReader),
            extensions(other.extensions)
        {}
#endif // HAVE_DUNE_PDELAB

        /**
         * @brief Name of type of this trend component
         *
         * @see TrendComponent
         */
        std::string name() const
        {
          return "image";
        }

        /**
         * @brief Number of degrees of freedom
         *
         * @see TrendComponent
         */
        unsigned int dofs() const
        {
          return 1;
        }

        /**
         * @brief Evaluate the trend component at a given location
         *
         * @see TrendComponent
         */
        void evaluate(
            const typename Traits::DomainType& location,
            typename Traits::RangeType& output
            ) const
        {
          output[0] = (this->meanVector[0] + this->shiftVector[0])
            * pngReader.evaluate(location,extensions);
        }

        /**
         * @brief Sum of abs. values of component
         *
         * @see TrendComponent
         */
        RF oneNorm() const
        {
          return std::abs(this->shiftVector[0]);
        }

        /**
         * @brief Maximum abs. value of component
         *
         * @see TrendComponent
         */
        RF infNorm() const
        {
          return std::abs(this->shiftVector[0]);
        }

        /**
         * @brief Write the trend component to hard disk
         *
         * @see TrendComponent
         */
        void writeToFile(std::ofstream& file) const
        {
          if ((*(this->traits)).rank == 0)
          {
            file << "image =";
            for (unsigned int i = 0; i < this->shiftVector.size(); i++)
              file << " " << this->meanVector[i] + this->shiftVector[i];
            file << std::endl;
          }
        }

      };

    /**
     * @brief Part of random field that consists of deterministic components
     *
     * This class represents deterministic trend functions that modify the
     * random field, e.g., a non-zero mean, a slope defining a local mean
     * that is a linear function of the coordinates, or localized artifacts.
     * Each such trend component can have coefficients with associated
     * uncertainty, e.g., a non-zero mean that is itself a random variable,
     * or be fixed to some deterministic values.
     *
     * @tparam Traits class containing types, definitions and parameters
     */
    template<typename Traits>
      class TrendPart
      {
        using RF = typename Traits::RF;

        std::shared_ptr<Traits> traits;

        std::vector<TrendComponent<Traits>>     componentVector;
        std::shared_ptr<ImageComponent<Traits>> imageComponent;

        public:

        /**
         * @brief Constructor reading from file or creating homogeneous field
         *
         * This constructor reads the coefficients of the trend components from
         * a file in ParameterTree format, which has been written using the
         * writeToFile method. If the file name argument is empty, a homogeneous
         * field is created instead, by setting each coefficient to its
         * expected value.
         *
         * @param config   ParameterTree object containing configuration
         * @param traits_  traits object with parameters, communication, etc.
         * @param fileName name of file containing coefficients, or empty string
         */
        TrendPart(
            const Dune::ParameterTree& config,
            const std::shared_ptr<Traits>& traits_,
            const std::string& fileName = ""
            )
          :
            traits(traits_)
        {
          std::vector<RF> emptyVector, trendVector, meanVector, varianceVector;

          meanVector = config.get<std::vector<RF>>("mean.mean",emptyVector);

          if (!meanVector.empty())
          {
            varianceVector = config.get<std::vector<RF>>("mean.variance");

            if (fileName == "")
            {
              trendVector = meanVector;
            }
            else
            {
              Dune::ParameterTree trendConfig;
              Dune::ParameterTreeParser parser;
              parser.readINITree(fileName+".trend",trendConfig);
              trendVector = trendConfig.get<std::vector<RF>>("mean");
            }

            componentVector.emplace_back(traits,trendVector,
                meanVector,varianceVector,TrendComponentType::Mean);
          }

          meanVector = config.get<std::vector<RF>>("slope.mean",emptyVector);

          if (!meanVector.empty())
          {
            varianceVector = config.get<std::vector<RF>>("slope.variance");

            if (fileName == "")
            {
              trendVector = meanVector;
            }
            else
            {
              Dune::ParameterTree trendConfig;
              Dune::ParameterTreeParser parser;
              parser.readINITree(fileName+".trend",trendConfig);
              trendVector = trendConfig.get<std::vector<RF>>("slope");
            }

            componentVector.emplace_back(traits,trendVector,
                meanVector,varianceVector,TrendComponentType::Slope);
          }

          int count = 0;
          std::stringstream s;
          bool endReached = false;

          while(!endReached)
          {
            s.clear();
            s.str(std::string());
            s << count;
            meanVector = config.get<std::vector<RF>>("disk"+s.str()+".mean",emptyVector);

            if (meanVector.empty())
            {
              endReached = true;
            }
            else
            {
              varianceVector = config.get<std::vector<RF>>("disk"+s.str()+".variance");

              if (fileName == "")
              {
                trendVector = meanVector;
              }
              else
              {
                Dune::ParameterTree trendConfig;
                Dune::ParameterTreeParser parser;
                parser.readINITree(fileName+".trend",trendConfig);
                trendVector = trendConfig.get<std::vector<RF>>("disk"+s.str());
              }

              componentVector.emplace_back(traits,trendVector,
                  meanVector,varianceVector,TrendComponentType::Disk,count);

              count++;
            }
          }

          count = 0;
          s.clear();
          endReached = false;

          while(!endReached)
          {
            s.clear();
            s.str(std::string());
            s << count;
            meanVector = config.get<std::vector<RF>>("block"+s.str()+".mean",emptyVector);

            if (meanVector.empty())
            {
              endReached = true;
            }
            else
            {
              varianceVector = config.get<std::vector<RF>>("block"+s.str()+".variance");

              if (fileName == "")
              {
                trendVector = meanVector;
              }
              else
              {
                Dune::ParameterTree trendConfig;
                Dune::ParameterTreeParser parser;
                parser.readINITree(fileName+".trend",trendConfig);
                trendVector = trendConfig.get<std::vector<RF>>("block"+s.str());
              }

              componentVector.emplace_back(traits,trendVector,
                  meanVector,varianceVector,TrendComponentType::Block,count);

              count++;
            }
          }

          meanVector = config.get<std::vector<RF>>("image.mean",emptyVector);

          if (!meanVector.empty())
          {
            varianceVector = config.get<std::vector<RF>>("image.variance");

            if (fileName == "")
            {
              trendVector = meanVector;
            }
            else
            {
              Dune::ParameterTree trendConfig;
              Dune::ParameterTreeParser parser;
              parser.readINITree(fileName+".trend",trendConfig);
              trendVector = trendConfig.get<std::vector<RF>>("image");
            }

            const std::string imageFile = config.get<std::string>("image.filename");
            imageComponent = std::make_shared<ImageComponent<Traits>>(traits,trendVector,
                meanVector,varianceVector,imageFile);
          }
        }

#if HAVE_DUNE_PDELAB
        /**
         * @brief Constructor based on PDELab solution
           *
           * This function creates list of trend components from a PDELab
           * GridFunctionSpace and associated coefficient vector. A typical
           * PDE solution can't be represented by these components, so it
           * has to be decided what this operation should do. The given
           * implementation interprets the PDE solution as the sensitivity of
           * some quantity with regard to the random field, and constructs
           * components with coefficients that represent the sensitivity
           * with regard to the components, via the chain rule and difference
           * quotients.
           *
           * @tparam GFS   type of GridFunctionSpace
           * @tparam Field type of coefficient vector
           *
           * @param other other trend part to copy components from
           * @param gfs   PDELab GridFunctionSpace
           * @param field corresponding coefficient vector
         */
        template<typename GFS, typename Field>
          TrendPart(
              const TrendPart<Traits>& other,
              const GFS& gfs,
              const Field& field
              )
          :
            traits(other.traits),
            componentVector(other.componentVector)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].construct(gfs,field);

          if (other.imageComponent)
          {
            imageComponent = std::make_shared<ImageComponent<Traits>>
              (*(other.imageComponent));
            imageComponent->construct(gfs,field);
          }
        }

        /**
         * @brief Constructor based on PDELab DiscreteGridFunction
           *
           * This function creates a list of trend components from a PDELab
           * DiscreteGridFunction object. A typical grid function can't be
           * represented by these components, so it has to be decided what this
           * operation should do. The given implementation interprets the function
           * as the sensitivity of some quantity with regard to the random field,
           * and constructs trend components with coefficients that represent the
           * sensitivity with regard to the components, via the chain rule and
           * difference quotients.
           *
           * @tparam DGF type of DiscreteGridFunction
           *
           * @param other other trend part to copy components from
           * @param dgf   DiscreteGridFunction containing sensitivities
         */
        template<typename DGF>
          TrendPart(const TrendPart<Traits>& other, const DGF& dgf)
          :
            traits(other.traits),
            componentVector(other.componentVector)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].construct(dgf);

          if (other.imageComponent)
          {
            imageComponent = std::make_shared<ImageComponent<Traits>>
              (*(other.imageComponent));
            imageComponent->construct(dgf);
          }
        }
#endif // HAVE_DUNE_PDELAB

        /**
         * @brief Number of degrees of freedom
           *
           * Returns the sum of the degrees of freedom of each component.
           *
           * @return total number of degrees of freedom
         */
        unsigned int dofs() const
        {
          unsigned int output = 0;

          for (unsigned int i = 0; i < componentVector.size(); i++)
            output += componentVector[i].dofs();

          if (imageComponent)
            output += (*imageComponent).dofs();

          return output;
        }

        /**
         * @brief Generate a trend part with desired covariance structure
           *
           * This function creates random coefficients with the assigned
           * mean and variance. Each component uses its own number generator,
           * and therefore requires a distict seed value.
           *
           * @param seed seed value for the random number generator
         */
        void generate(unsigned int seed)
        {
          // different seed than stochastic part
          seed += (*traits).commSize;
          for (unsigned int i = 0; i < componentVector.size(); i++)
          {
            // different seed for each component
            componentVector[i].generate(seed + i);
          }

          if (imageComponent)
            imageComponent->generate(seed + componentVector.size());
        }

        /**
         * @brief Generate a trend part without correlation (i.e. noise)
         *
           * This function generates white noise. Trend components and their
           * coefficients are generally uncorrelated, so the only difference
           * to the generate method is a variance of one.
           *
           * @param seed seed value for the random number generator
           *
           * @see generate
         */
        void generateUncorrelated(unsigned int seed)
        {
          // different seed than stochastic part
          seed += (*traits).commSize;
          for (unsigned int i = 0; i < componentVector.size(); i++)
          {
            // different seed for each component
            componentVector[i].generateUncorrelated(seed + i);
          }

          if (imageComponent)
            imageComponent->generateUncorrelated(seed + componentVector.size());
        }

        /**
         * @brief Number of stored trend components (excluding image)
         *
         * This function returns the number of trend components the
         * trend part contains, except for an optional PNG image component,
         * which is treated separately.
         *
         * @return number of components
         */
        unsigned int size() const
        {
          return componentVector.size();
        }

        /**
         * @brief Access ith trend component (excluding image)
         *
         * This function grants access to one of the components.
         *
         * @param i index of the desired component
         *
         * @return refererence to the selected component
         */
        const TrendComponent<Traits>& getComponent(unsigned int i) const
        {
          return componentVector[i];
        }

        /**
         * @brief Access image component if available
         *
         * This function returns the image component, if present.
         *
         * @return smart pointer to image component, or invalid pointer.
         */
        const std::shared_ptr<const ImageComponent<Traits>>& getImageComponent() const
        {
          return imageComponent;
        }

        /**
         * @brief Multiply the trend part with its covariance matrix
           *
           * This function performs multiplication with the covariance
           * matrix. Trend components are uncorrelated, so this is an
           * operation that can be performed for each component separately.
         */
        void timesMatrix()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesMatrix();

          if (imageComponent)
            imageComponent->timesMatrix();
        }

        /**
         * @brief Multiply the trend part with the inverse of its covariance matrix
         *
           * This function performs multiplication with the inverse of the
           * covariance matrix.
           *
           * @see timesMatrix
         */
        void timesInverseMatrix()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesInverseMatrix();

          if (imageComponent)
            imageComponent->timesInverseMatrix();
        }

        /**
         * @brief Multiply the trend part with approximate root of its cov. matrix
         *
           * This function performs multiplication with the square root of
           * the covariance matrix, which is multiplication with the standard
           * deviations of the coefficients for each component separately.
           *
           * @see timesMatrix
         */
        void timesMatrixRoot()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesMatrixRoot();

          if (imageComponent)
            imageComponent->timesMatrixRoot();
        }

        /**
         * @brief Multiply the trend part with approximate inverse root of its cov. matrix
         *
           * This function performs multiplication with the inverse square root
           * of the covariance matrix.
           *
           * @see timesMatrixRoot
         */
        void timesInvMatRoot()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesInvMatRoot();

          if (imageComponent)
            imageComponent->timesInvMatRoot();
        }

        /**
         * @brief Addition assignment operator
           *
           * @param other other trend part to add
           *
           * @return reference to updated trend part
         */
        TrendPart<Traits>& operator+=(const TrendPart<Traits>& other)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i] += other.componentVector[i];

          if (imageComponent)
            *imageComponent += *(other.imageComponent);

          return *this;
        }

        /**
         * @brief Subtraction assignment operator
           *
           * @param other other trend part to subtract
           *
           * @return reference to updated trend part
         */
        TrendPart<Traits>& operator-=(const TrendPart<Traits>& other)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i] -= other.componentVector[i];

          if (imageComponent)
            *imageComponent -= *(other.imageComponent);

          return *this;
        }

        /**
         * @brief Multiplication with scalar
           *
           * @param alpha scale factor
           *
           * @return reference to updated trend part
         */
        TrendPart<Traits>& operator*=(const RF alpha)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i] *= alpha;

          if (imageComponent)
            *imageComponent *= alpha;

          return *this;
        }

        /**
         * @brief AXPY scaled addition
           *
           * Adds a multiple of another trend part.
           *
           * @param other other trend part to add
           * @param alpha scale factor
           *
           * @return reference to updated trend part
         */
        TrendPart<Traits>& axpy(const TrendPart<Traits>& other, const RF alpha)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].axpy(other.componentVector[i],alpha);

          if (imageComponent)
            imageComponent->axpy(*(other.imageComponent),alpha);

          return *this;
        }

        /**
         * @brief Set the trend part to zero
           *
           * This function sets the coefficients to zero. The
           * coefficients are always represented as deviations from
           * the mean, so evaluating the trend components will produce
           * non-zero results, namely the trend the coefficient mean
           * would produce.
         */
        void zero()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].zero();

          if (imageComponent)
            imageComponent->zero();
        }

        /**
         * @brief Scalar product
           *
           * Sum of scalar products of components.
           *
           * @param other other trend part to multiply with
           *
           * @return resulting scalar value
         */
        RF operator*(const TrendPart<Traits>& other) const
        {
          RF output = 0.;

          for (unsigned int i = 0; i < componentVector.size(); i++)
            output += componentVector[i] * other.componentVector[i];

          if (imageComponent)
            output += *imageComponent * *(other.imageComponent);

          return output;
        }

        /**
         * @brief Equality operator
           *
           * @param other other trend part to compare to
           *
           * @return true if all components are the same, else false
         */
        bool operator==(const TrendPart<Traits>& other) const
        {
          bool same = true;

          for (unsigned int i = 0; i < componentVector.size(); i++)
            if (! (componentVector[i] == other.componentVector[i]))
            {
              same = false;
              break;
            }

          if (imageComponent)
            same = same && (*imageComponent == *(other.imageComponent));

          return same;
        }

        /**
         * @brief Inequality operator
           *
           * @param other other trend part to compare to
           *
           * @return true if operator== would return false, else false
           *
           * @see operator==
         */
        bool operator!=(const TrendPart<Traits>& other) const
        {
          return !operator==(other);
        }

        /**
         * @brief One norm
           *
           * Sum of one norms of individual components.
           *
           * @return resulting value
         */
        RF oneNorm() const
        {
          RF output = 0.;

          for (unsigned int i = 0; i < componentVector.size(); i++)
            output += componentVector[i].oneNorm();

          if (imageComponent)
            output += (*imageComponent).oneNorm();

          return output;
        }

        /**
         * @brief Infinity norm
           *
           * Maximum of infinity norms of individual components.
           *
           * @return resulting value
         */
        RF infNorm() const
        {
          RF output = 0.;

          for (unsigned int i = 0; i < componentVector.size(); i++)
            output = std::max(output, componentVector[i].infNorm());

          if (imageComponent)
            output = std::max(output, (*imageComponent).infNorm());

          return output;
        }

        /**
         * @brief Evaluate the trend part at a given location
           *
           * This function evaluates the trend part using the current
           * coefficients of its components. The result is the value of the
           * trend function at the given location.
           *
           * @param      x      coordinates where function should be evaluated
           * @param[out] output value of trend component at given coordinates
         */
        void evaluate(
            const typename Traits::DomainType& x,
            typename Traits::RangeType& output
            ) const
        {
          output = 0.;
          typename Traits::RangeType compOutput = 0.;

          for (unsigned int i = 0; i < componentVector.size(); i++)
          {
            componentVector[i].evaluate(x,compOutput);
            output[0] += compOutput[0];
          }

          if (imageComponent)
          {
            imageComponent->evaluate(x,compOutput);
            output[0] += compOutput[0];
          }
        }

        /**
         * @brief Write the trend part to hard disk
           *
           * This function writes the trend part to a file, using
           * the same file format as the ParameterTree. The resulting
           * file can then be read in again at a later point in time.
           *
           * @param fileName file to write trend part to
         */
        void writeToFile(const std::string& fileName) const
        {
          if ((*traits).rank == 0)
          {
            std::ofstream file(fileName+".trend",std::ofstream::trunc);

            for (unsigned int i = 0; i < componentVector.size(); i++)
              componentVector[i].writeToFile(file);

            if (imageComponent)
              imageComponent->writeToFile(file);
          }
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_TREND_HH
