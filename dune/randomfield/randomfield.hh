#ifndef DUNE_RANDOMFIELD_RANDOMFIELD_HH
#define DUNE_RANDOMFIELD_RANDOMFIELD_HH

#include<dune/common/parametertree.hh>

#if HAVE_DUNE_FUNCTIONS
// for VTK output functionality
#include<dune/grid/yaspgrid.hh>
#include<dune/grid/io/file/vtk.hh>
#include<dune/functions/gridfunctions/analyticgridviewfunction.hh>
#endif // HAVE_DUNE_FUNCTIONS

#include<dune/randomfield/io.hh>
#include<dune/randomfield/fieldtraits.hh>
#include<dune/randomfield/trend.hh>
#include<dune/randomfield/stochastic.hh>
#include<dune/randomfield/matrix.hh>
#include<dune/randomfield/mutators.hh>
#include<dune/randomfield/legacyvtk.hh>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Gaussian random field in arbitrary dimensions
     *
     * The central class, representing stationary Gaussian random fields on a
     * structured grid in arbitrary dimension. The underlying data types can be
     * controlled using the GridTraits template parameter, one of several
     * implementations of the circulant embedding technique can be chosen using
     * the IsoMatrix and AnisoMatrix template template parameters, and options
     * like the covariance function, correlation length, domain size, etc., can
     * be controlled through the ParameterTree constructor argument. The
     * constructor itself doesn't generate a field, you have to call the generate
     * method to do that.
     *
     * @tparam GridTraits  class defining dimension and data type of entries, etc.
     * @tparam IsoMatrix   covariance matrix implementation using underlying symmetries
     * @tparam AnisoMatrix covariance matrix implementation for general covariance functions
     */
    template<typename GridTraits,
      template<typename> class IsoMatrix = DefaultIsoMatrix<GridTraits::dim>::template Type,
      template<typename> class AnisoMatrix = DefaultAnisoMatrix<GridTraits::dim>::template Type>
        class RandomField
        {
          public:

            using Traits             = RandomFieldTraits<GridTraits,IsoMatrix,AnisoMatrix>;

          protected:

            using StochasticPartType = StochasticPart<Traits>;
            using RF                 = typename Traits::RF;

            /**
             * @brief Internal class for ParameterTree value extraction
             *
             * The field constructor needs to construct the ParameterTree
             * config object by reading in a file, but that is not possible
             * while in the initialization list. This helper class parses the
             * file in its constructor, so that the needed object is available
             * for all subsequent entries in the list.
             */
            class ParamTreeHelper
            {
              Dune::ParameterTree config;

              public:

              /**
               * @brief Constructor
               *
               * Reads config from given file, or creates an empty config if
               * file name was empty.
               *
               * @param fileName name of config file, or empty string
               */
              ParamTreeHelper(const std::string& fileName = "")
              {
                if (fileName != "")
                {
                  Dune::ParameterTreeParser parser;
                  parser.readINITree(fileName+".field",config);
                }
              }

              /**
               * @brief Access to the generated config object
               *
               * @return reference to the config object
               */
              const Dune::ParameterTree& get() const
              {
                return config;
              }
            };

            const ParamTreeHelper           treeHelper;
            const Dune::ParameterTree       config;
            const ValueTransform<RF>        valueTransform;
            std::shared_ptr<Traits>         traits;

            using IsoMatrixPtr   = std::shared_ptr<IsoMatrix<Traits>>;
            using AnisoMatrixPtr = std::shared_ptr<AnisoMatrix<Traits>>;
            IsoMatrixPtr       isoMatrix;
            AnisoMatrixPtr     anisoMatrix;
            bool               useAnisoMatrix;
            TrendPart<Traits>  trendPart;
            StochasticPartType stochasticPart;

            bool                                        cacheInvMatvec;
            bool                                        cacheInvRootMatvec;
            mutable std::shared_ptr<StochasticPartType> invMatvecPart;
            mutable std::shared_ptr<StochasticPartType> invRootMatvecPart;
            mutable bool                                invMatvecValid;
            mutable bool                                invRootMatvecValid;

          public:

            /**
             * @brief Constructor reading from file or creating homogeneous field
             *
             * This constructor creates a homogeneous (i.e., constant) field if an
             * empty string is passed as file name argument, and else tries to read
             * a field from file. After successful construction you can call the
             * generate method to generate a random field using circulant embedding,
             * and repeat this process whenever you need a new random field.
             *
             * @tparam LoadBalance class used for parallel data distribution with MPI
             *
             * @param config_     ParameterTree object containing configuration
             * @param fileName    name of file to read, empty string to not read anything
             * @param loadBalance instance of the load balancer, if needed
             * @param comm        MPI communicator for parallel field generation
             *
             * @see generate
             */
            template<typename LoadBalance = DefaultLoadBalance<GridTraits::dim>>
              explicit RandomField(const Dune::ParameterTree& config_, const std::string& fileName = "",
                  const LoadBalance& loadBalance = LoadBalance(), const MPI_Comm comm = MPI_COMM_WORLD)
              : config(config_), valueTransform(config), traits(new Traits(config,loadBalance,comm)),
              trendPart(config,traits,fileName), stochasticPart(traits,fileName),
              cacheInvMatvec((*traits).cacheInvMatvec), cacheInvRootMatvec((*traits).cacheInvRootMatvec),
              invMatvecValid(false), invRootMatvecValid(false)
          {
            if (fileName == "")
            {
              invMatvecValid = true;
              invRootMatvecValid = true;
            }

            const std::string& anisotropy
              = (*traits).config.template get<std::string>("stochastic.anisotropy","none");
            const std::string& covariance
              = (*traits).config.template get<std::string>("stochastic.covariance");
            if (covariance != "custom-aniso" && (anisotropy == "none" || anisotropy == "axiparallel"))
            {
              isoMatrix = IsoMatrixPtr(new IsoMatrix<Traits>(traits));
              useAnisoMatrix = false;
            }
            else
            {
              anisoMatrix = AnisoMatrixPtr(new AnisoMatrix<Traits>(traits));
              useAnisoMatrix = true;
            }

            if (cacheInvMatvec)
              invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));

            if (cacheInvRootMatvec)
              invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));
          }

            /**
             * @brief Constructor reading field and config from file
             *
             * This constructor reads not only the field, but also its underlying parameters
             * (i.e., the ParameterTree config object) from a file. This can be used to read
             * in a field that was generated by another program run or another program
             * altogether, and still have the correct parameterization available.
             *
             * @tparam LoadBalance class used for parallel data distribution with MPI
             *
             * @param fileName    name of file to read (base name only)
             * @param loadBalance instance of the load balancer, if needed
             * @param comm        MPI communicator for parallel field generation
             *
             * @see generate
             */
            template<typename LoadBalance = DefaultLoadBalance<GridTraits::dim>>
              RandomField(const std::string& fileName, const LoadBalance& loadBalance = LoadBalance(),
                  const MPI_Comm comm = MPI_COMM_WORLD)
              : treeHelper(fileName), config(treeHelper.get()),
              valueTransform(config), traits(new Traits(config,loadBalance,comm)),
              trendPart(config,traits,fileName), stochasticPart(traits,fileName),
              cacheInvMatvec((*traits).cacheInvMatvec), cacheInvRootMatvec((*traits).cacheInvRootMatvec),
              invMatvecValid(false), invRootMatvecValid(false)
          {
            const std::string& anisotropy
              = (*traits).config.template get<std::string>("stochastic.anisotropy","none");
            const std::string& covariance
              = (*traits).config.template get<std::string>("stochastic.covariance");
            if (covariance != "custom-aniso" && (anisotropy == "none" || anisotropy == "axiparallel"))
            {
              isoMatrix = IsoMatrixPtr(new IsoMatrix<Traits>(traits));
              useAnisoMatrix = false;
            }
            else
            {
              anisoMatrix = AnisoMatrixPtr(new AnisoMatrix<Traits>(traits));
              useAnisoMatrix = true;
            }

            if (cacheInvMatvec)
              invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));

            if (cacheInvRootMatvec)
              invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));
          }

            /**
             * @brief Constructor copying traits and covariance matrix
             *
             * This constructor also reads a field from file, but reuses the configuration
             * and covariance matrix from another field that has already been constructed.
             * This avoids generating the same covariance matrix several times, saving memory
             * when multiple fields have to be read in and subsequently modified.
             *
             * @param other    other random field to copy most members from
             * @param fileName name of file to read in (base name only)
             *
             * @see generate
             */
            RandomField(const RandomField& other, const std::string& fileName)
              : config(other.config), valueTransform(other.valueTransform), traits(other.traits),
              isoMatrix(other.isoMatrix), anisoMatrix(other.anisoMatrix), useAnisoMatrix(other.useAnisoMatrix),
              trendPart(config,traits,fileName), stochasticPart(traits,fileName),
              cacheInvMatvec((*traits).cacheInvMatvec), cacheInvRootMatvec((*traits).cacheInvRootMatvec),
              invMatvecValid(false), invRootMatvecValid(false)
          {
            if (cacheInvMatvec)
              invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));

            if (cacheInvRootMatvec)
              invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));
          }

#if HAVE_DUNE_PDELAB
            /**
             * @brief Constructor converting from GridFunctionSpace and GridVector
             *
             * This constructor uses a PDELab GridFunctionSpace (representation of the
             * discrete function space on a finite element mesh) and GridVector
             * (representation of the coefficient vector) to initialize the values of
             * a random field. Useful to convert sensitivities during Bayesian inversion.
             *
             * @tparam GFS   type of GridFunctionSpace
             * @tparam Field type of coefficient vector
             *
             * @param other other random field to copy most members from
             * @param gfs   instance of GridFunctionSpace
             * @param field coefficient vector to copy data from
             */
            template<typename GFS, typename Field>
              RandomField(const RandomField& other, const GFS& gfs, const Field& field)
              : config(other.config), valueTransform(other.valueTransform), traits(other.traits),
              isoMatrix(other.isoMatrix), anisoMatrix(other.anisoMatrix), useAnisoMatrix(other.useAnisoMatrix),
              trendPart(other.trendPart,gfs,field), stochasticPart(other.stochasticPart,gfs,field),
              cacheInvMatvec((*traits).cacheInvMatvec), cacheInvRootMatvec((*traits).cacheInvRootMatvec),
              invMatvecValid(false), invRootMatvecValid(false)
          {
            if (cacheInvMatvec)
              invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));

            if (cacheInvRootMatvec)
              invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));
          }

            /**
             * @brief Constructor converting from DiscreteGridFunction
             *
             * This constructor uses a PDELab DiscreteGridFunction (representation of the
             * of a function living on a finite element mesh) to initialize the values of
             * a random field. Useful to convert sensitivities during Bayesian inversion.
             *
             * @tparam DGF   type of DiscreteGridFunction
             *
             * @param other other random field to copy most members from
             * @param dgf   DiscreteGridFunction to copy data from
             */
            template<typename DGF>
              RandomField(const RandomField& other, const DGF& dgf)
              : config(other.config), valueTransform(other.valueTransform), traits(other.traits),
              isoMatrix(other.isoMatrix), anisoMatrix(other.anisoMatrix), useAnisoMatrix(other.useAnisoMatrix),
              trendPart(other.trendPart,dgf), stochasticPart(other.stochasticPart,dgf),
              cacheInvMatvec((*traits).cacheInvMatvec), cacheInvRootMatvec((*traits).cacheInvRootMatvec),
              invMatvecValid(false), invRootMatvecValid(false)
          {
            if (cacheInvMatvec)
              invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));

            if (cacheInvRootMatvec)
              invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(stochasticPart));
          }
#endif // HAVE_DUNE_PDELAB

            /**
             * @brief Copy constructor
             *
             * Standard copy constructor, creating a copy of a given random field,
             * sharing the covariance matrix between all instances.
             *
             * @param other other random field that should be copied
             */
            RandomField(const RandomField& other)
              : config(other.config), valueTransform(other.valueTransform), traits(other.traits),
              isoMatrix(other.isoMatrix), anisoMatrix(other.anisoMatrix), useAnisoMatrix(other.useAnisoMatrix),
              trendPart(other.trendPart), stochasticPart(other.stochasticPart),
              cacheInvMatvec((*traits).cacheInvMatvec), cacheInvRootMatvec((*traits).cacheInvRootMatvec),
              invMatvecValid(other.invMatvecValid), invRootMatvecValid(other.invRootMatvecValid)
          {
            if (cacheInvMatvec && other.cacheInvMatvec)
              invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(*(other.invMatvecPart)));

            if (cacheInvRootMatvec && other.cacheInvRootMatvec)
              invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(*(other.invRootMatvecPart)));
          }

            /**
             * @brief Assignment operator
             *
             * Standard assignment operator, sharing the covariance matrix
             * between all instances.
             *
             * @param other other random field that should be copied
             */
            RandomField& operator=(const RandomField& other)
            {
              if (this != &other)
              {
                config             = other.config;
                valueTransform     = other.valueTransform;
                traits             = other.traits;
                isoMatrix          = other.isoMatrix;
                anisoMatrix        = other.anisoMatrix;
                useAnisoMatrix     = other.useAnisoMatrix;
                trendPart          = other.trendPart;
                stochasticPart     = other.stochasticPart;
                cacheInvMatvec     = other.cacheInvMatvec;
                cacheInvRootMatvec = other.cacheInvRootMatvec;

                if (cacheInvMatvec)
                {
                  invMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(*(other.invMatvecPart)));
                  invMatvecValid = other.invMatvecValid;
                }

                if (cacheInvRootMatvec)
                {
                  invRootMatvecPart = std::shared_ptr<StochasticPartType>(new StochasticPartType(*(other.invRootMatvecPart)));
                  invRootMatvecValid = other.invRootMatvecValid;
                }
              }

              return *this;
            }

            /**
             * @brief Cell volume of the random field discretization
             *
             * @return volume of one of the grid cells the values are assigned to
             */
            RF cellVolume() const
            {
              return (*traits).cellVolume;
            }

            /**
             * @brief Number of degrees of freedom
             *
             * This function returns the total number of degrees of freedom, i.e.,
             * summed across different processors, if applicable, and counting
             * trend components.
             *
             * @return total degrees of freedom
             */
            unsigned int dofs() const
            {
              return stochasticPart.dofs() + trendPart.dofs();
            }

            /**
             * @brief Explicit matrix setup for custom covariance classes
             *
             * This function can be called if a custom user-supplied covariance
             * function should be used. The function is passed as a template
             * parameter, and has to fulfill the interface of the built-in
             * covariance functions. In the configuration, "custom-iso" or
             * "custom-aniso" has to be chosen as the desired covariance type.
             * The former assumes that the covariance function is symmetric in
             * each dimension, leading to significant memory savings, and will
             * not work if that is not the case.
             */
            template<typename Covariance>
            void fillMatrix()
            {
              if (useAnisoMatrix)
                (*anisoMatrix).template fillTransformedMatrix<Covariance>();
              else
                (*isoMatrix).template fillTransformedMatrix<Covariance>();
            }

            /**
             * @brief Generate a field with the desired correlation structure
             *
             * Generate a random field sample based on the configured
             * variance, covariance function, etc. By default, this function
             * does not proceed if a non-World MPI communicator has been
             * configured, because it would, e.g., silently generate different
             * random fields on each of the equivalence classes defined by separate
             * communicators. If you are sure that you want to do that (e.g.,
             * because you are running separate MCMC chains on different processors),
             * you can pass true as an argument to disable the check.
             *
             * @param allowNonWorldComm prevent inconsistent field generation by default
             */
            void generate(bool allowNonWorldComm = false)
            {
              // create seed out of current time
              unsigned int seed = (unsigned int) clock();
              // different seeds for different fields
              seed += static_cast<int>(reinterpret_cast<uintptr_t>(&stochasticPart));

              generate(seed,allowNonWorldComm);
            }

            /**
             * @brief Generate a field with desired correlation structure using seed
             *
             * Generate a random field sample, using a specific seed value for field
             * generation. Note that you may still end up with different fields if
             * you rerun with the same seed on different machines, or if you generate
             * a field in parallel and the data distribution changes.
             *
             * @param seed              seed value for random number generation
             * @param allowNonWorldComm prevent inconsistent field generation by default
             */
            void generate(unsigned int seed, bool allowNonWorldComm = false)
            {
              if (((*traits).comm != MPI_COMM_WORLD) && !allowNonWorldComm)
                DUNE_THROW(Dune::Exception,
                    "generation of inconsistent fields prevented, set allowNonWorldComm = true if you really want this");

              if ((*traits).verbose && (*traits).rank == 0)
                std::cout << "generate with seed: " << seed << std::endl;

              if (useAnisoMatrix)
                (*anisoMatrix).generateField(seed,stochasticPart);
              else
                (*isoMatrix).generateField(seed,stochasticPart);
              trendPart.generate(seed);

              invMatvecValid     = false;
              invRootMatvecValid = false;
            }

            /**
             * @brief Generate a field without correlation structure (i.e. noise)
             *
             * This is a convenience function that generates white noise on the grid.
             * Some applications require such white noise, and with this function it
             * can be generated without defining a second type of random field, or
             * applying circulant embedding for a case where it isn't actually needed.
             *
             * @param allowNonWorldComm prevent inconsistent field generation by default
             */
            void generateUncorrelated(bool allowNonWorldComm = false)
            {
              // create seed out of current time
              unsigned int seed = (unsigned int) clock();
              // different seeds for different fields
              seed += static_cast<int>(reinterpret_cast<uintptr_t>(&stochasticPart));

              generateUncorrelated(seed,allowNonWorldComm);
            }

            /**
             * @brief Generate a field containing noise using seed
             *
             * This is a convenience function that generates white noise, based on a
             * given seed value.
             *
             * @param seed              seed value for random number generation
             * @param allowNonWorldComm prevent inconsistent field generation by default
             */
            void generateUncorrelated(unsigned int seed, bool allowNonWorldComm = false)
            {
              if (((*traits).comm != MPI_COMM_WORLD) && !allowNonWorldComm)
                DUNE_THROW(Dune::Exception,
                    "generation of inconsistent fields prevented, set allowNonWorldComm = true if you really want this");

              if (useAnisoMatrix)
                (*anisoMatrix).generateUncorrelatedField(seed,stochasticPart);
              else
                (*isoMatrix).generateUncorrelatedField(seed,stochasticPart);
              trendPart.generateUncorrelated(seed);

              invMatvecValid     = false;
              invRootMatvecValid = false;
            }

#if HAVE_DUNE_GRID
            /**
             * @brief Evaluate the random field in the coordinates of an element
             *
             * This function evaluates the random field in the local coordinate of
             * a given finite element. This is needed during local assembly, e.g.,
             * in PDELab. Note that the finite element does not necessarily 
             * coincide with a cell of the random field grid, i.e., depending on
             * your discretization the random field is not constant on the element.
             *
             * @param      elem   finite element the field should be evaluated on
             * @param      xElem  local coordinates on the element
             * @param[out] output field value at given position
             */
            template<typename Element>
              void evaluate(
                  const Element& elem,
                  const typename Traits::DomainType& xElem,
                  typename Traits::RangeType& output
                  ) const
              {
                const typename Traits::DomainType location = elem.geometry().global(xElem);
                evaluate(location,output);
              }
#endif // HAVE_DUNE_GRID

            /**
             * @brief Evaluate the random field at given coordinates
             *
             * This function evaluates the random field at the given coordinates, i.e.,
             * sums up the spatially distributed component and possible contributions
             * from trend components.
             *
             * @param      location coordinates where field should be evaluated
             * @param[out] output   field value at given position
             */
            void evaluate(const typename Traits::DomainType& location, typename Traits::RangeType& output) const
            {
              typename Traits::RangeType stochastic = 0., trend = 0.;

              stochasticPart.evaluate(location,stochastic);
              trendPart     .evaluate(location,trend);

              output = stochastic + trend;
              valueTransform.apply(output);
            }

            /**
             * @brief Export random field to files on disk
             *
             * This function writes the random field, its trend components, and its
             * configuration into files on disk, so that they can be made persistent
             * and possibly read in again using the corresponding constructor.
             *
             * @param fileName base file name that should be used
             */
            void writeToFile(const std::string& fileName) const
            {
              stochasticPart.writeToFile(fileName);
              trendPart     .writeToFile(fileName);

              std::ofstream file(fileName+".field",std::ofstream::trunc);
              config.report(file);
            }

            /**
             * @brief Export random field as flat unstructured VTK file, requires dune-grid and dune-functions
             *
             * This function writes the whole random field to a VTK file, i.e., the
             * sum of the spatially distributed part and all present trend components.
             * These are the same values as returned by evaluate.
             *
             * @param fileName file name for VTK output
             * @param gv       Dune GridView defining the grid used in the VTK file
             */
            template<typename GridView>
              void writeToVTK(const std::string& fileName, const GridView& gv) const
              {
#if HAVE_DUNE_FUNCTIONS
                Dune::VTKWriter<GridView> vtkWriter(gv,Dune::VTK::conforming);
                auto f = Dune::Functions::makeAnalyticGridViewFunction([&](auto x)
                    {typename Traits::RangeType output; this->evaluate(x,output); return output;},gv);
                vtkWriter.addCellData(f,VTK::FieldInfo(fileName,VTK::FieldInfo::Type::scalar,1));
                vtkWriter.pwrite(fileName,"","",Dune::VTK::appendedraw);
#else //HAVE_DUNE_FUNCTIONS
                DUNE_THROW(Dune::NotImplemented,"Unstructured VTK output requires dune-grid and dune-functions");
#endif //HAVE_DUNE_FUNCTIONS
              }

            /**
             * @brief Export random field as unstructured VTK file, requires dune-grid and dune-functions
             *
             * This function writes the individual components of the random field to a
             * VTK file, i.e., the spatially distributed part and all present trend
             * components each as a separate data set.
             *
             * @param fileName file name for VTK output
             * @param gv       Dune GridView defining the grid used in the VTK file
             */
            template<typename GridView>
              void writeToVTKSeparate(const std::string& fileName, const GridView& gv) const
              {
#if HAVE_DUNE_FUNCTIONS
                Dune::VTKWriter<GridView> vtkWriter(gv,Dune::VTK::conforming);
                {
                  auto f = Dune::Functions::makeAnalyticGridViewFunction([&](auto x)
                      {typename Traits::RangeType output; stochasticPart.evaluate(x,output); return output;},gv);
                  vtkWriter.addCellData(f,VTK::FieldInfo("stochastic",VTK::FieldInfo::Type::scalar,1));
                }
                for (unsigned int i = 0; i < trendPart.size(); i++)
                {
                  const TrendComponent<Traits>& component = trendPart.getComponent(i);
                  auto f = Dune::Functions::makeAnalyticGridViewFunction([&](auto x)
                      {typename Traits::RangeType output; component.evaluate(x,output); return output;},gv);
                  vtkWriter.addCellData(f,VTK::FieldInfo(component.name(),VTK::FieldInfo::Type::scalar,1));
                }
                vtkWriter.pwrite(fileName,"","",Dune::VTK::appendedraw);
#else //HAVE_DUNE_FUNCTIONS
                DUNE_THROW(Dune::NotImplemented,"Unstructured VTK output requires dune-grid and dune-functions");
#endif //HAVE_DUNE_FUNCTIONS
              }

            /**
             * @brief Export random field as flat Legacy VTK file
             *
             * Same as writeToVTK, but writes a simple legacy VTK format and doesn't depend
             * on dune-grid and dune-functions. Unfortunately, this function doesn't work
             * for parallel runs, only for sequential field creation.
             *
             * @param fileName file name for VTK output
             *
             * @see writeToVTK
             */
            void writeToLegacyVTK(const std::string& fileName) const
            {
              if ((*traits).commSize > 1)
                DUNE_THROW(Dune::NotImplemented,"legacy VTK output doesn't work for parallel runs");

              LegacyVTKWriter<Traits> legacyWriter(config,fileName);
              legacyWriter.writeScalarData("field",*this);
            }

            /**
             * @brief Export random field as separate Legacy VTK entries
             *
             * Same as writeToVTKSeparate, but writes a simple legacy VTK format and
             * doesn't depend on dune-grid and dune-functions. Unfortunately, this function
             * doesn't work for parallel runs, only for sequential field creation.
             *
             * @param fileName file name for VTK output
             *
             * @see writeToVTK
             */
            void writeToLegacyVTKSeparate(const std::string& fileName) const
            {
              if ((*traits).commSize > 1)
                DUNE_THROW(Dune::NotImplemented,"legacy VTK output doesn't work for parallel runs");

              LegacyVTKWriter<Traits> legacyWriter(config,fileName);
              legacyWriter.writeScalarData("stochastic",stochasticPart);
              for (unsigned int i = 0; i < trendPart.size(); i++)
              {
                const TrendComponent<Traits>& component = trendPart.getComponent(i);
                legacyWriter.writeScalarData(component.name(),component);
              }
            }

            /**
             * @brief Make random field homogeneous
             *
             * This function sets both the field values and the trend coefficients
             * to zero, creating a field that represents a constant function with
             * value zero.
             */
            void zero()
            {
              trendPart     .zero();
              stochasticPart.zero();

              if (cacheInvMatvec)
              {
                (*invMatvecPart).zero();
                invMatvecValid = true;
              }

              if (cacheInvRootMatvec)
              {
                (*invRootMatvecPart).zero();
                invRootMatvecValid = true;
              }
            }

            /**
             * @brief Double spatial resolution of covariance matrix
             *
             * This function instructs the covariance matrix to subdivide each cell
             * in each dimension and recompute itself. It is not a part of the refine
             * method for design reasons, since several random fields can share a
             * covariance matrix, i.e., this method should only be called once, and
             * then refine should be called on all fields sharing the matrix instance.
             */
            void refineMatrix()
            {
              (*traits).refine();
              if (useAnisoMatrix)
                (*anisoMatrix).update();
              else
                (*isoMatrix).update();
            }

            /**
             * @brief Double spatial resolution of random field
             *
             * This function subdivides each cell in each dimension and interpolates
             * the random field. If available, it uses the cached matrix-vector
             * product, refines that instead, and multiplies with the new refined
             * covariance matrix, which yields a smoother interpolation.
             */
            void refine()
            {
              if (cacheInvMatvec && invMatvecValid)
              {
                (*invMatvecPart).refine();
                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix) * (*invMatvecPart);
                else
                  stochasticPart = (*isoMatrix) * (*invMatvecPart);

                const RF scale = std::pow(0.5,-(*traits).dim);
                stochasticPart *= scale;
                *invMatvecPart *= scale;

                if (cacheInvRootMatvec)
                {
                  if (useAnisoMatrix)
                    *invRootMatvecPart = (*anisoMatrix).multiplyRoot(*invMatvecPart);
                  else
                    *invRootMatvecPart = (*isoMatrix).multiplyRoot(*invMatvecPart);

                  *invRootMatvecPart *= scale;
                  invRootMatvecValid = true;
                }

              }
              else if (cacheInvRootMatvec && invRootMatvecValid)
              {
                (*invRootMatvecPart).refine();
                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix).multiplyRoot(*invRootMatvecPart);
                else
                  stochasticPart = (*isoMatrix).multiplyRoot(*invRootMatvecPart);

                const RF scale = std::pow(0.5,-(*traits).dim);
                stochasticPart     *= scale;
                *invRootMatvecPart *= scale;

                if (cacheInvMatvec)
                {
                  *invMatvecPart = stochasticPart;
                  invMatvecValid = false;
                }
              }
              else
              {
                stochasticPart.refine();

                if (cacheInvMatvec)
                  (*invMatvecPart).refine();

                if (cacheInvRootMatvec)
                  (*invRootMatvecPart).refine();
              }
            }

            /**
             * @brief Reduce spatial resolution of covariance matrix
             *
             * Inverse operation to refineMatrix, merging cells to build larger
             * cells out of them, and then recomputing the covariance matrix.
             *
             * @see refineMatrix
             */
            void coarsenMatrix()
            {
              (*traits).coarsen();
              if (useAnisoMatrix)
                (*anisoMatrix).update();
              else
                (*isoMatrix).update();
            }

            /**
             * @brief Reduce spatial resolution of random field
             *
             * Inverser operation to refine, merges cells by averaging their
             * values. Just like refine, makes use of cached matrix-vector
             * products if they are available.
             *
             * @see refine
             */
            void coarsen()
            {
              if (cacheInvMatvec && invMatvecValid)
              {
                (*invMatvecPart).coarsen();
                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix) * (*invMatvecPart);
                else
                  stochasticPart = (*isoMatrix) * (*invMatvecPart);

                const RF scale = std::pow(0.5,(*traits).dim);
                stochasticPart *= scale;
                *invMatvecPart *= scale;

                if (cacheInvRootMatvec)
                {
                  if (useAnisoMatrix)
                    *invRootMatvecPart = (*anisoMatrix).multiplyRoot(*invMatvecPart);
                  else
                    *invRootMatvecPart = (*isoMatrix).multiplyRoot(*invMatvecPart);

                  *invRootMatvecPart *= scale;
                  invRootMatvecValid = true;
                }
              }
              else if (cacheInvRootMatvec && invRootMatvecValid)
              {
                (*invRootMatvecPart).coarsen();
                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix).multiplyRoot(*invRootMatvecPart);
                else
                  stochasticPart = (*isoMatrix).multiplyRoot(*invRootMatvecPart);

                const RF scale = std::pow(0.5,(*traits).dim);
                stochasticPart     *= scale;
                *invRootMatvecPart *= scale;

                if (cacheInvMatvec)
                {
                  *invMatvecPart = stochasticPart;
                  invMatvecValid = false;
                }
              }
              else
              {
                stochasticPart.coarsen();

                if (cacheInvMatvec)
                  (*invMatvecPart).coarsen();

                if (cacheInvRootMatvec)
                  (*invRootMatvecPart).coarsen();
              }
            }

            /**
             * @brief Addition assignment operator
             *
             * @param other other random field that should be added
             *
             * @return reference to updated random field
             */
            RandomField& operator+=(const RandomField& other)
            {
              trendPart      += other.trendPart;
              stochasticPart += other.stochasticPart;

              if (cacheInvMatvec)
              {
                if (other.cacheInvMatvec)
                {
                  *invMatvecPart += *(other.invMatvecPart);
                  invMatvecValid = invMatvecValid && other.invMatvecValid;
                }
                else
                  invMatvecValid = false;
              }

              if (cacheInvRootMatvec)
              {
                if (other.cacheInvRootMatvec)
                {
                  *invRootMatvecPart += *(other.invRootMatvecPart);
                  invRootMatvecValid = invRootMatvecValid && other.invRootMatvecValid;
                }
                else
                  invRootMatvecValid = false;
              }

              return *this;
            }

            /**
             * @brief Subtraction assignment operator
             *
             * @param other other random field that should be subtracted
             *
             * @return reference to updated random field
             */
            RandomField& operator-=(const RandomField& other)
            {
              trendPart      -= other.trendPart;
              stochasticPart -= other.stochasticPart;

              if (cacheInvMatvec)
              {
                if (other.cacheInvMatvec)
                {
                  *invMatvecPart -= *(other.invMatvecPart);
                  invMatvecValid = invMatvecValid && other.invMatvecValid;
                }
                else
                  invMatvecValid = false;
              }

              if (cacheInvRootMatvec)
              {
                if (other.cacheInvRootMatvec)
                {
                  *invRootMatvecPart -= *(other.invRootMatvecPart);
                  invRootMatvecValid = invRootMatvecValid && other.invRootMatvecValid;
                }
                else
                  invRootMatvecValid = false;
              }

              return *this;
            }

            /**
             * @brief Multiplication with scalar
             *
             * @param alpha scale factor
             *
             * @return reference to updated random field
             */
            RandomField& operator*=(const RF alpha)
            {
              trendPart      *= alpha;
              stochasticPart *= alpha;

              if (cacheInvMatvec)
                *invMatvecPart *= alpha;

              if (cacheInvRootMatvec)
                *invRootMatvecPart *= alpha;

              return *this;
            }

            /**
             * @brief AXPY scaled addition
             *
             * Adds a multiple of another random field.
             *
             * @param other other random field to add
             * @param alpha scale factor for other field
             *
             * @return reference to updated random field
             */
            RandomField& axpy(const RandomField& other, const RF alpha)
            {
              trendPart     .axpy(other.trendPart     ,alpha);
              stochasticPart.axpy(other.stochasticPart,alpha);

              if (cacheInvMatvec)
              {
                if (other.cacheInvMatvec)
                {
                  (*invMatvecPart).axpy(*(other.invMatvecPart),alpha);
                  invMatvecValid = invMatvecValid && other.invMatvecValid;
                }
                else
                  invMatvecValid = false;
              }

              if (cacheInvRootMatvec)
              {
                if (other.cacheInvRootMatvec)
                {
                  (*invRootMatvecPart).axpy(*(other.invRootMatvecPart),alpha);
                  invRootMatvecValid = invRootMatvecValid && other.invRootMatvecValid;
                }
                else
                  invRootMatvecValid = false;
              }

              return *this;
            }

            /**
             * @brief AXPY scaled addition (swapped arguments)
             *
             * Second version of scaled addition, with swapped arguments.
             *
             * @param other other random field to add
             * @param alpha scale factor for other field
             *
             * @return reference to updated random field
             */
            RandomField& axpy(const RF alpha, const RandomField& other)
            {
              return axpy(other,alpha);
            }

            /**
             * @brief Scalar product
             *
             * Scalar product of spatially distributed coefficients plus
             * scalar product of trend coefficients.
             *
             * @param other other random field to multiply with
             *
             * @return resulting scalar value
             */
            RF operator*(const RandomField& other) const
            {
              RF output = 0.;

              output += (*this).stochasticPart * other.stochasticPart;
              output += (*this).trendPart * other.trendPart;

              return output;
            }

            /**
             * @brief Multiply random field with covariance matrix
             *
             * Multiplies the random field with the covariance matrix, useful
             * for Bayesian inversion. Caches the original field as the
             * matrix-vector product of the resulting field with the inverse
             * of the covariance matrix if configured to do so. This makes
             * evaluating corresponding objective functions very cheap.
             */
            void timesMatrix()
            {
              if (cacheInvMatvec)
              {
                *invMatvecPart = stochasticPart;
                invMatvecValid  = true;
              }

              if (cacheInvRootMatvec)
              {
                if (useAnisoMatrix)
                  *invRootMatvecPart = (*anisoMatrix).multiplyRoot(stochasticPart);
                else
                  *invRootMatvecPart = (*isoMatrix).multiplyRoot(stochasticPart);
                invRootMatvecValid = true;
              }

              if (useAnisoMatrix)
                stochasticPart = (*anisoMatrix) * stochasticPart;
              else
                stochasticPart = (*isoMatrix) * stochasticPart;

              trendPart.timesMatrix();
            }

            /**
             * @brief Multiply random field with inverse of covariance matrix
             *
             * Multiplies the random field with the inverse of the covariance matrix,
             * as is necessary in Bayesian inversion. Makes use of cached matrix-vector
             * products if configured to do so and they are available.
             */
            void timesInverseMatrix()
            {
              if (cacheInvMatvec && invMatvecValid)
              {
                if (cacheInvRootMatvec)
                {
                  if (useAnisoMatrix)
                    *invRootMatvecPart = (*anisoMatrix).multiplyRoot(*invMatvecPart);
                  else
                    *invRootMatvecPart = (*isoMatrix).multiplyRoot(*invMatvecPart);
                  invRootMatvecValid = true;
                }

                stochasticPart = *invMatvecPart;
                invMatvecValid = false;
              }
              else
              {
                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix).multiplyInverse(stochasticPart);
                else
                  stochasticPart = (*isoMatrix).multiplyInverse(stochasticPart);

                if (cacheInvMatvec)
                  invMatvecValid = false;

                if (cacheInvRootMatvec)
                  invRootMatvecValid = false;
              }

              trendPart.timesInverseMatrix();
            }

            /**
             * @brief Multiply random field with approximate root of cov. matrix
             *
             * Multiplies the random field with an approximation of the root of
             * the covariance matrix. The field is embedded in the extended
             * circulant embedding domain, multiplied with the root of the
             * extended covariance matrix (which is known exactly), and then
             * restricted to the original domain. This introduces boundary effects,
             * and therefore this matrix-vector product is not exact.
             */
            void timesMatrixRoot()
            {
              if (cacheInvMatvec && cacheInvRootMatvec)
              {
                *invMatvecPart = *invRootMatvecPart;
                invMatvecValid = invRootMatvecValid;
              }

              if (cacheInvRootMatvec)
              {
                *invRootMatvecPart = stochasticPart;
                invRootMatvecValid = true;
              }

              if (useAnisoMatrix)
                stochasticPart = (*anisoMatrix).multiplyRoot(stochasticPart);
              else
                stochasticPart = (*isoMatrix).multiplyRoot(stochasticPart);

              trendPart.timesMatrixRoot();
            }

            /**
             * @brief Multiply random field with approximate inverse root of cov. matrix
             *
             * Same as timesMatrixRoot, but with the inverse of the root, instead of
             * the root itself. Introduces the same kind of boundary effects.
             *
             * @see timesMatrixRoot
             */
            void timesInvMatRoot()
            {
              if (cacheInvRootMatvec && invRootMatvecValid)
              {
                stochasticPart = *invRootMatvecPart;
                invRootMatvecValid = false;

                if (cacheInvMatvec)
                {
                  *invRootMatvecPart = *invMatvecPart;
                  invRootMatvecValid = invMatvecValid;
                  invMatvecValid  = false;
                }
              }
              else
              {
                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix).multiplyInverse(stochasticPart);
                else
                  stochasticPart = (*isoMatrix).multiplyInverse(stochasticPart);

                if (cacheInvRootMatvec)
                {
                  *invRootMatvecPart = stochasticPart;
                  invRootMatvecValid = true;
                }

                if (useAnisoMatrix)
                  stochasticPart = (*anisoMatrix).multiplyRoot(stochasticPart);
                else
                  stochasticPart = (*isoMatrix).multiplyRoot(stochasticPart);

                if (cacheInvMatvec)
                  invMatvecValid = false;
              }

              trendPart.timesInvMatRoot();
            }

            /**
             * @brief One-norm (sum of absolute values)
             *
             * Sums the absolute values of spatially distributed cell values
             * and trend coefficients.
             *
             * @return resulting sum
             */
            RF oneNorm() const
            {
              return trendPart.oneNorm() + stochasticPart.oneNorm();
            }

            /**
             * @brief Euclidean norm
             *
             * Sums the squares of spatially distributed cell values and trend
             * components, and returns the square root of the sum.
             *
             * @return resulting value
             */
            RF twoNorm() const
            {
              return std::sqrt( *this * *this);
            }

            /**
             * @brief Maximum norm
             *
             * Returns the maximum value across both the spatially distributed
             * values and the trend coefficients.
             *
             * @return resulting value
             */
            RF infNorm() const
            {
              return std::max(trendPart.infNorm(), stochasticPart.infNorm());
            }

            /**
             * @brief Equality operator
             *
             * @param other other random field to compare with
             *
             * @return true if all field values are the same, else false
             */
            bool operator==(const RandomField& other) const
            {
              return (trendPart == other.trendPart && stochasticPart == other.stochasticPart);
            }

            /**
             * @brief Inequality operator
             *
             * @param other other random field to compare with
             *
             * @return true if operator== would return false, else false
             *
             * @see operator==
             */
            bool operator!=(const RandomField& other) const
            {
              return !operator==(other);
            }

            /**
             * @brief Multiply field with Gaussian with given center and radius
             *
             * This is a helper function that dampens the random field except for a
             * spherical region around a given location. The field is multiplied
             * with a Gaussian with height one.
             *
             * @param center location of maximum of Gaussian
             * @param radius scale parameter (standard deviation of Gaussian)
             */
            void localize(const typename Traits::DomainType& center, const RF radius)
            {
              stochasticPart.localize(center,radius);

              if (cacheInvMatvec)
                invMatvecValid = false;

              if (cacheInvRootMatvec)
                invRootMatvecValid = false;
            }
        };

    /**
     * @brief List of Gaussian random fields in arbitrary dimensions
     *
     * This class represents a list of uncorrelated stationary Gaussian random fields.
     * Each such field has its own variance, covariance function, etc. Fields can be
     * added to the list one by one, and the field list supports most operations the
     * individual fields do, so this class can be used as a drop-in when multiple fields
     * have to be managed at the same time, i.e., in joint Bayesian inversion of multiple
     * parameterizations.
     *
     * @tparam GridTraits  class defining dimension and data type of entries, etc.
     * @tparam IsoMatrix   covariance matrix implementation using underlying symmetries
     * @tparam AnisoMatrix covariance matrix implementation for general covariance functions
     * @tparam RandomField type of the field entries, can be a subtype of RandomField
     */
    template<typename GridTraits,
      template<typename> class IsoMatrix = DefaultIsoMatrix<GridTraits::dim>::template Type,
      template<typename> class AnisoMatrix = DefaultAnisoMatrix<GridTraits::dim>::template Type,
      template<typename, template<typename> class, template<typename> class>
        class RandomField = Dune::RandomField::RandomField>
        class RandomFieldList
        {
          public:

            using SubRandomField = RandomField<GridTraits,IsoMatrix,AnisoMatrix>;

          protected:

            /**
             * @brief Internal class for ParameterTree value extraction
             *
             * The field list constructor needs to construct the ParameterTree
             * config object by reading in a file, but that is not possible
             * while in the initialization list. This helper class parses the
             * file in its constructor, so that the needed object is available
             * for all subsequent entries in the list.
             */
            class ParamTreeHelper
            {
              Dune::ParameterTree config;

              public:

              /**
               * @brief Constructor
               *
               * Reads config from given file, or creates an empty config if
               * file name was empty.
               *
               * @param fileName name of config file, or empty string
               */
              ParamTreeHelper(const std::string& fileName = "")
              {
                if (fileName != "")
                {
                  Dune::ParameterTreeParser parser;
                  parser.readINITree(fileName+".fieldList",config);
                }
              }

              /**
               * @brief Access to the generated config object
               *
               * @return reference to the config object
               */
              const Dune::ParameterTree& get() const
              {
                return config;
              }
            };

            const ParamTreeHelper     treeHelper;
            const Dune::ParameterTree config;
            std::vector<std::string>  fieldNames;
            std::vector<std::string>  activeTypes;

            std::map<std::string, std::shared_ptr<SubRandomField>> list;
            std::shared_ptr<SubRandomField> emptyPointer;

            using Traits = RandomFieldTraits<GridTraits,IsoMatrix,AnisoMatrix>;
            using RF     = typename Traits::RF;

          public:

            /**
             * @brief Default constructor constructing empty list
             */
              RandomFieldList()
              {}

            /**
             * @brief Constructor reading random fields from file
             *
             * This constructor creates a list of homogeneous (i.e., constant) fields
             * if an empty string is passed as file name argument, and else tries to read
             * the fields from file. After successful construction you can call the
             * generate method to generate a list of random fields using circulant
             * embedding, and repeat this process whenever you need a new list of fields.
             *
             * @tparam LoadBalance class used for parallel data distribution with MPI
             *
             * @param config_     ParameterTree object containing configuration
             * @param fileName    name of file to read, empty string to not read anything
             * @param loadBalance instance of the load balancer, if needed
             * @param comm        MPI communicator for parallel field generation
             *
             * @see generate
             */
            template<typename LoadBalance = DefaultLoadBalance<Traits::dim>>
              RandomFieldList(
                  const Dune::ParameterTree& config_,
                  const std::string& fileName = "",
                  const LoadBalance loadBalance = LoadBalance(),
                  const MPI_Comm comm = MPI_COMM_WORLD
                  )
              : config(config_)
              {
                std::stringstream typeStream(config.get<std::string>("randomField.types"));
                std::string type;
                while(std::getline(typeStream, type, ' '))
                {
                  fieldNames.push_back(type);

                  Dune::ParameterTree subConfig;
                  Dune::ParameterTreeParser parser;
                  parser.readINITree(type+".field",subConfig);

                  // copy general keys to subConfig if necessary
                  if (!subConfig.hasKey("grid.extensions") && config.hasKey("grid.extensions"))
                    subConfig["grid.extensions"] = config["grid.extensions"];
                  if (!subConfig.hasKey("grid.cells") && config.hasKey("grid.cells"))
                    subConfig["grid.cells"] = config["grid.cells"];
                  if (!subConfig.hasKey("randomField.cgIterations")
                      && config.hasKey("randomField.cgIterations"))
                    subConfig["randomField.cgIterations"] = config["randomField.cgIterations"];

                  std::string subFileName = fileName;
                  if (subFileName != "")
                    subFileName += "." + type;

                  list.insert({type,
                      std::make_shared<SubRandomField>(subConfig,subFileName,loadBalance,comm)});
                }

                if (fieldNames.empty())
                  DUNE_THROW(Dune::Exception,"List of randomField types is empty");

                activateFields(config.get<int>("randomField.active",fieldNames.size()));
              }

            /**
             * @brief Constructor reading random fields and config from file
             *
             * This constructor reads not only the fields, but also their underlying
             * parameters (i.e., the ParameterTree config object) from a file. This can
             * be used to read in a field list that was generated by another program run
             * or another program altogether, and still have the correct parameterization
             * available.
             *
             * @tparam LoadBalance class used for parallel data distribution with MPI
             *
             * @param fileName    name of file to read (base name only)
             * @param loadBalance instance of the load balancer, if needed
             * @param comm        MPI communicator for parallel field generation
             *
             * @see generate
             */
            template<typename LoadBalance = DefaultLoadBalance<Traits::dim>>
              RandomFieldList(
                  const std::string& fileName,
                  const LoadBalance loadBalance = LoadBalance(),
                  const MPI_Comm comm = MPI_COMM_WORLD
                  )
              : treeHelper(fileName), config(treeHelper.get())
              {
                std::stringstream typeStream(config.get<std::string>("randomField.types"));
                std::string type;
                while(std::getline(typeStream, type, ' '))
                {
                  fieldNames.push_back(type);

                  std::string subFileName = fileName + "." + type;

                  list.insert({type, std::make_shared<SubRandomField>(subFileName,loadBalance,comm)});
                }

                if (fieldNames.empty())
                  DUNE_THROW(Dune::Exception,"List of randomField types is empty");

                activateFields(config.get<int>("randomField.active",fieldNames.size()));
              }

            /**
             * @brief Constructor reading random fields from file, but reusing covariance matrices
             *
             * This constructor also reads the fiels from file, but reuses the configuration
             * and covariance matrices from another field list that has already been constructed.
             * This avoids generating the same covariance matrices several times, saving memory
             * when multiple fields have to be read in and subsequently modified.
             *
             * @param other    other random field list to copy most members from
             * @param fileName name of file to read in (base name only)
             *
             * @see generate
             */
            RandomFieldList(const RandomFieldList& other, const std::string& fileName)
              : fieldNames(other.fieldNames), activeTypes(other.activeTypes)
            {
              for(const std::pair<std::string,std::shared_ptr<SubRandomField>>& pair : other.list)
                list.insert({pair.first,
                    std::make_shared<SubRandomField>(*(pair.second),fileName + "." + pair.first)});
            }

#if HAVE_DUNE_PDELAB
            /**
             * @brief Constructor converting from GridFunctionSpace and GridVector
             *
             * This constructor uses a PDELab GridFunctionSpace (representation of the
             * discrete function space on a finite element mesh) and a map of GridVectors
             * (representations of coefficient vectors) to initialize the values of
             * the random fields. Useful to convert sensitivities during Bayesian inversion.
             *
             * @tparam GFS   type of GridFunctionSpace
             * @tparam Field type of coefficient vector
             *
             * @param other other random field list to copy most members from
             * @param gfs   instance of GridFunctionSpace
             * @param field map of coefficient vectors to copy data from
             */
            template<typename GFS, typename FieldList>
              RandomFieldList(const RandomFieldList& other, const GFS& gfs, const FieldList& fieldList)
              : fieldNames(other.fieldNames), activeTypes(other.activeTypes)
              {
                for (const std::string& type : activeTypes)
                {
                  if (fieldList.find(type) == fieldList.end())
                    DUNE_THROW(Dune::Exception,"Field name " + type + " not found in grid function list");

                  std::shared_ptr<SubRandomField> otherField = other.list.find(type)->second;
                  list.insert({type,
                      std::make_shared<SubRandomField>(*otherField,gfs,*(fieldList.find(type)->second))});
                }

                for (const std::string& type : fieldNames)
                  if (fieldList.find(type) == fieldList.end())
                    list.insert({type,
                        std::make_shared<SubRandomField>(*(other.list.find(type)->second))});
              }

            /**
             * @brief Constructor converting from DiscreteGridFunction map
             *
             * This constructor uses a map of PDELab DiscreteGridFunctions (representations
             * of functions living on a finite element mesh) to initialize the values of
             * a random field list. Useful to convert sensitivities during Bayesian inversion.
             *
             * @tparam DGF   type of DiscreteGridFunction
             *
             * @param other other random field list to copy most members from
             * @param dgf   map of DiscreteGridFunctions to copy data from
             */
            template<typename DGFList>
              RandomFieldList(const RandomFieldList& other, const DGFList& dgfList)
              : fieldNames(other.fieldNames), activeTypes(other.activeTypes)
              {
                for (const std::string& type : activeTypes)
                {
                  if (dgfList.find(type) == dgfList.end())
                    DUNE_THROW(Dune::Exception,"Field name " + type + " not found in grid function list");

                  std::shared_ptr<SubRandomField> otherField = other.list.find(type)->second;
                  list.insert({type,
                      std::make_shared<SubRandomField>(*otherField,*(dgfList.find(type)->second))});
                }

                for (const std::string& type : fieldNames)
                  if (dgfList.find(type) == dgfList.end())
                    list.insert({type,
                        std::make_shared<SubRandomField>(*(other.list.find(type)->second))});
              }
#endif // HAVE_DUNE_PDELAB

            /**
             * @brief Copy constructor
             *
             * Standard copy constructor, creating a copy of a given random field list,
             * sharing the covariance matrices between all instances.
             *
             * @param other other random field list that should be copied
             */
            RandomFieldList(const RandomFieldList& other)
              : fieldNames(other.fieldNames), activeTypes(other.activeTypes)
            {
              for(const std::pair<std::string,std::shared_ptr<SubRandomField>>& pair : other.list)
                list.insert({pair.first, std::make_shared<SubRandomField>(*(pair.second))});
            }

            /**
             * @brief Assignment operator
             *
             * Standard assignment operator, sharing the covariance matrices
             * between all instances.
             *
             * @param other other random field that should be copied
             */
            RandomFieldList& operator=(const RandomFieldList& other)
            {
              if (this != &other)
              {
                fieldNames  = other.fieldNames;
                activeTypes = other.activeTypes;

                list.clear();
                for(const std::pair<std::string,std::shared_ptr<SubRandomField>>& pair : other.list)
                  list.insert({pair.first, std::make_shared<SubRandomField>(*(pair.second))});
              }

              return *this;
            }

            /**
             * @brief Insert additional random field into list
             *
             * This adds a random field to the list, assigning a name with which it
             * can be referenced. By default, this field is modified when the field list
             * is used in mathematical operations. Pass false as the third argument to
             * ensure that the random field is kept constant.
             *
             * @param type     name of the new random field
             * @param field    new random field for the list
             * @param activate true if field should participate in mathematical operations
             */
            void insert(const std::string& type, const SubRandomField& field, bool activate = true)
            {
              fieldNames.push_back(type);
              if (activate)
                activeTypes.push_back(type);

              list.insert({type, std::make_shared<SubRandomField>(field)});
            }

            /**
             * @brief Define subset of fields kept constant (i.e., not changed by mathematical operators)
             *
             * Activate the given number of fields, in the order they were added to the list.
             * Activated fields participate in mathematical operations and are changed when
             * another field list is added, subtracted, etc. If this number is smaller than
             * the number of fields that are stored, the remaining fiels are kept constant and
             * will not be modified by such operations.
             *
             * @param number number of fields to activate
             */
            void activateFields(const unsigned int number)
            {
              if (number > fieldNames.size())
                DUNE_THROW(Dune::Exception,"Too many randomFields activated");

              activeTypes.clear();
              for (unsigned int i = 0; i < number; i++)
                activeTypes.push_back(fieldNames[i]);
            }

            /**
             * @brief Number of degrees of freedom (for active types)
             *
             * This counts the total number of degrees of freedom across all active types.
             * In an optimization problem based on random field lists, this is the dimension
             * of the parameter space, since the coefficients of the inactive fields cannot
             * be modified.
             *
             * @return sum of dofs of constituent fields
             */
            unsigned int dofs() const
            {
              unsigned int output = 0;

              for (const std::string& type : activeTypes)
                output += list.find(type)->second->dofs();

              return output;
            }

            /**
             * @brief Generate fields with the desired correlation structure
             *
             * Generate a list of random field samples based on the configured
             * variances, covariance functions, etc. By default, this function
             * does not proceed if a non-World MPI communicator has been
             * configured, because it would, e.g., silently generate different
             * random fields on each of the equivalence classes defined by separate
             * communicators. If you are sure that you want to do that (e.g.,
             * because you are running separate MCMC chains on different processors),
             * you can pass true as an argument to disable the check.
             *
             * @param allowNonWorldComm prevent inconsistent field generation by default
             */
            void generate(bool allowNonWorldComm = false)
            {
              for(const std::string& type : fieldNames)
                list.find(type)->second->generate(allowNonWorldComm);
            }

            /**
             * @brief Generate fields without correlation structure (i.e. noise)
             *
             * This is a convenience function that generates white noise on the grid.
             * Some applications require such white noise, and with this function it
             * can be generated without defining a second type of random field, or
             * applying circulant embedding for a case where it isn't actually needed.
             *
             * @param allowNonWorldComm prevent inconsistent field generation by default
             */
            void generateUncorrelated(bool allowNonWorldComm = false)
            {
              for(const std::string& type : fieldNames)
                list.find(type)->second->generateUncorrelated(allowNonWorldComm);
            }

            /**
             * @brief Vector of random field types currently active
             *
             * This returns a list of currently active random fields, which can then
             * be accessed using the get method.
             *
             * @return vector of strings containing active types
             *
             * @see get
             */
            const std::vector<std::string> types() const
            {
              return activeTypes;
            }

            /**
             * @brief Access to individual random field
             *
             * @param type name of random field to access
             *
             * @return shared pointer to desired field, empty pointer if not found
             */
            const std::shared_ptr<SubRandomField>& get(const std::string& type) const
            {
              if (list.find(type) != list.end())
                return (list.find(type))->second;

              return emptyPointer;
            }

            /**
             * @brief Export random fields to files on disk
             *
             * This function writes the random fields, their trend components, and their
             * configuration into files on disk, so that they can be made persistent
             * and possibly read in again using the corresponding constructor.
             *
             * @param fileName base file name that should be used
             */
            void writeToFile(const std::string& fileName) const
            {
              for(const std::string& type : fieldNames)
                list.find(type)->second->writeToFile(fileName + "." + type);

              std::ofstream file(fileName+".fieldList",std::ofstream::trunc);
              config.report(file);
            }

            /**
             * @brief Export random fields as flat unstructured VTK files, requires dune-grid and dune-functions
             *
             * This function writes the random fields to a VTK file, i.e., for each field
             * the sum of the spatially distributed part and all present trend components.
             * These are the same values as returned by evaluate.
             *
             * @param fileName file name for VTK output
             * @param gv       Dune GridView defining the grid used in the VTK file
             */
            template<typename GridView>
              void writeToVTK(const std::string& fileName, const GridView& gv) const
              {
#if HAVE_DUNE_FUNCTIONS
                for (const std::string& type : fieldNames)
                  list.find(type)->second->writeToVTK(fileName + "." + type,gv);
#else //HAVE_DUNE_FUNCTIONS
                DUNE_THROW(Dune::NotImplemented,"Unstructured VTK output requires dune-grid and dune-functions");
#endif //HAVE_DUNE_FUNCTIONS
              }

            /**
             * @brief Export random fields as unstructured VTK files, requires dune-grid and dune-functions
             *
             * This function writes the individual components of each random field to a
             * VTK file, i.e., the spatially distributed part and all present trend
             * components each as a separate data set.
             *
             * @param fileName file name for VTK output
             * @param gv       Dune GridView defining the grid used in the VTK file
             */
            template<typename GridView>
              void writeToVTKSeparate(const std::string& fileName, const GridView& gv) const
              {
#if HAVE_DUNE_FUNCTIONS
                for (const std::string& type : fieldNames)
                  list.find(type)->second->writeToVTKSeparate(fileName + "." + type,gv);
#else //HAVE_DUNE_FUNCTIONS
                DUNE_THROW(Dune::NotImplemented,"Unstructured VTK output requires dune-grid and dune-functions");
#endif //HAVE_DUNE_FUNCTIONS
              }

            /**
             * @brief Export random fields as flat Legacy VTK files
             *
             * Same as writeToVTK, but writes a simple legacy VTK format and doesn't depend
             * on dune-grid and dune-functions. Unfortunately, this function doesn't work
             * for parallel runs, only for sequential field creation.
             *
             * @param fileName file name for VTK output
             *
             * @see writeToVTK
             */
            void writeToLegacyVTK(const std::string& fileName) const
            {
              for (const std::string& type : fieldNames)
                list.find(type)->second->writeToLegacyVTK(fileName + "." + type);
            }

            /**
             * @brief Export random fields as separate Legacy VTK entries
             *
             * Same as writeToVTKSeparate, but writes a simple legacy VTK format and
             * doesn't depend on dune-grid and dune-functions. Unfortunately, this function
             * doesn't work for parallel runs, only for sequential field creation.
             *
             * @param fileName file name for VTK output
             *
             * @see writeToVTK
             */
            void writeToLegacyVTKSeparate(const std::string& fileName) const
            {
              for (const std::string& type : fieldNames)
                list.find(type)->second->writeToLegacyVTKSeparate(fileName + "." + type);
            }

            /**
             * @brief Set the random fields to zero
             *
             * This function sets both the field values and the trend coefficients
             * of the random fields to zero, creating a list of fields that represent
             * a constant function with value zero.
             */
            void zero()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->zero();
            }

            /**
             * @brief Double spatial resolution of covariance matrices
             *
             * This function instructs the covariance matrices to subdivide each cell
             * in each dimension and recompute themselves. It is not a part of the refine
             * method for design reasons, since several random fields can share a
             * covariance matrix, i.e., this method should only be called once, and
             * then refine should be called on all fields sharing the matrix instance.
             */
            void refineMatrix()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->refineMatrix();
            }

            /**
             * @brief Double spatial resolution of random fields
             *
             * This function subdivides each cell in each dimension and interpolates
             * the random fields. If available, it uses the cached matrix-vector
             * products, refines those instead, and multiplies with the new refined
             * covariance matrices, which yields a smoother interpolation.
             */
            void refine()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->refine();
            }

            /**
             * @brief Reduce spatial resolution of covariance matrix
             *
             * Inverse operation to refineMatrix, merging cells to build larger
             * cells out of them, and then recomputing the covariance matrices.
             *
             * @see refineMatrix
             */
            void coarsenMatrix()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->coarsenMatrix();
            }

            /**
             * @brief Reduce spatial resolution of random fields
             *
             * Inverser operation to refine, merges cells by averaging their
             * values. Just like refine, makes use of cached matrix-vector
             * products if they are available.
             *
             * @see refine
             */
            void coarsen()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->coarsen();
            }

            /**
             * @brief Addition assignment operator
             *
             * @param other other random field list that should be added
             *
             * @return reference to updated random field list
             */
            RandomFieldList& operator+=(const RandomFieldList& other)
            {
              for(const std::string& type : activeTypes)
              {
                if (other.list.find(type) == other.list.end())
                  DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator+=");

                list.find(type)->second->operator+=(*(other.list.find(type)->second));
              }

              return *this;
            }

            /**
             * @brief Subtraction assignment operator
             *
             * @param other other random field list that should be subtracted
             *
             * @return reference to updated random field list
             */
            RandomFieldList& operator-=(const RandomFieldList& other)
            {
              for(const std::string& type : activeTypes)
              {
                if (other.list.find(type) == other.list.end())
                  DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator+=");

                list.find(type)->second->operator-=(*(other.list.find(type)->second));
              }

              return *this;
            }

            /**
             * @brief Multiplication with scalar
             *
             * @param alpha scale factor
             *
             * @return reference to updated random field list
             */
            RandomFieldList& operator*=(const RF alpha)
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->operator*=(alpha);

              return *this;
            }

            /**
             * @brief AXPY scaled addition
             *
             * Adds a multiple of another random field list.
             *
             * @param other other random field list to add
             * @param alpha scale factor for other field list
             *
             * @return reference to updated random field list
             */
            RandomFieldList& axpy(const RandomFieldList& other, const RF alpha)
            {
              for(const std::string& type : activeTypes)
              {
                if (other.list.find(type) == other.list.end())
                  DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in axpy");

                list.find(type)->second->axpy(*(other.list.find(type)->second),alpha);
              }

              return *this;
            }

            /**
             * @brief AXPY scaled addition
             *
             * Second version of scaled addition, with swapped arguments.
             *
             * @param other other random field list to add
             * @param alpha scale factor for other field list
             *
             * @return reference to updated random field list
             */
            RandomFieldList& axpy(const RF alpha, const RandomFieldList& other)
            {
              return axpy(other,alpha);
            }

            /**
             * @brief Scalar product
             *
             * Sum of scalar product of constituent random fields.
             *
             * @param other other random field list to multiply with
             *
             * @return resulting scalar value
             */
            RF operator*(const RandomFieldList& other) const
            {
              RF output = 0.;

              for(const std::string& type : activeTypes)
              {
                if (other.list.find(type) == other.list.end())
                  DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator*");

                output += list.find(type)->second->operator*(*(other.list.find(type)->second));
              }

              return output;
            }

            /**
             * @brief Multiply random fields with covariance matrix
             *
             * Multiplies the random fields with their covariance matrix, useful
             * for Bayesian inversion. Caches the original fields as the
             * matrix-vector product of the resulting field with the inverse
             * of the covariance matrix if configured to do so. This makes
             * evaluating corresponding objective functions very cheap.
             */
            void timesMatrix()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->timesMatrix();
            }

            /**
             * @brief Multiply random fields with inverse of covariance matrix
             *
             * Multiplies the random fields with the inverse of their covariance matrix,
             * as is necessary in Bayesian inversion. Makes use of cached matrix-vector
             * products if configured to do so and they are available.
             */
            void timesInverseMatrix()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->timesInverseMatrix();
            }

            /**
             * @brief Multiply random fields with approximate root of cov. matrix
             *
             * Multiplies the random fields with an approximation of the root of
             * their covariance matrix. The field is embedded in the extended
             * circulant embedding domain, multiplied with the root of the
             * extended covariance matrix (which is known exactly), and then
             * restricted to the original domain. This introduces boundary effects,
             * and therefore this matrix-vector products are not exact.
             */
            void timesMatrixRoot()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->timesMatrixRoot();
            }

            /**
             * @brief Multiply random fields with approximate inverse root of cov. matrix
             *
             * Same as timesMatrixRoot, but with the inverse of the root, instead of
             * the root itself. Introduces the same kind of boundary effects.
             *
             * @see timesMatrixRoot
             */
            void timesInvMatRoot()
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->timesInvMatRoot();
            }

            /**
             * @brief One-norm (sum of absolute values)
             *
             * Sum of the one-norms of the constituent fields.
             *
             * @return resulting scalar value
             */
            RF oneNorm() const
            {
              RF sum = 0.;
              for(const std::string& type : activeTypes)
                sum += list.find(type)->second->oneNorm();

              return sum;
            }

            /**
             * @brief Euclidean norm
             *
             * Euclidean norm of the field list, i.e., the square root of
             * the sum of squares of the Euclidean norms of the individual
             * fields.
             *
             * @return resulting scalar value
             */
            RF twoNorm() const
            {
              RF sum = 0.;
              for(const std::string& type : activeTypes)
                sum += std::pow(list.find(type)->second->twoNorm(),2);

              return std::sqrt(sum);
            }

            /**
             * @brief Maximum norm
             *
             * Maximum norm of the field list, i.e., maximum of the maximum
             * norms of the individual fields.
             *
             * @return resulting scalar value
             */
            RF infNorm() const
            {
              RF max = 0;
              for (const std::string& type : activeTypes)
                max = std::max(max, list.find(type)->second->infNorm());

              return max;
            }

            /**
             * @brief Equality operator
             *
             * @param other other random field lst to compare with
             *
             * @return true if all field values are the same, else false
             */
            bool operator==(const RandomFieldList& other) const
            {
              bool same = true;
              for (const std::string& type : fieldNames)
              {
                if (other.list.find(type) == other.list.end())
                  DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator==");

                if (!(list.find(type)->second->operator==(*(other.list.find(type)->second))))
                {
                  same = false;
                  break;
                }
              }

              return same;
            }

            /**
             * @brief Inequality operator
             *
             * @param other other random field list to compare with
             *
             * @return true if operator== would return false, else false
             *
             * @see operator==
             */
            bool operator!=(const RandomFieldList& other) const
            {
              return !operator==(other);
            }

            /**
             * @brief Multiply fields with Gaussian with given center and radius
             *
             * This is a helper function that dampens the random fields except for a
             * spherical region around a given location. Each field is multiplied
             * with a Gaussian with height one.
             *
             * @param center location of maximum of Gaussian
             * @param radius scale parameter (standard deviation of Gaussian)
             */
            void localize(const typename GridTraits::Domain& center, const RF radius)
            {
              for(const std::string& type : activeTypes)
                list.find(type)->second->localize(center,radius);
            }

        };
  }
}

#endif // DUNE_RANDOMFIELD_RANDOMFIELD_HH
