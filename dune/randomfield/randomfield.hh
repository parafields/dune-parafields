// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_RANDOMFIELD_HH
#define	DUNE_RANDOMFIELD_RANDOMFIELD_HH

#include<dune/common/parametertree.hh>
#include<dune/common/shared_ptr.hh>

#if HAVE_DUNE_PDELAB
// for VTK output functionality
#include<dune/grid/yaspgrid.hh>
#include<dune/grid/io/file/vtk.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#endif // HAVE_DUNE_PDELAB

// file check
#include<sys/stat.h>

#include<mpi.h>

// hdf5 support
#include<hdf5.h>

#include <fftw3.h>
#include <fftw3-mpi.h>

#include<dune/randomfield/io.hh>
#include<dune/randomfield/fieldtraits.hh>
#include<dune/randomfield/trend.hh>
#include<dune/randomfield/stochastic.hh>
#include<dune/randomfield/matrix.hh>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Gaussian random field in 2D or 3D
     */
    template<typename GridTraits, bool storeInvMat = true, bool storeInvRoot = false>
      class RandomField
      {

        public:

          typedef RandomFieldTraits<GridTraits,storeInvMat,storeInvRoot> Traits;
          typedef typename Traits::RF                                    RF;

        private:

          const Dune::ParameterTree                         config;
          Dune::shared_ptr<Traits>                          traits;
          Dune::shared_ptr<RandomFieldMatrix<Traits> >      matrix;
          TrendPart<Traits>                                 trendPart;
          StochasticPart<Traits>                            stochasticPart;
          mutable Dune::shared_ptr<StochasticPart<Traits> > invMatPart;
          mutable bool                                      invMatValid;
          mutable Dune::shared_ptr<StochasticPart<Traits> > invRootPart;
          mutable bool                                      invRootValid;

        public:

          /**
           * @brief Constructor reading from file or creating homogeneous field
           */
          template<typename LoadBalance = DefaultLoadBalance<GridTraits::dim> >
            RandomField(const Dune::ParameterTree& config_, const std::string fileName = "", const LoadBalance loadBalance = LoadBalance())
            : config(config_), traits(new Traits(config,loadBalance)), matrix(new RandomFieldMatrix<Traits>(traits)),
            trendPart(config,traits,fileName), stochasticPart(traits,fileName),
            invMatValid(false), invRootValid(false)
            {
              if (storeInvMat)
                invMatPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(stochasticPart));

              if (storeInvRoot)
                invRootPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(stochasticPart));
            }

          /**
           * @brief Constructor copying traits and covariance matrix
           */
          RandomField(const RandomField& other, const std::string fileName)
            : config(other.config), traits(other.traits), matrix(other.matrix),
            trendPart(config,traits,fileName), stochasticPart(traits,fileName),
            invMatValid(false), invRootValid(false)
        {
          if (storeInvMat)
            invMatPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(stochasticPart));

          if (storeInvRoot)
            invRootPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(stochasticPart));
        }

          /**
           * @brief Copy constructor
           */
          RandomField(const RandomField& other)
            : config(other.config), traits(other.traits), matrix(other.matrix),
            trendPart(other.trendPart), stochasticPart(other.stochasticPart),
            invMatValid(other.invMatValid), invRootValid(other.invRootValid)
        {
          if (storeInvMat)
            invMatPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(*(other.invMatPart)));

          if (storeInvRoot)
            invRootPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(*(other.invRootPart)));
        }

          /**
           * @brief Assignment operator
           */
          RandomField& operator=(const RandomField& other)
          {
            config         = other.config;
            traits         = other.traits;
            matrix         = other.matrix;
            trendPart      = other.trendPart;
            stochasticPart = other.stochasticPart;
            invMatValid    = other.invMatValid;
            invRootValid   = other.invRootValid;

            if (storeInvMat && invMatValid)
              invMatPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(*(other.invMatPart)));

            if (storeInvRoot && invRootValid)
              invRootPart = std::shared_ptr<StochasticPart<Traits> >(new StochasticPart<Traits>(*(other.invRootPart)));

            return *this;
          }

          /**
           * @brief Cell volume of the random field discretization
           */
          RF cellVolume() const
          {
            return (*traits).cellVolume;
          }

          /**
           * @brief Generate a field with the desired correlation structure
           */
          void generate()
          {
            (*matrix).generateField(stochasticPart);
            trendPart.generate();
          }

          /**
           * @brief Generate a field without correlation structure (i.e. noise)
           */
          void generateUncorrelated()
          {
            (*matrix).generateUncorrelatedField(stochasticPart);
            trendPart.generateUncorrelated();
          }

#if HAVE_DUNE_GRID
          /**
           * @brief Evaluate the random field in the coordinates of an element
           */
          template<typename Element>
            void evaluate(const Element& elem, const typename Traits::DomainType& xElem, typename Traits::RangeType& output) const
            {
              const typename Traits::DomainType location = elem.geometry().global(xElem);
              evaluate(location,output);
            }
#endif // HAVE_DUNE_GRID

          /**
           * @brief Evaluate the random field at given coordinates
           */
          void evaluate(const typename Traits::DomainType& location, typename Traits::RangeType& output) const
          {
            typename Traits::RangeType stochastic = 0., trend = 0.;

            stochasticPart.evaluate(location,stochastic);
            trendPart     .evaluate(location,trend);

            output = stochastic + trend;
          }

          /**
           * @brief Export random field to files on disk
           */
          void writeToFile(const std::string& fileName) const
          {
            stochasticPart.writeToFile(fileName);
            trendPart     .writeToFile(fileName);
          }

#if HAVE_DUNE_PDELAB
          /**
           * @brief Export random field as VTK file, requires dune-grid and PDELab
           */
          template<typename GridView>
            void writeToVTK(const std::string& fileName, const GridView& gv) const
            {
              Dune::VTKWriter<GridView> vtkWriter(gv,Dune::VTK::conforming);
              std::shared_ptr<Dune::PDELab::VTKGridFunctionAdapter<RandomField<GridTraits,storeInvMat,storeInvRoot> > > fieldPtr(new Dune::PDELab::VTKGridFunctionAdapter<RandomField<GridTraits,storeInvMat,storeInvRoot> >(*this,fileName));
              vtkWriter.addCellData(fieldPtr);
              vtkWriter.pwrite(fileName,"vtk","",Dune::VTK::appendedraw);
            }
#endif // HAVE_DUNE_PDELAB

          /**
           * @brief Make random field homogeneous
           */
          void zero()
          {
            trendPart     .zero();
            stochasticPart.zero();

            if (storeInvMat)
            {
              (*invMatPart).zero();
              invMatValid = true;
            }

            if (storeInvRoot)
            {
              (*invRootPart).zero();
              invRootValid = true;
            }
          }

          /**
           * @brief Double spatial resolution of covariance matrix
           */
          void refineMatrix()
          {
            (*traits).refine();
            (*matrix).update();
          }

          /**
           * @brief Double spatial resolution of random field
           */
          void refine()
          {
            if (storeInvMat && invMatValid)
            {
              (*invMatPart).refine();
              stochasticPart = (*matrix) * (*invMatPart);

              if ((*traits).dim == 3)
              {
                stochasticPart *= 1./8.;
                *invMatPart    *= 1./8.;
              }
              else
              {
                stochasticPart *= 1./4.;
                *invMatPart    *= 1./4.;
              }

              if (storeInvRoot)
              {
                *invRootPart = (*matrix).multiplyRoot(*invMatPart);

                if ((*traits).dim == 3)
                  *invRootPart *= 1./8.;
                else
                  *invRootPart *= 1./4.;

                invRootValid = true;
              }

            }
            else if (storeInvRoot && invRootValid)
            {
              (*invRootPart).refine();
              stochasticPart = (*matrix).multiplyRoot(*invRootPart);

              if ((*traits).dim == 3)
              {
                stochasticPart *= 1./8.;
                *invRootPart   *= 1./8.;
              }
              else
              {
                stochasticPart *= 1./4.;
                *invRootPart   *= 1./4.;
              }

              if (storeInvMat)
              {
                *invMatPart = stochasticPart;
                invMatValid = false;
              }
            }
            else
            {
              stochasticPart.refine();

              if (storeInvMat)
                (*invMatPart).refine();

              if (storeInvRoot)
                (*invRootPart).refine();
            }
          }

          /**
           * @brief Addition assignment operator
           */
          RandomField& operator+=(const RandomField& other)
          {
            trendPart      += other.trendPart;
            stochasticPart += other.stochasticPart;

            if (storeInvMat)
            {
              *invMatPart += *(other.invMatPart);
              invMatValid = invMatValid && other.invMatValid;
            }

            if (storeInvRoot)
            {
              *invRootPart += *(other.invRootPart);
              invRootValid = invRootValid && other.invRootValid;
            }

            return *this;
          }

          /**
           * @brief Subtraction assignment operator
           */
          RandomField& operator-=(const RandomField& other)
          {
            trendPart      -= other.trendPart;
            stochasticPart -= other.stochasticPart;

            if (storeInvMat)
            {
              *invMatPart -= *(other.invMatPart);
              invMatValid = invMatValid && other.invMatValid;
            }

            if (storeInvRoot)
            {
              *invRootPart -= *(other.invRootPart);
              invRootValid = invRootValid && other.invRootValid;
            }

            return *this;
          }

          /**
           * @brief Multiplication with scalar
           */
          RandomField& operator*=(const RF alpha)
          {
            trendPart      *= alpha;
            stochasticPart *= alpha;

            if (storeInvMat)
            {
              *invMatPart *= alpha;
            }

            if (storeInvRoot)
            {
              *invRootPart *= alpha;
            }

            return *this;
          }

          /**
           * @brief AXPY scaled addition
           */
          RandomField& axpy(const RandomField& other, const RF alpha)
          {
            trendPart     .axpy(other.trendPart     ,alpha);
            stochasticPart.axpy(other.stochasticPart,alpha);

            if (storeInvMat)
            {
              (*invMatPart).axpy(*(other.invMatPart),alpha);
              invMatValid = invMatValid && other.invMatValid;
            }

            if (storeInvRoot)
            {
              (*invRootPart).axpy(*(other.invRootPart),alpha);
              invRootValid = invRootValid && other.invRootValid;
            }

            return *this;
          }

          /**
           * @brief Scalar product
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
           */
          void timesMatrix()
          {
            if (storeInvMat)
            {
              *invMatPart = stochasticPart;
              invMatValid  = true;
            }

            if (storeInvRoot)
            {
              *invRootPart = (*matrix).multiplyRoot(stochasticPart);
              invRootValid = true;
            }

            stochasticPart = (*matrix) * stochasticPart;

            trendPart.timesMatrix();
          }

          /**
           * @brief Multiply random field with inverse of covariance matrix
           */
          void timesInverseMatrix()
          {
            if (storeInvMat && invMatValid)
            {
              if (storeInvRoot)
              {
                *invRootPart = (*matrix).multiplyRoot(*invMatPart);
                invRootValid = true;
              }

              stochasticPart = *invMatPart;
              invMatValid = false;
            }
            else
            {
              stochasticPart = (*matrix).multiplyInverse(stochasticPart);

              if (storeInvMat)
                invMatValid = false;

              if (storeInvRoot)
                invRootValid = false;
            }

            trendPart.timesInverseMatrix();
          }

          /**
           * @brief Multiply random field with approximate root of cov. matrix
           */
          void timesMatrixRoot()
          {
            if (storeInvMat && storeInvRoot)
            {
              *invMatPart = *invRootPart;
              invMatValid = invRootValid;
            }

            if (storeInvRoot)
            {
              *invRootPart = stochasticPart;
              invRootValid = true;
            }

            stochasticPart = (*matrix).multiplyRoot(stochasticPart);

            trendPart.timesMatrixRoot();
          }

          /**
           * @brief Multiply random field with approximate inverse root of cov. matrix
           */
          void timesInvMatRoot()
          {
            if (storeInvRoot && invRootValid)
            {
              stochasticPart = *invRootPart;
              invRootValid = false;

              if (storeInvMat)
              {
                *invRootPart = *invMatPart;
                invRootValid = invMatValid;
                invMatValid  = false;
              }
            }
            else
            {
              stochasticPart = (*matrix).multiplyInverse(stochasticPart);

              if (storeInvRoot)
              {
                *invRootPart = stochasticPart;
                invRootValid = true;
              }

              stochasticPart = (*matrix).multiplyRoot(stochasticPart);

              if (storeInvMat)
                invMatValid = false;
            }

            trendPart.timesInvMatRoot();
          }

          void localize(const typename Traits::DomainType& center, const RF radius)
          {
            stochasticPart.localize(center,radius);

            if (storeInvMat)
              invMatValid = false;

            if (storeInvRoot)
              invRootValid = false;
          }
      };

    /**
     * @brief List of Gaussian random fields in 2D or 3D
     */
    template<typename GridTraits, bool storeInvMat = true, bool storeInvRoot = false>
      class RandomFieldList
      {
        public:

          typedef RandomField<GridTraits, storeInvMat, storeInvRoot> SubRandomField;

        private:

          std::vector<std::string> fieldNames;
          std::vector<std::string> activeTypes;
          std::map<std::string, Dune::shared_ptr<SubRandomField> > list;
          Dune::shared_ptr<SubRandomField> emptyPointer;

          typedef typename GridTraits::RangeField RF;
          typedef typename std::vector<std::string>::const_iterator Iterator;

        public:

          /**
           * @brief Constructor reading random fields from file
           */
          template<typename LoadBalance = DefaultLoadBalance<GridTraits::dim> >
            RandomFieldList(const Dune::ParameterTree& config, const std::string& fileName = "", const LoadBalance loadBalance = LoadBalance())
            {
              std::stringstream typeStream(config.get<std::string>("randomField.types"));
              std::string type;
              while(std::getline(typeStream, type, ' '))
              {
                fieldNames.push_back(type);

                Dune::ParameterTree subConfig;
                Dune::ParameterTreeParser parser;
                parser.readINITree(type+".props",subConfig);
                // copy general keys to subConfig if necessary
                if (!subConfig.hasKey("grid.extensions") && config.hasKey("grid.extensions"))
                  subConfig["grid.extensions"] = config["grid.extensions"];
                if (!subConfig.hasKey("grid.cells") && config.hasKey("grid.cells"))
                  subConfig["grid.cells"] = config["grid.cells"];
                if (!subConfig.hasKey("randomField.cgIterations") && config.hasKey("randomField.cgIterations"))
                  subConfig["randomField.cgIterations"] = config["randomField.cgIterations"];

                std::string subFileName = fileName;
                if (subFileName != "")
                  subFileName += "." + type;

                list.insert(std::pair<std::string,Dune::shared_ptr<SubRandomField> >(type, Dune::shared_ptr<SubRandomField>(new SubRandomField(subConfig,subFileName,loadBalance))));
              }

              if (fieldNames.empty())
                DUNE_THROW(Dune::Exception,"List of randomField types is empty");

              activateFields(config.get<int>("randomField.active",fieldNames.size()));
            }

          /**
           * @brief Constructor reading random fields from file, but reusing covariance matrices
           */
          RandomFieldList(const RandomFieldList& other, const std::string& fileName)
          {
            for(typename std::map<std::string, Dune::shared_ptr<SubRandomField> >::const_iterator it = other.list.begin(); it!= other.list.end(); ++it)
            {
              list.insert(std::pair<std::string,Dune::shared_ptr<SubRandomField> >((*it).first, Dune::shared_ptr<SubRandomField>(new SubRandomField(*(*it).second,fileName+"."+(*it).first))));
            }
          }

          /**
           * @brief Copy constructor
           */
          RandomFieldList(const RandomFieldList& other)
            : fieldNames(other.fieldNames), activeTypes(other.activeTypes)
          {
            for(typename std::map<std::string, Dune::shared_ptr<SubRandomField> >::const_iterator it = other.list.begin(); it!= other.list.end(); ++it)
            {
              list.insert(std::pair<std::string,Dune::shared_ptr<SubRandomField> >((*it).first, Dune::shared_ptr<SubRandomField>(new SubRandomField(*(*it).second))));
            }
          }

          /**
           * @brief Assignment operator
           */
          RandomFieldList& operator=(const RandomFieldList& other)
          {
            fieldNames  = other.fieldNames;
            activeTypes = other.activeTypes;

            list.clear();
            for(typename std::map<std::string, Dune::shared_ptr<SubRandomField> >::const_iterator it = other.list.begin(); it!= other.list.end(); ++it)
            {
              list.insert(std::pair<std::string,Dune::shared_ptr<SubRandomField> >((*it).first, Dune::shared_ptr<SubRandomField>(new SubRandomField(*(*it).second))));
            }

            return *this;
          }

          /**
           * @brief Define subset of fields kept constant (i.e. not changed by calculus operators)
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
           * @brief Generate fields with the desired correlation structure
           */
          void generate()
          {
            for(Iterator it = fieldNames.begin(); it != fieldNames.end(); ++it)
              list.find(*it)->second->generate();
          }

          /**
           * @brief Generate fields without correlation structure (i.e. noise)
           */
          void generateUncorrelated()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->generateUncorrelated();
          }

          /**
           * @brief Vector of random field types currently active
           */
          const std::vector<std::string> types() const
          {
            return activeTypes;
          }

          /**
           * @brief Access to individual random field
           */
          const Dune::shared_ptr<SubRandomField>& get(const std::string& type) const
          {
            if (list.find(type) != list.end())
              return (list.find(type))->second;

            return emptyPointer;
          }

          /**
           * @brief Export random fields to files on disk
           */
          void writeToFile(const std::string& fileName) const
          {
            for(Iterator it = fieldNames.begin(); it != fieldNames.end(); ++it)
              list.find(*it)->second->writeToFile(fileName+"."+(*it));
          }

#if HAVE_DUNE_PDELAB
          /**
           * @brief Export random fields as VTK files, requires dune-grid and PDELab
           */
          template<typename GridView>
            void writeToVTK(const std::string& fileName, const GridView& gv) const
            {
              for (Iterator it = fieldNames.begin(); it != fieldNames.end(); ++it)
                list.find(*it)->second->writeToVTK(fileName+"."+(*it),gv);
            }
#endif // HAVE_DUNE_PDELAB

          /**
           * @brief Set the random fields to zero
           */
          void zero()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->zero();
          }

          /**
           * @brief Double spatial resolution of covariance matrix
           */
          void refineMatrix()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->refineMatrix();
          }

          /**
           * @brief Double spatial resolution of random fields
           */
          void refine()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->refine();
          }

          /**
           * @brief Addition assignment operator
           */
          RandomFieldList& operator+=(const RandomFieldList& other)
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
            {
              if (other.list.find(*it) == other.list.end())
                DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator+=");

              list.find(*it)->second->operator+=(*(other.list.find(*it)->second));
            }

            return *this;
          }

          /**
           * @brief Subtraction assignment operator
           */
          RandomFieldList& operator-=(const RandomFieldList& other)
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
            {
              if (other.list.find(*it) == other.list.end())
                DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator+=");

              list.find(*it)->second->operator-=(*(other.list.find(*it)->second));
            }

            return *this;
          }

          /**
           * @brief Multiplication with scalar
           */
          RandomFieldList& operator*=(const RF alpha)
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->operator*=(alpha);

            return *this;
          }

          /**
           * @brief AXPY scaled addition
           */
          RandomFieldList& axpy(const RandomFieldList& other, const RF alpha)
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
            {
              if (other.list.find(*it) == other.list.end())
                DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in axpy");

              list.find(*it)->second->axpy(*(other.list.find(*it)->second),alpha);
            }

            return *this;
          }

          /**
           * @brief Scalar product
           */
          RF operator*(const RandomFieldList& other) const
          {
            RF output = 0.;

            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
            {
              if (other.list.find(*it) == other.list.end())
                DUNE_THROW(Dune::Exception,"RandomFieldLists don't match in operator*");

              output += list.find(*it)->second->operator*(*(other.list.find(*it)->second));
            }

            return output;
          }

          /**
           * @brief Multiply random fields with covariance matrix
           */
          void timesMatrix()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->timesMatrix();
          }

          /**
           * @brief Multiply random fields with inverse of covariance matrix
           */
          void timesInverseMatrix()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->timesInverseMatrix();
          }

          /**
           * @brief Multiply random fields with approximate root of cov. matrix
           */
          void timesMatrixRoot()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->timesMatrixRoot();
          }

          /**
           * @brief Multiply random fields with approximate inverse root of cov. matrix
           */
          void timesInvMatRoot()
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->timesInvMatRoot();
          }

          void localize(const typename GridTraits::Domain& center, const RF radius)
          {
            for(Iterator it = activeTypes.begin(); it != activeTypes.end(); ++it)
              list.find(*it)->second->localize(center,radius);
          }

      };
  }
}

#endif // DUNE_RANDOMFIELD_RANDOMFIELD_HH
