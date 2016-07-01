// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_RANDOMFIELD_HH
#define	DUNE_RANDOMFIELD_RANDOMFIELD_HH

#include<dune/common/parametertree.hh>
#include<dune/common/shared_ptr.hh>

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
    template<typename GridTraits, typename Covariance>
      class RandomField
      {

        public:

          typedef RandomFieldTraits<GridTraits,Covariance> Traits;
          typedef typename Traits::RF                      RF;

        private:

          std::string                                  fieldName;
          Dune::shared_ptr<Traits>                     traits;
          Dune::shared_ptr<RandomFieldMatrix<Traits> > pMatrix;
          TrendPart<Traits>                            trendPart;
          StochasticPart<Traits>                       stochasticPart;
          //mutable StochasticPart<Traits>               invMatPart;
          //mutable bool                                 invMatValid;
          //mutable StochasticPart<Traits>               invRootPart;
          //mutable bool                                 invRootValid;

        public:

          /**
           * @brief Constructor reading from file or creating homogeneous field
           */
          RandomField(const Dune::ParameterTree& config_, std::string fieldName_, std::string fileName = "")
            : fieldName(fieldName_), traits(new Traits(config_,fieldName)), pMatrix(new RandomFieldMatrix<Traits>(traits)),
            trendPart(traits,fieldName,fileName), stochasticPart(traits,fieldName,fileName)
            //invMatPart(stochasticPart), invMatValid(false), invRootPart(stochasticPart), invRootValid(false)
            {}

          /**
           * @brief Constructor copying traits and covariance matrix
           */
          RandomField(const RandomField& other, std::string fileName)
            : fieldName(other.fieldName), traits(other.traits), pMatrix(other.pMatrix),
            trendPart(traits,fieldName,fileName), stochasticPart(traits,fieldName,fileName)
            //invMatPart(stochasticPart), invMatValid(false), invRootPart(stochasticPart), invRootValid(false)
        {}

          /**
           * @brief Copy constructor
           */
          RandomField(const RandomField& other)
            : fieldName(other.fieldName), traits(other.traits), pMatrix(other.pMatrix),
            trendPart(other.trendPart), stochasticPart(other.stochasticPart)
            //invMatPart(other.invMatPart), invMatValid(other.invMatValid),
            //invRootPart(other.invRootPart), invRootValid(other.invRootValid)
        {}

          /**
           * @brief Assignment operator
           */
          RandomField& operator=(const RandomField& other)
          {
            fieldName      = other.fieldName;
            traits         = other.traits;
            pMatrix        = other.pMatrix;
            trendPart      = other.trendPart;
            stochasticPart = other.stochasticPart;
            //invMatPart     = other.invMatPart;
            //invMatValid    = other.invMatValid;
            //invRootPart    = other.invRootPart;
            //invRootValid   = other.invRootValid;

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
            (*pMatrix).generateField(stochasticPart);
            trendPart.generate();
          }

          /**
           * @brief Generate a field without correlation structure (i.e. noise)
           */
          void generateUncorrelated()
          {
            (*pMatrix).generateUncorrelatedField(stochasticPart);
            trendPart.generateUncorrelated();
          }

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
            stochasticPart.writeToFile(fileName,fieldName);
            trendPart     .writeToFile(fileName,fieldName);
          }

          /**
           * @brief Make random field homogeneous
           */
          void zero()
          {
            trendPart     .zero();
            stochasticPart.zero();
            //invMatPart    .zero();
            //invRootPart   .zero();

            //invMatValid = true;
            //invRootValid  = true;
          }

          /**
           * @brief Double spatial resolution of covariance matrix
           */
          void refineMatrix()
          {
            (*traits).refine();
            (*pMatrix).update();
          }

          /**
           * @brief Double spatial resolution of random field
           */
          void refine()
          {
              /*
            if (invMatValid)
            {
              invMatPart.refine();
              stochasticPart = (*pMatrix) * invMatPart;
              invRootPart    = (*pMatrix).multiplyRoot(invMatPart);

              if ((*traits).dim == 3)
              {
                stochasticPart *= 1./8.;
                invMatPart     *= 1./8.;
                invRootPart    *= 1./8.;
              }
              else
              {
                stochasticPart *= 1./4.;
                invMatPart     *= 1./4.;
                invRootPart    *= 1./4.;
              }
            }
            else if (invRootValid)
            {
              invRootPart.refine();
              stochasticPart = (*pMatrix).multiplyRoot(invRootPart);
              invMatPart     = stochasticPart;

              if ((*traits).dim == 3)
              {
                stochasticPart *= 1./8.;
                invMatPart     *= 1./8.;
                invRootPart    *= 1./8.;
              }
              else
              {
                stochasticPart *= 1./4.;
                invMatPart     *= 1./4.;
                invRootPart    *= 1./4.;
              }

            }
            else
            {
            */
              stochasticPart.refine();
              /*
              invMatPart    .refine();
              invRootPart   .refine();
            }
            */
          }

          /**
           * @brief Addition assignment operator
           */
          RandomField& operator+=(const RandomField& other)
          {
            trendPart      += other.trendPart;
            stochasticPart += other.stochasticPart;
            //invMatPart     += other.invMatPart;
            //invRootPart    += other.invRootPart;

            //invMatValid   = invMatValid   && other.invMatValid;
            //invRootValid  = invRootValid  && other.invRootValid;

            return *this;
          }

          /**
           * @brief Subtraction assignment operator
           */
          RandomField& operator-=(const RandomField& other)
          {
            trendPart      -= other.trendPart;
            stochasticPart -= other.stochasticPart;
            //invMatPart     -= other.invMatPart;
            //invRootPart    -= other.invRootPart;

            //invMatValid   = invMatValid   && other.invMatValid;
            //invRootValid  = invRootValid  && other.invRootValid;

            return *this;
          }

          /**
           * @brief Multiplication with scalar
           */
          RandomField& operator*=(const RF alpha)
          {
            trendPart      *= alpha;
            stochasticPart *= alpha;
            //invMatPart     *= alpha;
            //invRootPart    *= alpha;

            return *this;
          }

          /**
           * @brief AXPY scaled addition
           */
          RandomField& axpy(const RandomField& other, const RF alpha)
          {
            trendPart     .axpy(other.trendPart     ,alpha);
            stochasticPart.axpy(other.stochasticPart,alpha);
            //invMatPart    .axpy(other.invMatPart    ,alpha);
            //invRootPart   .axpy(other.invRootPart   ,alpha);

            //invMatValid   = invMatValid   && other.invMatValid;
            //invRootValid  = invRootValid  && other.invRootValid;

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
            //invMatPart     = stochasticPart;
            //invRootPart    = (*pMatrix).multiplyRoot(stochasticPart);
            stochasticPart = (*pMatrix) * stochasticPart;

            trendPart.timesMatrix();

            //invMatValid  = true;
            //invRootValid = true;
          }

          /**
           * @brief Multiply random field with inverse of covariance matrix
           */
          void timesInverseMatrix()
          {
            //if (!invMatValid)
            //  invMatPart = (*pMatrix).multiplyInverse(stochasticPart);

            //stochasticPart = invMatPart;
            //invRootPart    = (*pMatrix).multiplyRoot(invMatPart);
            stochasticPart = (*pMatrix).multiplyInverse(stochasticPart);

            trendPart.timesInverseMatrix();

            //invMatValid  = false;
            //invRootValid = true;
          }

          /**
           * @brief Multiply random field with approximate root of cov. matrix
           */
          void timesMatrixRoot()
          {
            //invMatPart     = invRootPart;
            //invRootPart    = stochasticPart;
            stochasticPart = (*pMatrix).multiplyRoot(stochasticPart);

            trendPart.timesMatrixRoot();

            //invMatValid  = invRootValid;
            //invRootValid = true;
          }

          /**
           * @brief Multiply random field with approximate inverse root of cov. matrix
           */
          void timesInvMatRoot()
          {
              /*
            if (invRootValid)
            {
              stochasticPart = invRootPart;
              invRootPart    = invMatPart;

              invRootValid = invMatValid;
              invMatValid  = false;
            }
            else
            {
            */
              //if (!invMatValid)
              //  invMatPart = (*pMatrix).multiplyInverse(stochasticPart);

              //invRootPart = invMatPart;
              //stochasticPart = (*pMatrix).multiplyRoot(invRootPart);
              stochasticPart = (*pMatrix).multiplyInverse(stochasticPart);
              stochasticPart = (*pMatrix).multiplyRoot(stochasticPart);

              trendPart.timesInvMatRoot();

              /*
              invMatValid  = false;
              invRootValid = true;
            }
            */
          }

          void localize(const typename Traits::DomainType& center, const RF radius)
          {
            stochasticPart.localize(center,radius);

            /*
            invMatValid  = false;
            invRootValid = false;
            */
          }
      };

    /**
     * @brief List of Gaussian random fields in 2D or 3D
     */
    template<typename GridTraits, typename Covariance>
      class RandomFieldList
      {
        public:

          typedef RandomField<GridTraits, Covariance> SubRandomField;

        private:

          std::vector<std::string> fieldNames;
          std::vector<std::string> activeTypes;
          std::map<std::string, Dune::shared_ptr<SubRandomField> > list;
          Dune::shared_ptr<SubRandomField> emptyPointer;

          mutable std::shared_ptr<std::vector<typename GridTraits::RangeField> > postEigen;
          mutable std::shared_ptr<std::vector<RandomFieldList> >                 postContrib;

          typedef typename GridTraits::RangeField RF;        
          typedef typename std::vector<std::string>::const_iterator Iterator;

        public:

          /**
           * @brief Constructor reading random fields from file
           */
          RandomFieldList(const Dune::ParameterTree& config, const std::string& fileName = "")
            : postEigen(new std::vector<typename GridTraits::RangeField>()), postContrib(new std::vector<RandomFieldList>())
            {
              std::stringstream typeStream(config.get<std::string>("randomField.types"));
              std::string type;
              while(std::getline(typeStream, type, ' '))
              {
                fieldNames.push_back(type);
                list.insert(std::pair<std::string,Dune::shared_ptr<SubRandomField> >(type, Dune::shared_ptr<SubRandomField>(new SubRandomField(config,type,fileName))));
              }

              if (fieldNames.empty())
                DUNE_THROW(Dune::Exception,"List of randomField types is empty");

              activateFields(config.get<int>("randomField.active",fieldNames.size()));
            }

          /**
           * @brief Constructor reading random fields from file, but reusing covariance matrices
           */
          RandomFieldList(const RandomFieldList& other, const std::string& fileName)
            : postEigen(new std::vector<typename GridTraits::RangeField>()), postContrib(new std::vector<RandomFieldList>())
          {
            for(typename std::map<std::string, Dune::shared_ptr<SubRandomField> >::const_iterator it = other.list.begin(); it!= other.list.end(); ++it)
            {
              list.insert(std::pair<std::string,Dune::shared_ptr<SubRandomField> >((*it).first, Dune::shared_ptr<SubRandomField>(new SubRandomField(*(*it).second,fileName))));
            }
          }

          /**
           * @brief Copy constructor
           */
          RandomFieldList(const RandomFieldList& other)
            : fieldNames(other.fieldNames), activeTypes(other.activeTypes), postEigen(other.postEigen), postContrib(other.postContrib)
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
            postEigen   = other.postEigen;
            postContrib = other.postContrib;

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
              list.find(*it)->second->writeToFile(fileName);
          }

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
