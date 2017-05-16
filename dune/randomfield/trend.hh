// -*- tab-width: 2; indent-tabs-mtode: nil -*-
#ifndef DUNE_RANDOMFIELD_TREND_HH
#define	DUNE_RANDOMFIELD_TREND_HH

#include<fstream>
#include<random>

#include<dune/common/parametertreeparser.hh>
#if HAVE_DUNE_PDELAB
#include<dune/pdelab/gridfunctionspace/localfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/lfsindexcache.hh>
#endif //HAVE_DUNE_PDELAB

namespace Dune {
  namespace RandomField {

    /**
     * @brief Predefined types of trend component
     */
    struct TrendComponentType
    {

      enum Type {Mean, Slope, Disk, Block};

      static bool isMean(Type i)
      {
        return (i == Mean);
      }

      static bool isSlope(Type i)
      {
        return (i == Slope);
      }

      static bool isDisk(Type i)
      {
        return (i == Disk);
      }

      static bool isBlock(Type i)
      {
        return (i == Block);
      }

    };

    /**
     * @brief Component of random field representing deterministic structure
     */
    template<typename Traits>
      class TrendComponent
      {

        private:

          typedef typename Traits::RF RF;

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
           */
          TrendComponent<Traits>(const std::shared_ptr<Traits>& traits_, const std::vector<RF>& trendVector, const std::vector<RF>& meanVector_, const std::vector<RF>& varianceVector_, const TrendComponentType::Type& componentType_, unsigned int componentCount_ = 0)
            : traits(traits_), componentType(componentType_), componentCount(componentCount_), extensions((*traits).extensions),
            shiftVector(trendVector), meanVector(meanVector_), varianceVector(varianceVector_)
        {
          if (trendVector.size() != meanVector.size() || trendVector.size() != varianceVector.size())
            DUNE_THROW(Dune::Exception,"trend component size does not match");

          if (TrendComponentType::isMean(componentType) && trendVector.size() != 1)
            DUNE_THROW(Dune::Exception,"Trend mean component must only contain one parameter");
          if (TrendComponentType::isSlope(componentType) && trendVector.size() != dim)
            DUNE_THROW(Dune::Exception,"Trend slope component must contain dim parameters: slope in each dimension");
          if (TrendComponentType::isDisk(componentType) && trendVector.size() != dim+2)
            DUNE_THROW(Dune::Exception,"Trend disk component must contain dim+2 parameters: position, radius, value");
          if (TrendComponentType::isBlock(componentType) && trendVector.size() != (2*dim)+1)
            DUNE_THROW(Dune::Exception,"Trend block component must contain (2*dim)+1 parameters: center, extent, value");

          for (unsigned int i = 0; i < shiftVector.size(); i++)
            shiftVector[i] -= meanVector[i];
        }

#if HAVE_DUNE_PDELAB
          template<typename GFS, typename Field>
            void construct(const GFS& gfs, const Field& field)
            {
              std::vector<RF> newShiftVector(shiftVector.size(),0.), myNewShiftVector(shiftVector.size(),0.);

              typedef Dune::PDELab::LocalFunctionSpace<GFS> LFS;
              typedef Dune::PDELab::LFSIndexCache<LFS> LFSCache;
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
                const typename Traits::DomainType& x = elem.geometry().global(elem.geometry().center());
                evaluate(x,shift);
                for (unsigned int i = 0; i < shiftVector.size(); i++)
                {
                  shiftVector[i] += delta;
                  evaluate(x,deltaShift);
                  shiftVector[i] -= delta;

                  myNewShiftVector[i] += (deltaShift[0] - shift[0]) / delta * vLocal[0];
                }
              }

              MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),shiftVector.size(),MPI_DOUBLE,MPI_SUM,(*traits).comm);
              shiftVector = newShiftVector;

            }
#endif // HAVE_DUNE_PDELAB

          /**
           * @brief Type of this trend component
           */
          TrendComponentType::Type type() const
          {
            return componentType;
          }

          /**
           * @brief Name of type of this trend component
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
           * @brief Generate trend component coefficients with correct variance
           */
          void generate()
          {
            std::vector<RF> newShiftVector(shiftVector.size(),0.), myNewShiftVector(shiftVector.size(),0.);

            if ((*traits).rank == 0)
            {
              // initialize pseudo-random generator
              unsigned int seed = (unsigned int) clock(); // create seed out of current time
              seed += static_cast<unsigned int>(reinterpret_cast<uintptr_t>(this));  // different seeds for different components
              std::default_random_engine generator(seed);
              std::normal_distribution<RF> normalDist(0.,1.);

              for (unsigned int i = 0; i < shiftVector.size(); i++)
              {
                myNewShiftVector[i] = normalDist(generator) * std::sqrt(varianceVector[i]);
              }
            }

            MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),shiftVector.size(),MPI_DOUBLE,MPI_SUM,(*traits).comm);
            shiftVector = newShiftVector;
          }

          /**
           * @brief Generate trend component coefficients that are noise
           */
          void generateUncorrelated()
          {
            std::vector<RF> newShiftVector(shiftVector.size(),0.), myNewShiftVector(shiftVector.size(),0.);

            if ((*traits).rank == 0)
            {
              // initialize pseudo-random generator
              unsigned int seed = (unsigned int) clock(); // create seed out of current time
              seed += static_cast<unsigned int>(reinterpret_cast<uintptr_t>(this));  // different seeds for different components
              std::default_random_engine generator(seed);
              std::normal_distribution<RF> normalDist(0.,1.);

              for (unsigned int i = 0; i < shiftVector.size(); i++)
              {
                myNewShiftVector[i] = normalDist(generator);
              }
            }

            MPI_Allreduce(&(myNewShiftVector[0]),&(newShiftVector[0]),shiftVector.size(),MPI_DOUBLE,MPI_SUM,(*traits).comm);
            shiftVector = newShiftVector;
          }

          /**
           * @brief Multiply the trend coefficients with their variances
           */
          void timesMatrix()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= varianceVector[i];
          }

          /**
           * @brief Divide the trend coefficients by their variances
           */
          void timesInverseMatrix()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= 1./varianceVector[i];
          }

          /**
           * @brief Multiply the trend coefficients with their standard deviations
           */
          void timesMatrixRoot()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= std::sqrt(varianceVector[i]);
          }

          /**
           * @brief Divide the trend coefficients by their standard deviations
           */
          void timesInvMatRoot()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= 1./std::sqrt(varianceVector[i]);
          }

          /**
           * @brief Addition assignment operator
           */
          TrendComponent<Traits>& operator+=(const TrendComponent<Traits>& other)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] += other.shiftVector[i];

            return *this;
          }

          /**
           * @brief Subtraction assignment operator
           */
          TrendComponent<Traits>& operator-=(const TrendComponent<Traits>& other)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] -= other.shiftVector[i];

            return *this;
          }

          /**
           * @brief Multiplication with scalar
           */
          TrendComponent<Traits>& operator*=(const RF alpha)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] *= alpha;

            return *this;
          }

          /**
           * @brief AXPY scaled addition
           */
          TrendComponent<Traits>& axpy(const TrendComponent<Traits>& other, const RF alpha)
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] += other.shiftVector[i] * alpha;

            return *this;
          }

          /**
           * @brief Set trend component coefficients to zero
           */
          void zero()
          {
            for (unsigned int i = 0; i < shiftVector.size(); ++i)
              shiftVector[i] = 0.;
          }

          /**
           * @brief Scalar product
           */
          RF operator*(const TrendComponent<Traits>& other) const
          {
            RF output = 0;

            for (unsigned int i = 0; i < shiftVector.size(); i++)
              output += shiftVector[i] * other.shiftVector[i];

            return output;
          }

          /**
           * @brief Evaluate the trend component at a given location
           */
          void evaluate(const typename Traits::DomainType& location, typename Traits::RangeType& output) const
          {
            if (TrendComponentType::isMean(componentType))
            {
              output[0] = meanVector[0] + shiftVector[0];
            }
            else if (TrendComponentType::isSlope(componentType))
            {
              output[0] = 0.;

              for (unsigned int i = 0; i < dim; i++)
                output[0] += (meanVector[i] + shiftVector[i]) * (location[i] - extensions[i]/2.);
            }
            else if (TrendComponentType::isDisk(componentType))
            {
              output[0] = 0.;

              RF distSquared = 0.;
              for (unsigned int i = 0; i < dim; i++)
                distSquared += (location[i] - (meanVector[i] + shiftVector[i])) * (location[i] - (meanVector[i] + shiftVector[i]));

              output[0] = std::exp(- distSquared/((meanVector[dim] + shiftVector[dim])*(meanVector[dim] + shiftVector[dim]))) * (meanVector[dim+1] + shiftVector[dim+1]);
            }
            else if (TrendComponentType::isBlock(componentType))
            {
              output[0] = 0.;

              for (unsigned int i = 0; i < dim; i++)
              {
                if (std::abs(location[i] - (meanVector[i] + shiftVector[i])) > 0.5 * (meanVector[dim+i] + shiftVector[dim+i]))
                  return;
              }

              output[0] = meanVector[2*dim] + shiftVector[2*dim];
            }
            else
              DUNE_THROW(Dune::Exception,"Trend component type not found!");
          }

          /**
           * @brief Write the trend component to hard disk
           */
          void writeToFile(std::ofstream& file, unsigned int count) const
          {
            if ((*traits).rank == 0)
            {
              if (TrendComponentType::isMean(componentType))
              {
                file << "mean =";
              }
              else if (TrendComponentType::isSlope(componentType))
              {
                file << "slope =";
              }
              else if (TrendComponentType::isDisk(componentType))
              {
                file << "disk" << componentCount << " =";
              }
              else if (TrendComponentType::isBlock(componentType))
              {
                file << "block" << componentCount << " =";
              }
              else
                DUNE_THROW(Dune::Exception,"Trend component type not found!");

              for (unsigned int i = 0; i < shiftVector.size(); i++)
                file << " " << meanVector[i] + shiftVector[i];

              file << std::endl;
            }
          }

      };

    /**
     * @brief Part of random field that consists of deterministic components
     */
    template<typename Traits>
      class TrendPart {

        typedef typename Traits::RF RF;

        std::shared_ptr<Traits> traits;
        std::vector<TrendComponent<Traits> > componentVector;

        public:

        /**
         * @brief Constructor
         */
        TrendPart<Traits>(const Dune::ParameterTree& config, const std::shared_ptr<Traits>& traits_, const std::string& fileName = "")
          : traits(traits_)
          {
            std::vector<RF> emptyVector, trendVector, meanVector, varianceVector;

            meanVector = config.get<std::vector<RF> >("mean.mean",emptyVector);

            if (!meanVector.empty())
            {
              varianceVector = config.get<std::vector<RF> >("mean.variance");

              if (fileName == "")
              {
                trendVector = meanVector;
              }
              else
              {
                Dune::ParameterTree trendConfig;
                Dune::ParameterTreeParser parser;
                parser.readINITree(fileName+".trend",trendConfig);
                trendVector = trendConfig.get<std::vector<RF> >("mean");
              }

              componentVector.push_back(TrendComponent<Traits>(traits,trendVector, meanVector,varianceVector,TrendComponentType::Mean));
            }

            meanVector = config.get<std::vector<RF> >("slope.mean",emptyVector);

            if (!meanVector.empty())
            {
              varianceVector = config.get<std::vector<RF> >("slope.variance");

              if (fileName == "")
              {
                trendVector = meanVector;
              }
              else
              {
                Dune::ParameterTree trendConfig;
                Dune::ParameterTreeParser parser;
                parser.readINITree(fileName+".trend",trendConfig);
                trendVector = trendConfig.get<std::vector<RF> >("slope");
              }

              componentVector.push_back(TrendComponent<Traits>(traits,trendVector,meanVector,varianceVector,TrendComponentType::Slope));
            }

            int count = 0;
            std::stringstream s;
            bool endReached = false;

            while(!endReached)
            {
              s.clear();
              s.str(std::string());
              s << count;
              meanVector = config.get<std::vector<RF> >("disk"+s.str()+".mean",emptyVector);

              if (meanVector.empty())
              {
                endReached = true;
              }
              else
              {
                varianceVector = config.get<std::vector<RF> >("disk"+s.str()+".variance");

                if (fileName == "")
                {
                  trendVector = meanVector;
                }
                else
                {
                  Dune::ParameterTree trendConfig;
                  Dune::ParameterTreeParser parser;
                  parser.readINITree(fileName+".trend",trendConfig);
                  trendVector = trendConfig.get<std::vector<RF> >("disk"+s.str());
                }

                componentVector.push_back(TrendComponent<Traits>(traits,trendVector,meanVector,varianceVector,TrendComponentType::Disk,count));

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
              meanVector = config.get<std::vector<RF> >("block"+s.str()+".mean",emptyVector);

              if (meanVector.empty())
              {
                endReached = true;
              }
              else
              {
                varianceVector = config.get<std::vector<RF> >("block"+s.str()+".variance");

                if (fileName == "")
                {
                  trendVector = meanVector;
                }
                else
                {
                  Dune::ParameterTree trendConfig;
                  Dune::ParameterTreeParser parser;
                  parser.readINITree(fileName+".trend",trendConfig);
                  trendVector = trendConfig.get<std::vector<RF> >("block"+s.str());
                }

                componentVector.push_back(TrendComponent<Traits>(traits,trendVector,meanVector,varianceVector,TrendComponentType::Block,count));

                count++;
              }
            }
          }

#if HAVE_DUNE_PDELAB
        template<typename GFS, typename Field>
          TrendPart<Traits>(const TrendPart<Traits>& other, const GFS& gfs, const Field& field)
          : traits(other.traits), componentVector(other.componentVector)
          {
            for (unsigned int i = 0; i < componentVector.size(); i++)
              componentVector[i].construct(gfs,field);
          }
#endif // HAVE_DUNE_PDELAB

        /**
         * @brief Generate a trend part with desired covariance structure
         */
        void generate()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
          {
            componentVector[i].generate();
          }
        }

        /**
         * @brief Generate a trend part without correlation (i.e. noise)
         */
        void generateUncorrelated()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
          {
            componentVector[i].generateUncorrelated();
          }
        }

        /**
         * @brief Number of stored trend components
         */
        unsigned int size() const
        {
          return componentVector.size();
        }

        /**
         * @brief Access ith trend component
         */
        const TrendComponent<Traits>& getComponent(unsigned int i) const
        {
          return componentVector[i];
        }

        /**
         * @brief Multiply the trend part with its covariance matrix
         */
        void timesMatrix()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesMatrix();
        }

        /**
         * @brief Multiply the trend part with the inverse of its covariance matrix
         */
        void timesInverseMatrix()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesInverseMatrix();
        }

        /**
         * @brief Multiply the trend part with approximate root of its cov. matrix
         */
        void timesMatrixRoot()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesMatrixRoot();
        }

        /**
         * @brief Multiply the trend part with approximate inverse root of its cov. matrix
         */
        void timesInvMatRoot()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].timesInvMatRoot();
        }

        /**
         * @brief Addition assignment operator
         */
        TrendPart<Traits>& operator+=(const TrendPart<Traits>& other)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i] += other.componentVector[i];

          return *this;
        }

        /**
         * @brief Subtraction assignment operator
         */
        TrendPart<Traits>& operator-=(const TrendPart<Traits>& other)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i] -= other.componentVector[i];

          return *this;
        }

        /**
         * @brief Multiplication with scalar
         */
        TrendPart<Traits>& operator*=(const RF alpha)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i] *= alpha;

          return *this;
        }

        /**
         * @brief AXPY scaled addition
         */
        TrendPart<Traits>& axpy(const TrendPart<Traits>& other, const RF alpha)
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].axpy(other.componentVector[i],alpha);

          return *this;
        }

        /**
         * @brief Set the trend part to zero
         */
        void zero()
        {
          for (unsigned int i = 0; i < componentVector.size(); i++)
            componentVector[i].zero();
        }

        /**
         * @brief Scalar product
         */
        RF operator*(const TrendPart<Traits>& other) const
        {
          RF output = 0.;

          for (unsigned int i = 0; i < componentVector.size(); i++)
            output += componentVector[i] * other.componentVector[i];

          return output;
        }

        /**
         * @brief Evaluate the trend part at a given location
         */
        void evaluate(const typename Traits::DomainType& x, typename Traits::RangeType& output) const
        {
          output = 0.;
          typename Traits::RangeType compOutput = 0.;

          for (unsigned int i = 0; i < componentVector.size(); i++)
          {
            componentVector[i].evaluate(x,compOutput);
            output[0] += compOutput[0];
          }
        }

        /**
         * @brief Write the trend part to hard disk
         */
        void writeToFile(const std::string& fileName) const
        {
          if ((*traits).rank == 0)
          {
            std::ofstream file(fileName+".trend",std::ofstream::trunc);

            unsigned int count = 0;
            for (unsigned int i = 0; i < componentVector.size(); i++)
            {
              if (i != 0)
                if (componentVector[i].type() != componentVector[i-1].type())
                  count = 0;

              componentVector[i].writeToFile(file,count);
            }
          }
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_TREND_HH
