#ifndef DUNE_RANDOMFIELD_STOCHASTIC_HH
#define DUNE_RANDOMFIELD_STOCHASTIC_HH

#include<fstream>

#include<dune/common/power.hh>
#if HAVE_DUNE_PDELAB
#include<dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#endif //HAVE_DUNE_PDELAB

#include<dune/randomfield/fieldtraits.hh>

namespace Dune {
  namespace RandomField {

    /*
     * @brief Part of random field that consists of cell values
     */
    template<typename Traits>
      class StochasticPart
      {
        using RF      = typename Traits::RF;
        using Index   = typename Traits::Index;
        using Indices = typename Traits::Indices;

        enum {dim = Traits::dim};

        friend typename Traits::IsoMatrixType;
        friend typename Traits::AnisoMatrixType;

        std::shared_ptr<Traits> traits;

        int rank, commSize;
        std::array<RF,dim>  extensions;
        Indices             cells;
        unsigned int        level;
        Indices             localCells;
        Indices             localOffset;
        Index               localDomainSize;
        Index               sliceSize;
        Indices             localEvalCells;
        Indices             localEvalOffset;
        std::array<int,dim> procPerDim;

        mutable std::vector<RF>              dataVector;
        mutable std::vector<RF>              evalVector;
        mutable std::vector<std::vector<RF>> overlap;

        mutable bool    evalValid;
        mutable Indices cellIndices;
        mutable Indices evalIndices;
        mutable Indices countIndices;

        enum {toEval, fromEval, toOverlap};

        public:

        /**
         * @brief Constructor reading from file or creating homogeneous field
         */
        StochasticPart(
            const std::shared_ptr<Traits>& traits_,
            const std::string& fileName
            )
          :
            traits(traits_)
        {
          update();

          if (fileName != "")
          {
#if HAVE_HDF5
            if(!fileExists(fileName+".stoch.h5"))
              DUNE_THROW(Dune::Exception,"File is missing: " + fileName + ".stoch.h5");

            if ((*traits).verbose && rank == 0)
              std::cout << "loading random field from file " << fileName << std::endl;
            readParallelFromHDF5<RF,dim>(dataVector, localCells, localOffset,
                (*traits).comm, "/stochastic", fileName+".stoch.h5");

            evalValid  = false;
#else //HAVE_HDF5
            DUNE_THROW(Dune::NotImplemented,
                "Writing and reading field files requires parallel HDF5 support");
#endif //HAVE_HDF5
          }
          else
          {
            if ((*traits).verbose && rank == 0)
              std::cout << "generating homogeneous random field" << std::endl;

            zero();
          }
        }

#if HAVE_DUNE_PDELAB
        /**
         * @brief Constructor converting from DiscreteGridFunction
         */
        template<typename DGF>
          StochasticPart(const StochasticPart& other, const DGF& dgf)
          :
            traits(other.traits)
        {
          update();

          using DF         = typename Traits::DomainField;
          using DomainType = typename Traits::DomainType;

          DomainType minCoords;
          DomainType maxCoords;
          DomainType coords;
          Indices minIndices;
          Indices maxIndices;
          Dune::FieldVector<RF,1> value;

          for (const auto& elem : elements(dgf.getGridView(),Dune::Partitions::interior))
          {
            for (unsigned int i = 0; i < dim; i++)
            {
              minCoords[i] =  std::numeric_limits<DF>::max();
              maxCoords[i] = -std::numeric_limits<DF>::max();
            }

            for (int j = 0; j < elem.geometry().corners(); j++)
            {
              const DomainType& corner = elem.geometry().corner(j);

              for (unsigned int i = 0; i < dim; i++)
              {
                const DF coord = corner[i];
                if (coord + 1e-6 < minCoords[i])
                  minCoords[i] = coord + 1e-6;
                if (coord - 1e-6 > maxCoords[i])
                  maxCoords[i] = coord - 1e-6;
              }
            }

            (*traits).coordsToIndices(minCoords,minIndices,localEvalOffset);
            (*traits).coordsToIndices(maxCoords,maxIndices,localEvalOffset);

            evalIndices = minIndices;

            // iterate over all matching indices
            while (true)
            {
              (*traits).indicesToCoords(evalIndices,localEvalOffset,coords);
              const typename Traits::DomainType& local = elem.geometry().local(coords);

              if (referenceElement(elem.geometry()).checkInside(local))
              {
                const Index index
                  = Traits::indicesToIndex(evalIndices,localEvalCells);
                dgf.evaluate(elem,local,value);
                evalVector[index] = value[0];
              }

              // select next set of indices
              unsigned int i;
              for (i = 0; i < dim; i++)
              {
                evalIndices[i]++;
                if (evalIndices[i] <= maxIndices[i])
                  break;
                evalIndices[i] = minIndices[i];
              }
              if (i == dim)
                break;
            }
          }

          evalToData();
          evalValid  = true;
        }

        /**
         * @brief Constructor converting from GridFunctionSpace and GridVector
         */
        template<typename GFS, typename Field>
          StochasticPart(const StochasticPart& other, const GFS& gfs, const Field& field)
          :
            StochasticPart(other,Dune::PDELab::DiscreteGridFunction<GFS,Field>(gfs,field))
        {}
#endif // HAVE_DUNE_PDELAB

        /**
         * @brief Calculate container sizes after construction or refinement
         */
        void update()
        {
          rank            = (*traits).rank;
          commSize        = (*traits).commSize;
          extensions      = (*traits).extensions;
          cells           = (*traits).cells;
          level           = (*traits).level;
          localCells      = (*traits).localCells;
          localOffset     = (*traits).localOffset;
          localDomainSize = (*traits).localDomainSize;
          procPerDim      = (*traits).procPerDim;

          for (unsigned int i = 0; i < dim; i++)
          {
            if (cells[i] % procPerDim[i] != 0)
              DUNE_THROW(Dune::Exception,"cells in dimension not divisable by numProcs");
            localEvalCells[i] = cells[i] / procPerDim[i];
          }
          if (dim == 3)
          {
            localEvalOffset[0] = (rank%(procPerDim[0]*procPerDim[1]))%procPerDim[0] * localEvalCells[0];
            localEvalOffset[1] = (rank%(procPerDim[0]*procPerDim[1]))/procPerDim[0] * localEvalCells[1];
            localEvalOffset[2] =  rank/(procPerDim[0]*procPerDim[1])                * localEvalCells[2];
          }
          else if (dim == 2)
          {
            localEvalOffset[0] = rank%procPerDim[0] * localEvalCells[0];
            localEvalOffset[1] = rank/procPerDim[0] * localEvalCells[1];
          }
          else if (dim == 1)
          {
            localEvalOffset[0] = rank * localEvalCells[0];
          }
          else if ((*traits).verbose)
            std::cout << "Note: dimension of field has to be 1, 2 or 3"
              << " for data redistribution and overlap" << std::endl;

          evalValid = false;
        }

        /**
         * @brief Write stochastic part of field to hard disk
         */
        void writeToFile(const std::string& fileName) const
        {
#if HAVE_HDF5
          if ((*traits).verbose && rank == 0)
            std::cout << "writing random field to file " << fileName << std::endl;

          writeParallelToHDF5<RF,dim>((*traits).cells, dataVector, localCells, localOffset,
              (*traits).comm, "/stochastic", fileName+".stoch.h5");

          if (rank == 0)
            writeToXDMF<RF,dim>((*traits).cells,(*traits).extensions,fileName);
#else //HAVE_HDF5
          DUNE_THROW(Dune::NotImplemented,
              "Writing and reading field files requires parallel HDF5 support");
#endif //HAVE_HDF5
        }

        /**
         * @brief Number of degrees of freedom
         */
        unsigned int dofs() const
        {
          unsigned int output = 0.;

          MPI_Allreduce(&localDomainSize,&output,1,MPI_INT,MPI_SUM,(*traits).comm);
          return output;
        }

        /**
         * @brief Addition assignment operator
         */
        StochasticPart& operator+=(const StochasticPart& other)
        {
          if (!other.dataVector.empty())
          {
            if (dataVector.empty())
            {
              dataVector.resize(localDomainSize);

              for (Index i = 0; i < dataVector.size(); ++i)
                dataVector[i] = other.dataVector[i];
            }
            else
              for (Index i = 0; i < dataVector.size(); ++i)
                dataVector[i] += other.dataVector[i];

            evalValid = false;
          }

          return *this;
        }

        /**
         * @brief Subtraction assignment operator
         */
        StochasticPart& operator-=(const StochasticPart& other)
        {
          if (!other.dataVector.empty())
          {
            if (dataVector.empty())
            {
              dataVector.resize(localDomainSize);

              for (Index i = 0; i < dataVector.size(); ++i)
                dataVector[i] = - other.dataVector[i];
            }
            else
              for (Index i = 0; i < dataVector.size(); ++i)
                dataVector[i] -= other.dataVector[i];

            evalValid = false;
          }

          return *this;
        }

        /**
         * @brief Multiplication with scalar
         */
        StochasticPart& operator*=(const RF alpha)
        {
          if (!dataVector.empty())
          {
            for (Index i = 0; i < dataVector.size(); ++i)
              dataVector[i] *= alpha;

            evalValid = false;
          }

          return *this;
        }

        /**
         * @brief AXPY scaled addition
         */
        StochasticPart& axpy(const StochasticPart& other, const RF alpha)
        {
          if (!other.dataVector.empty())
          {
            if (dataVector.empty())
            {
              dataVector.resize(localDomainSize);

              for (Index i = 0; i < dataVector.size(); ++i)
                dataVector[i] = other.dataVector[i] * alpha;
            }
            else
              for (Index i = 0; i < dataVector.size(); ++i)
                dataVector[i] += other.dataVector[i] * alpha;

            evalValid = false;
          }

          return *this;
        }

        /**
         * @brief Scalar product
         */
        RF operator*(const StochasticPart& other) const
        {
          RF sum = 0., mySum = 0.;

          if (dataVector.size() == other.dataVector.size())
            for (Index i = 0; i < dataVector.size(); ++i)
              mySum += dataVector[i] * other.dataVector[i];

          MPI_Allreduce(&mySum,&sum,1,mpiType<RF>,MPI_SUM,(*traits).comm);
          return sum;
        }

        /**
         * @brief Equality operator
         */
        bool operator==(const StochasticPart& other) const
        {
          int same = true, mySame = true;

          if (dataVector.size() != other.dataVector.size())
            mySame = false;
          else
            for (Index i = 0; i < dataVector.size(); ++i)
              if (dataVector[i] != other.dataVector[i])
              {
                mySame = false;
                break;
              }

          MPI_Allreduce(&mySame,&same,1,MPI_INT,MPI_MIN,(*traits).comm);
          return same;
        }

        /**
         * @brief Inequality operator
         */
        bool operator!=(const StochasticPart& other) const
        {
          return !operator==(other);
        }

        /**
         * @brief Evaluate stochastic part at given location
         */
        void evaluate(
            const typename Traits::DomainType& location,
            typename Traits::RangeType& output
            ) const
        {
          if (!evalValid)
            dataToEval();

          (*traits).coordsToIndices(location,evalIndices,localEvalOffset);

          for (Index i = 0; i < dim; i++)
          {
            if (evalIndices[i] > localEvalCells[i])
              countIndices[i] = 2*i;
            else if (evalIndices[i] == localEvalCells[i])
              countIndices[i] = 2*i+1;
            else
              countIndices[i] = 2*dim;
          }

          if (dim == 3)
          {
            if (countIndices[0] == 2*dim && countIndices[1] == 2*dim
                && countIndices[2] != 2*dim)
            {
              output[0] = overlap[countIndices[2]][evalIndices[0]
                + evalIndices[1]*localEvalCells[0]];
            }
            else if (countIndices[0] == 2*dim && countIndices[1] != 2*dim
                && countIndices[2] == 2*dim)
            {
              output[0] = overlap[countIndices[1]][evalIndices[2]
                + evalIndices[0]*localEvalCells[2]];
            }
            else if (countIndices[0] != 2*dim && countIndices[1] == 2*dim
                && countIndices[2] == 2*dim)
            {
              output[0] = overlap[countIndices[0]][evalIndices[1]
                + evalIndices[2]*localEvalCells[1]];
            }
            else
            {
              for (unsigned int i = 0; i < dim; i++)
              {
                if (evalIndices[i] > localEvalCells[i])
                  evalIndices[i]++;
                else if (evalIndices[i] == localEvalCells[i])
                  evalIndices[i]--;
              }

              const Index& index = Traits::indicesToIndex(evalIndices,localEvalCells);
              output[0] = evalVector[index];
            }
          }
          else if (dim == 2)
          {
            if (countIndices[0] == 2*dim && countIndices[1] != 2*dim)
            {
              output[0] = overlap[countIndices[1]][evalIndices[0]];
            }
            else if (countIndices[0] != 2*dim && countIndices[1] == 2*dim)
            {
              output[0] = overlap[countIndices[0]][evalIndices[1]];
            }
            else
            {
              for (unsigned int i = 0; i < dim; i++)
              {
                if (evalIndices[i] > localEvalCells[i])
                  evalIndices[i]++;
                else if (evalIndices[i] == localEvalCells[i])
                  evalIndices[i]--;
              }

              const Index& index = Traits::indicesToIndex(evalIndices,localEvalCells);
              output[0] = evalVector[index];
            }
          }
          else if (dim == 1)
          {
            if (countIndices[0] != 2*dim)
            {
              output[0] = overlap[countIndices[0]][0];
            }
            else
            {
              for (unsigned int i = 0; i < dim; i++)
              {
                if (evalIndices[i] > localEvalCells[i])
                  evalIndices[i]++;
                else if (evalIndices[i] == localEvalCells[i])
                  evalIndices[i]--;
              }

              const Index& index = Traits::indicesToIndex(evalIndices,localEvalCells);
              output[0] = evalVector[index];
            }
          }
          else
            DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");
        }

        /**
         * @brief Set stochastic part to zero
         */
        void zero()
        {
          for (Index i = 0; i < dataVector.size(); i++)
            dataVector[i] = 0.;

          evalValid = false;
        }

        /**
         * @brief Double spatial resolution and transfer field values
         */
        void refine()
        {
          if (level != (*traits).level)
          {
            const std::vector<RF> oldData = dataVector;
            update();

            Indices oldLocalCells;
            for (unsigned int i = 0; i < dim; i++)
            {
              oldLocalCells[i] = localCells[i]/2;
            }

            if (!dataVector.empty())
            {
              dataVector.resize(localDomainSize);

              Indices oldIndices;
              Indices newIndices;
              if (dim == 3)
              {
                for (oldIndices[2] = 0; oldIndices[2] < oldLocalCells[2]; oldIndices[2]++)
                  for (oldIndices[1] = 0; oldIndices[1] < oldLocalCells[1]; oldIndices[1]++)
                    for (oldIndices[0] = 0; oldIndices[0] < oldLocalCells[0]; oldIndices[0]++)
                    {
                      newIndices[0] = 2*oldIndices[0];
                      newIndices[1] = 2*oldIndices[1];
                      newIndices[2] = 2*oldIndices[2];

                      const Index oldIndex = Traits::indicesToIndex(oldIndices,oldLocalCells);
                      const Index newIndex = Traits::indicesToIndex(newIndices,localCells);
                      const RF oldValue = oldData[oldIndex];

                      dataVector[newIndex                                                  ] = oldValue;
                      dataVector[newIndex + 1                                              ] = oldValue;
                      dataVector[newIndex + localCells[0]                                  ] = oldValue;
                      dataVector[newIndex + localCells[0] + 1                              ] = oldValue;
                      dataVector[newIndex + localCells[1]*localCells[0]                    ] = oldValue;
                      dataVector[newIndex + localCells[1]*localCells[0] + 1                ] = oldValue;
                      dataVector[newIndex + localCells[1]*localCells[0] + localCells[0]    ] = oldValue;
                      dataVector[newIndex + localCells[1]*localCells[0] + localCells[0] + 1] = oldValue;
                    }
              }
              else if (dim == 2)
              {
                for (oldIndices[1] = 0; oldIndices[1] < oldLocalCells[1]; oldIndices[1]++)
                  for (oldIndices[0] = 0; oldIndices[0] < oldLocalCells[0]; oldIndices[0]++)
                  {
                    newIndices[0] = 2*oldIndices[0];
                    newIndices[1] = 2*oldIndices[1];

                    const Index oldIndex = Traits::indicesToIndex(oldIndices,oldLocalCells);
                    const Index newIndex = Traits::indicesToIndex(newIndices,localCells);
                    const RF oldValue = oldData[oldIndex];

                    dataVector[newIndex                    ] = oldValue;
                    dataVector[newIndex + 1                ] = oldValue;
                    dataVector[newIndex + localCells[0]    ] = oldValue;
                    dataVector[newIndex + localCells[0] + 1] = oldValue;
                  }
              }
              else if (dim == 1)
              {
                DUNE_THROW(Dune::Exception,"not implemented");
              }
              else
                DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");
            }

            evalValid = false;
          }
        }

        /**
         * @brief Reduce spatial resolution and transfer field values
         */
        void coarsen()
        {
          if (level != (*traits).level)
          {
            const std::vector<RF> oldData = dataVector;
            update();

            Indices oldLocalCells;
            for (unsigned int i = 0; i < dim; i++)
            {
              oldLocalCells[i] = localCells[i]*2;
            }

            if (!dataVector.empty())
            {
              dataVector.resize(localDomainSize);

              Indices oldIndices;
              Indices newIndices;
              if (dim == 3)
              {
                for (newIndices[2] = 0; newIndices[2] < localCells[2]; newIndices[2]++)
                  for (newIndices[1] = 0; newIndices[1] < localCells[1]; newIndices[1]++)
                    for (newIndices[0] = 0; newIndices[0] < localCells[0]; newIndices[0]++)
                    {
                      oldIndices[0] = 2*newIndices[0];
                      oldIndices[1] = 2*newIndices[1];
                      oldIndices[2] = 2*newIndices[2];

                      const Index oldIndex = Traits::indicesToIndex(oldIndices,oldLocalCells);
                      const Index newIndex = Traits::indicesToIndex(newIndices,localCells);

                      RF newValue = 0.;
                      newValue += oldData[oldIndex                                                           ];
                      newValue += oldData[oldIndex + 1                                                       ];
                      newValue += oldData[oldIndex + oldLocalCells[0]                                        ];
                      newValue += oldData[oldIndex + oldLocalCells[0] + 1                                    ];
                      newValue += oldData[oldIndex + oldLocalCells[1]*oldLocalCells[0]                       ];
                      newValue += oldData[oldIndex + oldLocalCells[1]*oldLocalCells[0] + 1                   ];
                      newValue += oldData[oldIndex + oldLocalCells[1]*oldLocalCells[0] + oldLocalCells[0]    ];
                      newValue += oldData[oldIndex + oldLocalCells[1]*oldLocalCells[0] + oldLocalCells[0] + 1];
                      dataVector[newIndex] = newValue / 8;
                    }
              }
              else if (dim == 2)
              {
                for (newIndices[1] = 0; newIndices[1] < localCells[1]; newIndices[1]++)
                  for (newIndices[0] = 0; newIndices[0] < localCells[0]; newIndices[0]++)
                  {
                    oldIndices[0] = 2*newIndices[0];
                    oldIndices[1] = 2*newIndices[1];

                    const Index oldIndex = Traits::indicesToIndex(oldIndices,oldLocalCells);
                    const Index newIndex = Traits::indicesToIndex(newIndices,localCells);

                    RF newValue = 0.;
                    newValue += oldData[oldIndex                       ];
                    newValue += oldData[oldIndex + 1                   ];
                    newValue += oldData[oldIndex + oldLocalCells[0]    ];
                    newValue += oldData[oldIndex + oldLocalCells[0] + 1];
                    dataVector[newIndex] = newValue / 4;
                  }
              }
              else if (dim == 1)
              {
                DUNE_THROW(Dune::Exception,"not implemented");
              }
              else
                DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");
            }

            evalValid = false;
          }
        }

        /**
         * @brief One norm
         */
        RF oneNorm() const
        {
          RF sum = 0., mySum = 0.;

          for (Index i = 0; i < dataVector.size(); ++i)
            mySum += std::abs(dataVector[i]);

          MPI_Allreduce(&mySum,&sum,1,mpiType<RF>,MPI_SUM,(*traits).comm);
          return sum;
        }

        /**
         * @brief Infinity norm
         */
        RF infNorm() const
        {
          RF max = 0., myMax = 0.;

          for (Index i = 0; i < dataVector.size(); ++i)
            myMax = std::max(myMax, std::abs(dataVector[i]));

          MPI_Allreduce(&myMax,&max,1,mpiType<RF>,MPI_MAX,(*traits).comm);
          return max;
        }

        /**
         * @brief Multiply field with Gaussian with given center and radius
         */
        void localize(const typename Traits::DomainType& center, const RF radius)
        {
          typename Traits::DomainType location;
          const RF factor = std::pow(2.*3.14159,-(dim/2.));
          RF distSquared;

          for (Index i = 0; i < dataVector.size(); i++)
          {
            Traits::indexToIndices(i,cellIndices,localCells);
            Traits::indicesToCoords(cellIndices,localOffset,location);

            distSquared = 0.;
            for (unsigned int j = 0; j < dim; j++)
              distSquared += (location[j] - center[j]) * (location[j] - center[j]);

            dataVector[i] *= factor * std::exp(-0.5*distSquared/(radius*radius));
          }

          evalValid = false;
        }

        private:

        /**
         * @brief Convert data in striped (FFT compatible) format to setup using blocks
         */
        void dataToEval() const
        {
          if (dataVector.empty())
          {
            dataVector.resize(localDomainSize);
            for (Index i = 0; i < dataVector.size(); i++)
              dataVector[i] = 0.;
          }

          evalVector.resize(localDomainSize);
          overlap.resize((*traits).dim*2);
          for (unsigned int i = 0; i < dim; i++)
          {
            overlap[2*i    ].resize(localDomainSize/localEvalCells[i]);
            overlap[2*i + 1].resize(localDomainSize/localEvalCells[i]);
          }

          std::vector<RF> resorted(dataVector.size(),0.);
          std::vector<RF> temp = dataVector;

          if (commSize == 1)
          {
            evalVector = dataVector;
            evalValid  = true;
            return;
          }

          if (dim == 1)
          {
            evalVector = dataVector;
            exchangeOverlap();
            evalValid  = true;
            return;
          }

          Index numSlices = procPerDim[0]*localDomainSize/localCells[0];
          Index sliceSize = localDomainSize/numSlices;

          if (dim == 3)
          {
            Index px = procPerDim[0];
            Index py = procPerDim[1];
            Index ny = localCells[dim-2];
            Index nz = localCells[dim-1];
            Index dy = ny/py;

            for (Index i = 0; i < numSlices; i++)
            {
              Index term1 = (i%px) * (dy*nz);
              Index term2 = ((i/(dy*px)*dy)%ny) * (nz*px);
              Index term3 = (i/(ny*px)) * dy;
              Index term4 = (i/px) % dy;

              Index iNew = term1 + term2 + term3 + term4;

              for (Index j = 0; j < sliceSize; j++)
                resorted[iNew * sliceSize + j] = dataVector[i * sliceSize + j];
            }
          }
          else if (dim == 2)
          {
            for (Index i = 0; i < numSlices; i++)
            {
              const Index iNew = i/procPerDim[0] + (i%procPerDim[0])*localCells[dim-1];

              for (Index j = 0; j < sliceSize; j++)
                resorted[iNew * sliceSize + j] = dataVector[i * sliceSize + j];
            }
          }
          else
            DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

          unsigned int numComms;
          if (dim == 3)
            numComms = procPerDim[0]*procPerDim[1];
          else if (dim == 2)
            numComms = procPerDim[0];
          else
            DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");
          std::vector<MPI_Request> request(numComms);

          for (unsigned int i = 0; i < numComms; i++)
            MPI_Isend(&(resorted  [i*localDomainSize/numComms]), localDomainSize/numComms,
                mpiType<RF>, (rank/numComms)*numComms + i, toEval, (*traits).comm, &request[i]);

          for (unsigned int i = 0; i < numComms; i++)
            MPI_Recv (&(evalVector[i*localDomainSize/numComms]), localDomainSize/numComms,
                mpiType<RF>, (rank/numComms)*numComms + i, toEval, (*traits).comm, MPI_STATUS_IGNORE);

          MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);

          exchangeOverlap();

          evalValid = true;
        }

        /**
         * @brief Convert data in blocks to setup using stripes (FFT compatible)
         */
        void evalToData()
        {
          dataVector.resize(localDomainSize);

          if (commSize == 1 || dim == 1)
          {
            dataVector = evalVector;
            return;
          }

          std::vector<RF> resorted(dataVector.size(),0.);

          unsigned int numComms;
          if (dim == 3)
            numComms = procPerDim[0]*procPerDim[1];
          else if (dim == 2)
            numComms = procPerDim[0];
          else
            DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");
          std::vector<MPI_Request> request(numComms);

          for (unsigned int i = 0; i < numComms; i++)
            MPI_Isend(&(evalVector[i*localDomainSize/numComms]), localDomainSize/numComms,
                mpiType<RF>, (rank/numComms)*numComms + i, fromEval, (*traits).comm, &request[i]);

          for (unsigned int i = 0; i < numComms; i++)
            MPI_Recv (&(resorted  [i*localDomainSize/numComms]), localDomainSize/numComms,
                mpiType<RF>, (rank/numComms)*numComms + i, fromEval, (*traits).comm, MPI_STATUS_IGNORE);

          Index numSlices = procPerDim[0]*localDomainSize/localCells[0];
          Index sliceSize = localDomainSize/numSlices;

          if (dim == 3)
          {
            for (Index i = 0; i < numSlices; i++)
            {
              Index px = procPerDim[0];
              Index py = procPerDim[1];
              Index ny = localCells[dim-2];
              Index nz = localCells[dim-1];
              Index dy = ny/py;

              Index term1 = (i%px) * (dy*nz);
              Index term2 = ((i/(dy*px)*dy)%ny) * (nz*px);
              Index term3 = (i/(ny*px)) * dy;
              Index term4 = (i/px) % dy;

              Index iNew = term1 + term2 + term3 + term4;

              for (Index j = 0; j < sliceSize; j++)
                dataVector[i * sliceSize + j] = resorted[iNew * sliceSize + j];
            }
          }
          else if (dim == 2)
          {
            for (Index i = 0; i < numSlices; i++)
            {
              Index iNew = i/procPerDim[0] + (i%procPerDim[0])*localCells[dim-1];

              for (Index j = 0; j < sliceSize; j++)
                dataVector[i * sliceSize + j] = resorted[iNew * sliceSize + j];
            }
          }
          else
            DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

          MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);
        }

        /**
         * @brief Communicate the overlap regions at the block boundaries
         */
        void exchangeOverlap() const
        {
          std::array<unsigned int,2*dim> neighbor;
          std::vector<std::vector<RF>> extract = overlap;

          if (dim == 3)
          {
            for (unsigned int i = 0; i < dim; i++)
            {
              const Index iNext     = (i+1)%dim;
              const Index iNextNext = (i+2)%dim;
              for (evalIndices[iNext] = 0; evalIndices[iNext] < localEvalCells[iNext]; evalIndices[iNext]++)
              {
                for (evalIndices[iNextNext] = 0;
                    evalIndices[iNextNext] < localEvalCells[iNextNext]; evalIndices[iNextNext]++)
                {
                  evalIndices[i] = 0;
                  const unsigned int index  = Traits::indicesToIndex(evalIndices,localEvalCells);
                  extract[2*i  ][evalIndices[iNext] + evalIndices[iNextNext] * localEvalCells[iNext]]
                    = evalVector[index];

                  evalIndices[i] = localEvalCells[i] - 1;
                  const unsigned int index2 = Traits::indicesToIndex(evalIndices,localEvalCells);
                  extract[2*i+1][evalIndices[iNext] + evalIndices[iNextNext] * localEvalCells[iNext]]
                    = evalVector[index2];
                }
              }
            }

            neighbor[0] = (rank/procPerDim[0])*procPerDim[0] + (rank+(procPerDim[0]-1))%procPerDim[0];
            neighbor[1] = (rank/procPerDim[0])*procPerDim[0] + (rank+1                )%procPerDim[0];
            neighbor[2] = (rank/(procPerDim[0]*procPerDim[1]))*(procPerDim[0]*procPerDim[1])
              + (rank+(procPerDim[0]*procPerDim[1]-procPerDim[0]))%(procPerDim[0]*procPerDim[1]);
            neighbor[3] = (rank/(procPerDim[0]*procPerDim[1]))*(procPerDim[0]*procPerDim[1])
              + (rank+procPerDim[0]                              )%(procPerDim[0]*procPerDim[1]);
            neighbor[4] = (rank+(commSize-(procPerDim[0]*procPerDim[1])))%commSize;
            neighbor[5] = (rank+(procPerDim[0]*procPerDim[1])           )%commSize;

          }
          else if (dim == 2)
          {
            for (unsigned int i = 0; i < dim; i++)
            {
              const Index iNext = (i+1)%dim;
              for (evalIndices[iNext] = 0; evalIndices[iNext] < localEvalCells[iNext]; evalIndices[iNext]++)
              {
                evalIndices[i] = 0;
                const Index index  = Traits::indicesToIndex(evalIndices,localEvalCells);
                extract[2*i  ][evalIndices[iNext]] = evalVector[index];

                evalIndices[i] = localEvalCells[i] - 1;
                const Index index2 = Traits::indicesToIndex(evalIndices,localEvalCells);
                extract[2*i+1][evalIndices[iNext]] = evalVector[index2];
              }
            }

            neighbor[0] = (rank/procPerDim[0])*procPerDim[0] + (rank+(procPerDim[0]-1))%procPerDim[0];
            neighbor[1] = (rank/procPerDim[0])*procPerDim[0] + (rank+1                )%procPerDim[0];
            neighbor[2] = (rank+(commSize-procPerDim[0]))%commSize;
            neighbor[3] = (rank+procPerDim[0]           )%commSize;

          }
          else if (dim == 1)
          {
            neighbor[0] = (rank+(commSize-1))%commSize;
            neighbor[1] = (rank+1           )%commSize;
          }
          else
            DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

          std::vector<MPI_Request> request(2*dim);

          for (unsigned int i = 0; i < dim; i++)
          {
            MPI_Isend(&(extract[2*i  ][0]), localDomainSize/localEvalCells[i], mpiType<RF>,
                neighbor[2*i  ], toOverlap, (*traits).comm, &request[2*i]);
            MPI_Recv (&(overlap[2*i+1][0]), localDomainSize/localEvalCells[i], mpiType<RF>,
                neighbor[2*i+1], toOverlap, (*traits).comm, MPI_STATUS_IGNORE);

            MPI_Isend(&(extract[2*i+1][0]), localDomainSize/localEvalCells[i], mpiType<RF>,
                neighbor[2*i+1], toOverlap, (*traits).comm, &request[2*i+1]);
            MPI_Recv (&(overlap[2*i  ][0]), localDomainSize/localEvalCells[i], mpiType<RF>,
                neighbor[2*i  ], toOverlap, (*traits).comm, MPI_STATUS_IGNORE);
          }

          MPI_Waitall(request.size(),&(request[0]),MPI_STATUSES_IGNORE);
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_STOCHASTIC_HH
