// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_STOCHASTIC_HH
#define	DUNE_RANDOMFIELD_STOCHASTIC_HH

#include<fstream>

#include<dune/common/power.hh>
#if HAVE_DUNE_PDELAB
#include<dune/pdelab/gridfunctionspace/localfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/lfsindexcache.hh>
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
          using RF = typename Traits::RF;

          enum {dim = Traits::dim};

          friend class RandomFieldMatrix<Traits>;

          std::shared_ptr<Traits> traits;

          int rank, commSize;
          std::array<RF,dim>           extensions;
          std::array<unsigned int,dim> cells;
          unsigned int                 level;
          std::array<unsigned int,dim> localCells;
          std::array<unsigned int,dim> localOffset;
          unsigned int                 localDomainSize;
          unsigned int                 sliceSize;
          std::array<unsigned int,dim> localEvalCells;
          std::array<unsigned int,dim> localEvalOffset;
          std::array<int,dim>          procPerDim;

          std::vector<RF> dataVector;
          mutable std::vector<RF> evalVector;
          mutable std::vector<std::vector<RF> > overlap;

          mutable bool evalValid;
          mutable std::array<unsigned int,dim> cellIndices;
          mutable std::array<unsigned int,dim> evalIndices;
          mutable std::array<unsigned int,dim> countIndices;

        public:

          /**
           * @brief Constructor
           */
          StochasticPart(const std::shared_ptr<Traits>& traits_, const std::string& fileName)
            : traits(traits_)
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
              DUNE_THROW(Dune::NotImplemented,"Writing and reading field files requires parallel HDF5 support");
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
          template<typename GFS, typename Field>
            StochasticPart(const StochasticPart& other, const GFS& gfs, const Field& field)
            : traits(other.traits)
            {
              update();

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

                const typename Traits::DomainType& coords = elem.geometry().center();
                (*traits).coordsToIndices(coords,evalIndices,localEvalOffset);
                const unsigned int index = (*traits).indicesToIndex(evalIndices,localEvalCells);

                evalVector[index] = vLocal[0];
              }

              evalToData();
              evalValid  = true;
            }
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
            else
              DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

            dataVector.resize(localDomainSize);
            evalVector.resize(localDomainSize);
            overlap.resize((*traits).dim*2);
            for (unsigned int i = 0; i < dim; i++)
            {
              overlap[2*i    ].resize(localDomainSize/localEvalCells[i]);
              overlap[2*i + 1].resize(localDomainSize/localEvalCells[i]);
            }

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
            {
              std::ofstream file(fileName+".xdmf",std::ofstream::trunc);

              file
                << "<?xml version=\"1.0\" ?>\n"
                << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
                << "<Xdmf Version=\"2.0\">\n"
                << " <Domain>\n"
                << "  <Grid Name=\"StructuredGrid\" GridType=\"Uniform\">\n"
                << "   <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"";
              for (unsigned int i = 0; i < dim; i++)
                file << cells[dim-(i+1)] << " ";
              file
                << "\"/>\n"
                << "   <Geometry GeometryType=\"origin_dxdydz\">\n"
                << "    <DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n"
                << "     0. 0. 0.\n"
                << "    </DataItem>\n"
                << "    <DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n"
                << "     ";
              for (unsigned int i = 0; i < 3 - dim; i++)
                file << extensions[0]/cells[0] << " "; // additional entries to visualize 1D and 2D files
              for (unsigned int i = 0; i < dim; i++)
                file << extensions[i]/cells[i] << " ";
              file
                << "\n"
                << "    </DataItem>\n"
                << "   </Geometry>\n"
                << "   <Attribute Name=\""
                << fileName
                << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                << "    <DataItem Dimensions=\"";
              for (unsigned int i = 0; i < dim; i++)
                file << cells[dim-(i+1)] << " ";
              file
                << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n"
                << "     " << fileName+".stoch.h5" << ":/stochastic\n"
                << "    </DataItem>\n"
                << "   </Attribute>\n"
                << "  </Grid>\n"
                << " </Domain>\n"
                << "</Xdmf>"
                << std::endl;
            }
#else //HAVE_HDF5
            DUNE_THROW(Dune::NotImplemented,"Writing and reading field files requires parallel HDF5 support");
#endif //HAVE_HDF5
          }

          /**
           * @brief Addition assignment operator
           */
          StochasticPart& operator+=(const StochasticPart& other)
          {
            for (unsigned int i = 0; i < localDomainSize; ++i)
            {
              dataVector[i] += other.dataVector[i];
            }

            evalValid = false;

            return *this;
          }

          /**
           * @brief Subtraction assignment operator
           */
          StochasticPart& operator-=(const StochasticPart& other)
          {
            for (unsigned int i = 0; i < localDomainSize; ++i)
            {
              dataVector[i] -= other.dataVector[i];
            }

            evalValid = false;

            return *this;
          }

          /**
           * @brief Multiplication with scalar
           */
          StochasticPart& operator*=(const RF alpha)
          {
            for (unsigned int i = 0; i < localDomainSize; ++i)
            {
              dataVector[i] *= alpha;
            }

            evalValid = false;

            return *this;
          }

          /**
           * @brief AXPY scaled addition
           */
          StochasticPart& axpy(const StochasticPart& other, const RF alpha)
          {
            for (unsigned int i = 0; i < localDomainSize; ++i)
            {
              dataVector[i] += other.dataVector[i] * alpha;
            }

            evalValid = false;

            return *this;
          }

          /**
           * @brief Scalar product
           */
          RF operator*(const StochasticPart& other) const
          {
            RF sum = 0., mySum = 0.;

            for (unsigned int i = 0; i < localDomainSize; ++i)
            {
              mySum += dataVector[i] * other.dataVector[i];
            }

            MPI_Allreduce(&mySum,&sum,1,MPI_DOUBLE,MPI_SUM,(*traits).comm);
            return sum;
          }

          /**
           * @brief Evaluate stochastic part at given location
           */
          void evaluate(const typename Traits::DomainType& location, typename Traits::RangeType& output) const
          {
            if (!evalValid)
              dataToEval();

            (*traits).coordsToIndices(location,evalIndices,localEvalOffset);

            for (unsigned int i = 0; i < dim; i++)
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
              if (countIndices[0] == 2*dim && countIndices[1] == 2*dim && countIndices[2] != 2*dim)
              {
                output[0] = overlap[countIndices[2]][evalIndices[0] + evalIndices[1]*localEvalCells[0]];
              }
              else if (countIndices[0] == 2*dim && countIndices[1] != 2*dim && countIndices[2] == 2*dim)
              {
                output[0] = overlap[countIndices[1]][evalIndices[2] + evalIndices[0]*localEvalCells[2]];
              }
              else if (countIndices[0] != 2*dim && countIndices[1] == 2*dim && countIndices[2] == 2*dim)
              {
                output[0] = overlap[countIndices[0]][evalIndices[1] + evalIndices[2]*localEvalCells[1]];
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

                const unsigned int index = (*traits).indicesToIndex(evalIndices,localEvalCells);
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

                const unsigned int index = (*traits).indicesToIndex(evalIndices,localEvalCells);
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

                const unsigned int index = (*traits).indicesToIndex(evalIndices,localEvalCells);
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
            for (unsigned int i = 0; i < localDomainSize; i++)
            {
              dataVector[i] = 0.;
            }

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

              std::array<unsigned int,dim> oldLocalCells;
              for (unsigned int i = 0; i < dim; i++)
              {
                oldLocalCells[i] = localCells[i]/2;
              }

              dataVector.resize(localDomainSize);

              std::array<unsigned int,dim> oldIndices;
              std::array<unsigned int,dim> newIndices;
              if (dim == 3)
              {
                for (oldIndices[2] = 0; oldIndices[2] < oldLocalCells[2]; oldIndices[2]++)
                  for (oldIndices[1] = 0; oldIndices[1] < oldLocalCells[1]; oldIndices[1]++)
                    for (oldIndices[0] = 0; oldIndices[0] < oldLocalCells[0]; oldIndices[0]++)
                    {
                      newIndices[0] = 2*oldIndices[0];
                      newIndices[1] = 2*oldIndices[1];
                      newIndices[2] = 2*oldIndices[2];

                      const unsigned int oldIndex = (*traits).indicesToIndex(oldIndices,oldLocalCells);
                      const unsigned int newIndex = (*traits).indicesToIndex(newIndices,localCells);
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

                    const unsigned int oldIndex = (*traits).indicesToIndex(oldIndices,oldLocalCells);
                    const unsigned int newIndex = (*traits).indicesToIndex(newIndices,localCells);
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

              std::array<unsigned int,dim> oldLocalCells;
              for (unsigned int i = 0; i < dim; i++)
              {
                oldLocalCells[i] = localCells[i]*2;
              }

              dataVector.resize(localDomainSize);

              std::array<unsigned int,dim> oldIndices;
              std::array<unsigned int,dim> newIndices;
              if (dim == 3)
              {
                for (newIndices[2] = 0; newIndices[2] < localCells[2]; newIndices[2]++)
                  for (newIndices[1] = 0; newIndices[1] < localCells[1]; newIndices[1]++)
                    for (newIndices[0] = 0; newIndices[0] < localCells[0]; newIndices[0]++)
                    {
                      oldIndices[0] = 2*newIndices[0];
                      oldIndices[1] = 2*newIndices[1];
                      oldIndices[2] = 2*newIndices[2];

                      const unsigned int oldIndex = (*traits).indicesToIndex(oldIndices,oldLocalCells);
                      const unsigned int newIndex = (*traits).indicesToIndex(newIndices,localCells);

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

                    const unsigned int oldIndex = (*traits).indicesToIndex(oldIndices,oldLocalCells);
                    const unsigned int newIndex = (*traits).indicesToIndex(newIndices,localCells);

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

              evalValid = false;
            }
          }

          void localize(const typename Traits::DomainType& center, const RF radius)
          {
            typename Traits::DomainType location;
            const RF factor = std::pow(2.*3.14159,-(dim/2.));
            RF distSquared;

            for (unsigned int i = 0; i < localDomainSize; i++)
            {
              (*traits).indexToIndices(i,cellIndices,localCells);
              (*traits).indicesToCoords(cellIndices,localOffset,location);

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

            MPI_Request request;
            MPI_Status status;

            unsigned int numSlices = procPerDim[0]*localDomainSize/localCells[0];
            unsigned int sliceSize = localDomainSize/numSlices;

            if (dim == 3)
            {
              unsigned int px = procPerDim[0];
              unsigned int py = procPerDim[1];
              unsigned int ny = localCells[dim-2];
              unsigned int nz = localCells[dim-1];
              unsigned int dy = ny/py;

              for (unsigned int i = 0; i < numSlices; i++)
              {
                unsigned int term1 = (i%px) * (dy*nz);
                unsigned int term2 = ((i/(dy*px)*dy)%ny) * (nz*px);
                unsigned int term3 = (i/(ny*px)) * dy;
                unsigned int term4 = (i/px) % dy;

                unsigned int iNew = term1 + term2 + term3 + term4;

                for (unsigned int j = 0; j < sliceSize; j++)
                {
                  resorted[iNew * sliceSize + j] = dataVector[i * sliceSize + j];
                }
              }
            }
            else if (dim == 2)
            {
              for (unsigned int i = 0; i < numSlices; i++)
              {
                unsigned int iNew = i/procPerDim[0] + (i%procPerDim[0])*localCells[dim-1];

                for (unsigned int j = 0; j < sliceSize; j++)
                {
                  resorted[iNew * sliceSize + j] = dataVector[i * sliceSize + j];
                }
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

            for (unsigned int i = 0; i < numComms; i++)
              MPI_Isend(&(resorted  [i*localDomainSize/numComms]), localDomainSize/numComms,
                  MPI_DOUBLE, (rank/numComms)*numComms + i, 0, (*traits).comm, &request);

            for (unsigned int i = 0; i < numComms; i++)
              MPI_Recv (&(evalVector[i*localDomainSize/numComms]), localDomainSize/numComms,
                  MPI_DOUBLE, (rank/numComms)*numComms + i, 0, (*traits).comm, &status);

            MPI_Barrier((*traits).comm);

            exchangeOverlap();

            evalValid = true;
          }

          /**
           * @brief Convert data in blocks to setup using stripes (FFT compatible)
           */
          void evalToData()
          {
            if (commSize == 1 || dim == 1)
            {
              dataVector = evalVector;
              return;
            }

            std::vector<RF> resorted(dataVector.size(),0.);

            MPI_Request request;
            MPI_Status status;

            unsigned int numComms;
            if (dim == 3)
              numComms = procPerDim[0]*procPerDim[1];
            else if (dim == 2)
              numComms = procPerDim[0];
            else
              DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

            for (unsigned int i = 0; i < numComms; i++)
              MPI_Isend(&(evalVector[i*localDomainSize/numComms]), localDomainSize/numComms,
                  MPI_DOUBLE, (rank/numComms)*numComms + i, 0, (*traits).comm, &request);

            for (unsigned int i = 0; i < numComms; i++)
              MPI_Recv (&(resorted  [i*localDomainSize/numComms]), localDomainSize/numComms,
                  MPI_DOUBLE, (rank/numComms)*numComms + i, 0, (*traits).comm, &status);

            unsigned int numSlices = procPerDim[0]*localDomainSize/localCells[0];
            unsigned int sliceSize = localDomainSize/numSlices;

            if (dim == 3)
            {
              for (unsigned int i = 0; i < numSlices; i++)
              {
                unsigned int px = procPerDim[0];
                unsigned int py = procPerDim[1];
                unsigned int ny = localCells[dim-2];
                unsigned int nz = localCells[dim-1];
                unsigned int dy = ny/py;

                unsigned int term1 = (i%px) * (dy*nz);
                unsigned int term2 = ((i/(dy*px)*dy)%ny) * (nz*px);
                unsigned int term3 = (i/(ny*px)) * dy;
                unsigned int term4 = (i/px) % dy;

                unsigned int iNew = term1 + term2 + term3 + term4;

                for (unsigned int j = 0; j < sliceSize; j++)
                {
                  dataVector[i * sliceSize + j] = resorted[iNew * sliceSize + j];
                }
              }
            }
            else if (dim == 2)
            {
              for (unsigned int i = 0; i < numSlices; i++)
              {
                unsigned int iNew = i/procPerDim[0] + (i%procPerDim[0])*localCells[dim-1];
                for (unsigned int j = 0; j < sliceSize; j++)
                {
                  dataVector[i * sliceSize + j] = resorted[iNew * sliceSize + j];
                }
              }
            }
            else
              DUNE_THROW(Dune::Exception,"dimension of field has to be 1, 2 or 3");

            MPI_Barrier((*traits).comm);
          }

          /**
           * @brief Communicate the overlap regions at the block boundaries
           */
          void exchangeOverlap() const
          {
            std::array<unsigned int,2*dim> neighbor;
            std::vector<std::vector<RF> > extract = overlap;

            if (dim == 3)
            {
              for (unsigned int i = 0; i < dim; i++)
              {
                const unsigned int iNext     = (i+1)%dim;
                const unsigned int iNextNext = (i+2)%dim;
                for (evalIndices[iNext] = 0; evalIndices[iNext] < localEvalCells[iNext]; evalIndices[iNext]++)
                {
                  for (evalIndices[iNextNext] = 0;
                      evalIndices[iNextNext] < localEvalCells[iNextNext]; evalIndices[iNextNext]++)
                  {
                    evalIndices[i] = 0;
                    const unsigned int index  = (*traits).indicesToIndex(evalIndices,localEvalCells);
                    extract[2*i  ][evalIndices[iNext] + evalIndices[iNextNext] * localEvalCells[iNext]]
                      = evalVector[index];

                    evalIndices[i] = localEvalCells[i] - 1;
                    const unsigned int index2 = (*traits).indicesToIndex(evalIndices,localEvalCells);
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
                const unsigned int iNext = (i+1)%dim;
                for (evalIndices[iNext] = 0; evalIndices[iNext] < localEvalCells[iNext]; evalIndices[iNext]++)
                {
                  evalIndices[i] = 0;
                  const unsigned int index  = (*traits).indicesToIndex(evalIndices,localEvalCells);
                  extract[2*i  ][evalIndices[iNext]] = evalVector[index];

                  evalIndices[i] = localEvalCells[i] - 1;
                  const unsigned int index2 = (*traits).indicesToIndex(evalIndices,localEvalCells);
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

            MPI_Request request;
            MPI_Status status;

            for (unsigned int i = 0; i < dim; i++)
            {
              MPI_Isend(&(extract[2*i  ][0]), localDomainSize/localEvalCells[i], MPI_DOUBLE,
                  neighbor[2*i  ], 0, (*traits).comm, &request);
              MPI_Recv (&(overlap[2*i+1][0]), localDomainSize/localEvalCells[i], MPI_DOUBLE,
                  neighbor[2*i+1], 0, (*traits).comm, &status);

              MPI_Isend(&(extract[2*i+1][0]), localDomainSize/localEvalCells[i], MPI_DOUBLE,
                  neighbor[2*i+1], 0, (*traits).comm, &request);
              MPI_Recv (&(overlap[2*i  ][0]), localDomainSize/localEvalCells[i], MPI_DOUBLE,
                  neighbor[2*i  ], 0, (*traits).comm, &status);
            }

            MPI_Barrier((*traits).comm);
          }

      };

  }
}

#endif // DUNE_RANDOMFIELD_STOCHASTIC_HH
