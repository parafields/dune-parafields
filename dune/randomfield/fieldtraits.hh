// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_FIELDTRAITS_HH
#define	DUNE_RANDOMFIELD_FIELDTRAITS_HH

#include<vector>

#include<dune/common/parametertreeparser.hh>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Spherical covariance function
     */
    class SphericalCovariance
    {
      public:

        template<typename RF>
          RF operator()(RF variance, std::vector<RF> x, std::vector<RF> lambda) const
          {
            const unsigned int dim = x.size();
            if( variance == 0.0 )
              return 0.0;

            RF sum = 0.0;
            for(unsigned int i=0; i<dim; i++)
            {
              sum += (x[i] * x[i]) / (lambda[i] * lambda[i]);
            }
            RF h_eff = std::sqrt(sum);
            if (h_eff > 1.0)
              return 0.0;
            else
              return variance * (1.0 - 1.5 * h_eff + 0.5 * std::pow(h_eff, 3));
          }
    };

    /**
     * @brief Exponential covariance function
     */
    class ExponentialCovariance
    {
      public:

        template<typename RF>
          RF operator()(RF variance, std::vector<RF> x, std::vector<RF> lambda) const
          {
            const unsigned int dim = x.size();
            if( variance == 0.0 )
              return 0.0;

            RF sum = 0.0;
            for(unsigned int i=0; i<dim; i++)
            {
              sum += (x[i] * x[i]) / (lambda[i] * lambda[i]);
            }
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

        template<typename RF>
          RF operator()(RF variance, std::vector<RF> x, std::vector<RF> lambda) const
          {
            const unsigned int dim = x.size();
            if( variance == 0.0 )
              return 0.0;

            RF sum = 0.0;
            for(unsigned int i=0; i < dim; i++)
            {
              sum += (x[i] * x[i]) / (lambda[i] * lambda[i]);
            }
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

        template<typename RF>
          RF operator()(RF variance, std::vector<RF> x, std::vector<RF> lambda) const
          {
            const unsigned int dim = x.size();
            if( variance == 0.0 )
              return 0.0;

            RF sum = 0.0;
            for(unsigned int i=0; i < dim; i++)
            {
              sum += std::abs(x[i] / lambda[i]);
            }
            RF h_eff = sum;
            return variance * std::exp(-h_eff);
          }

    };

    template<typename Traits> class TrendPart;
    template<typename Traits> class TrendComponent;
    template<typename Traits> class StochasticPart;
    template<typename Traits> class RandomFieldMatrix;
    template<typename GridTraits, typename Covariance, bool storeInvMat, bool storeInvRoot> class RandomField;

    /**
     * @brief Traits for the RandomField class
     */
    template<typename GridTraits, typename Cov, bool storeInvMat, bool storeInvRoot>
      class RandomFieldTraits
      {
        typedef RandomFieldTraits<GridTraits,Cov,storeInvMat,storeInvRoot> ThisType;

        public:

        enum {dim = GridTraits::dim};
        enum {dimRange  = 1};
        enum {dimDomain = dim};

        typedef typename GridTraits::RangeField  RF;
        typedef typename GridTraits::DomainField DomainField;
        typedef typename GridTraits::Domain      DomainType;
        typedef typename GridTraits::Scalar      RangeType;

        typedef Cov Covariance;

        private:

        friend class TrendPart<ThisType>;
        friend class TrendComponent<ThisType>;
        friend class StochasticPart<ThisType>;
        friend class RandomFieldMatrix<ThisType>;
        friend class RandomField<GridTraits,Covariance,storeInvMat,storeInvRoot>;

        // MPI constants
        int rank, commSize;

        const Dune::ParameterTree& config;

        const std::vector<RF> extensions;
        unsigned int          level;
        std::vector<RF>       meshsize;
        RF                    cellVolume;

        RF              variance;
        std::vector<RF> corrLength;
        unsigned int    cgIterations;

        ptrdiff_t allocLocal, localN0, local0Start;

        // factor used in domain embedding
        int embeddingFactor;

        // properties of random field
        std::vector<unsigned int> cells;
        unsigned int              domainSize;
        std::vector<unsigned int> localCells;
        std::vector<unsigned int> localOffset;
        unsigned int              localDomainSize;

        // properties on extended domain
        std::vector<unsigned int> extendedCells;
        unsigned int              extendedDomainSize;
        std::vector<unsigned int> localExtendedCells;
        std::vector<unsigned int> localExtendedOffset;
        unsigned int              localExtendedDomainSize;

        mutable std::vector<unsigned int> globalIndices;
        mutable std::vector<unsigned int> localIndices;

        public:

        RandomFieldTraits(const Dune::ParameterTree& config_, const std::string& fieldName)
          : config(config_),
          extensions    (config.get<std::vector<RF> >          ("grid.extensions")),
          cgIterations  (config.get<unsigned int>              ("randomField.cgIterations",100)),
          cells         (config.get<std::vector<unsigned int> >("grid.cells"))
        {
          Dune::ParameterTree fieldProps;
          Dune::ParameterTreeParser parser;
          parser.readINITree(fieldName+".props",fieldProps);
          variance   = fieldProps.get<RF>              ("stochastic.variance");
          corrLength = fieldProps.get<std::vector<RF> >("stochastic.corrLength");

          MPI_Comm_rank(MPI_COMM_WORLD,&rank);
          MPI_Comm_size(MPI_COMM_WORLD,&commSize);

          if (corrLength.size() == 1)
          {
            if (rank == 0) std::cout << "homogeneous correlation length detected, extending" << std::endl;
            corrLength.resize(dim);
            for (int i = 1; i < dim; i ++)
              corrLength[i] = corrLength[0];
          }

          level = 0;

          fftw_mpi_init();
          update();
        }

        /**
         * @brief Compute constants after construction or refinement
         */
        void update()
        {
          /// @todo determine factor automatically
          embeddingFactor = 4;

          meshsize.resize(dim);
          extendedCells.resize(dim);
          localCells.resize(dim);
          localOffset.resize(dim);
          localExtendedCells.resize(dim);
          localExtendedOffset.resize(dim);
          globalIndices.resize(dim);
          localIndices.resize(dim);

          for (unsigned int i = 0; i < dim; i++)
          {
            meshsize[i]      = extensions[i] / cells[i];
            extendedCells[i] = embeddingFactor*cells[i];
          }

          getFFTData(allocLocal, localN0, local0Start);

          for (unsigned int i = 0; i < dim - 1; i++)
          {
            localExtendedCells [i] = extendedCells[i];
            localExtendedOffset[i] = 0;
            localCells [i] = cells[i];
            localOffset[i] = 0;
          }
          localExtendedCells [dim-1] = localN0;
          localExtendedOffset[dim-1] = local0Start;
          localCells [dim-1] = localN0/embeddingFactor;
          localOffset[dim-1] = local0Start/embeddingFactor;

          domainSize              = 1;
          extendedDomainSize      = 1;
          localDomainSize         = 1;
          localExtendedDomainSize = 1;
          cellVolume              = 1.;
          for (unsigned int i = 0; i < dim; i++)
          {
            domainSize              *= cells[i];
            extendedDomainSize      *= extendedCells[i];
            localDomainSize         *= localCells[i];
            localExtendedDomainSize *= localExtendedCells[i];
            cellVolume              *= meshsize[i];
          }

          if (rank == 0)
          {
            std::cout << "RandomField size:        " << localDomainSize << std::endl;
            std::cout << "RandomField cells:       ";
            for (unsigned int i = 0; i < dim; i++)
            {
              std::cout << cells[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "RandomField local cells: ";
            for (unsigned int i = 0; i < dim; i++)
            {
              std::cout << localCells[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "RandomField cell volume: " << cellVolume << std::endl;
          }
        }

        /**
         * @brief Request global refinement of the data structure
         */
        void refine()
        {
          for (unsigned int i = 0; i < dim; ++i)
            cells[i] *= 2;
          
          level++;
          
          update();
        }

        /**
         * @brief Get the domain decomposition data of the Fourier transform
         */
        template<typename T>
          void getFFTData(T& allocLocal, T& localN0, T& local0Start) const
          {
            if (dim == 3)
            {
              ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1],(ptrdiff_t)extendedCells[2]};
              allocLocal = fftw_mpi_local_size_3d(n[2] , n[1], n[0], MPI_COMM_WORLD, &localN0, &local0Start);
            }
            else
            {
              ptrdiff_t n[] = {(ptrdiff_t)extendedCells[0],(ptrdiff_t)extendedCells[1]};
              allocLocal = fftw_mpi_local_size_2d(n[1], n[0], MPI_COMM_WORLD, &localN0, &local0Start);
            }
          }

        /**
         * @brief Convert an index tuple into a one dimensional encoding
         */
        unsigned int indicesToIndex(const std::vector<unsigned int>& indices, const std::vector<unsigned int>& bound) const
        {
          if (dim == 3)
          {
            return indices[0] + bound[0] * (indices[1] + bound[1]*indices[2]);
          }
          else
          {
            return indices[1] * bound[0] + indices[0];
          }
        }

        /**
         * @brief Convert a one dimensional encoding into the original index tuple
         */
        void indexToIndices(const unsigned int index, std::vector<unsigned int>& indices, const std::vector<unsigned int>& bound) const
        {
          if (dim == 3)
          {
            indices[0] = index % bound[0];
            indices[1] = (index / bound[0]) % bound[1];
            indices[2] = (index / bound[0]) / bound[1];
          }
          else
          {
            indices[0] = index % bound[0];
            indices[1] = index / bound[0];
          }
        }

        /**
         * @brief Convert spatial coordinates into the corresponding integer indices
         */
        void coordsToIndices(const DomainType& location, std::vector<unsigned int>& localIndices, const std::vector<unsigned int>& offset) const
        {
          for (unsigned int i = 0; i < dim; i++)
          {
            globalIndices[i] = (unsigned int) (location[i] * cells[i] / extensions[i]);
            localIndices[i]  = globalIndices[i] - offset[i];
          }
        }

        /**
         * @brief Convert integer indices into corresponding spatial coordinates
         */
        void indicesToCoords(const std::vector<unsigned int>& localIndices, const std::vector<unsigned int>& offset, DomainType& location) const
        {
          for (unsigned int i = 0; i < dim; i++)
          {
            globalIndices[i] = localIndices[i] + offset[i];
            location[i]      = (globalIndices[i] * extensions[i] + 0.5) / cells[i];
          }
        }

      };

  }
}

#endif // DUNE_RANDOMFIELD_FIELDTRAITS_HH
