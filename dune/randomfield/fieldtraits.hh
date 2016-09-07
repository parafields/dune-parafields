// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_FIELDTRAITS_HH
#define	DUNE_RANDOMFIELD_FIELDTRAITS_HH

#include<array>
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

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x, const std::vector<RF>& lambda) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
            {
              sum += (x[i] * x[i]) / (lambda[i] * lambda[i]);
            }
            RF h_eff = std::sqrt(sum);
            if (h_eff > 1.)
              return 0.;
            else
              return variance * (1. - 1.5 * h_eff + 0.5 * std::pow(h_eff, 3));
          }
    };

    /**
     * @brief Exponential covariance function
     */
    class ExponentialCovariance
    {
      public:

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x, const std::vector<RF>& lambda) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
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

        template<typename RF, long unsigned int dim>
          RF operator()(const RF variance, const std::array<RF,dim>& x, const std::vector<RF>& lambda) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
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

        template<typename RF, unsigned int dim>
          RF operator()(RF variance, std::array<RF,dim>& x, std::array<RF,dim>& lambda) const
          {
            RF sum = 0.;
            for(unsigned int i = 0; i < dim; i++)
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

        typedef typename GridTraits::RangeField  RF;
        typedef typename GridTraits::DomainField DomainField;
        typedef typename GridTraits::Domain      DomainType;
        typedef typename GridTraits::Scalar      RangeType;

#if HAVE_DUNE_PDELAB
        // allows treating a RandomField as a PDELab function
        typedef typename Dune::YaspGrid<dim>::LeafGridView GridViewType;
        enum {dimRange  = 1};
        enum {dimDomain = dim};
#endif // HAVE_DUNE_PDELAB

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

        const std::array<RF,dim> extensions;
        unsigned int             level;
        std::array<RF,dim>       meshsize;
        RF                       cellVolume;

        RF              variance;
        std::vector<RF> corrLength;
        unsigned int    cgIterations;

        ptrdiff_t allocLocal, localN0, local0Start;

        // factor used in domain embedding
        int embeddingFactor;

        // properties of random field
        std::array<unsigned int,dim> cells;
        unsigned int                 domainSize;
        std::array<unsigned int,dim> localCells;
        std::array<unsigned int,dim> localOffset;
        unsigned int                 localDomainSize;

        // properties on extended domain
        std::array<unsigned int,dim> extendedCells;
        unsigned int                 extendedDomainSize;
        std::array<unsigned int,dim> localExtendedCells;
        std::array<unsigned int,dim> localExtendedOffset;
        unsigned int                 localExtendedDomainSize;

        mutable std::array<unsigned int,dim> globalIndices;
        mutable std::array<unsigned int,dim> localIndices;

        public:

        RandomFieldTraits(const Dune::ParameterTree& config_, const std::string& fieldName)
          : config(config_),
          extensions    (config.get<std::array<RF,dim> >          ("grid.extensions")),
          cgIterations  (config.get<unsigned int>                 ("randomField.cgIterations",100)),
          cells         (config.get<std::array<unsigned int,dim> >("grid.cells"))
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
        unsigned int indicesToIndex(const std::array<unsigned int,dim>& indices, const std::array<unsigned int,dim>& bound) const
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
        void indexToIndices(const unsigned int index, std::array<unsigned int,dim>& indices, const std::array<unsigned int,dim>& bound) const
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
        void coordsToIndices(const DomainType& location, std::array<unsigned int,dim>& localIndices, const std::array<unsigned int,dim>& offset) const
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
        void indicesToCoords(const std::array<unsigned int,dim>& localIndices, const std::array<unsigned int,dim>& offset, DomainType& location) const
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
