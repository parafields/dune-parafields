#ifndef DUNE_RANDOMFIELD_OPTPROBLEM_HH
#define	DUNE_RANDOMFIELD_OPTPROBLEM_HH

#if HAVE_DUNE_NONLINOPT

#include<dune/nonlinopt/nonlinopt.hh>

#include<dune/randomfield/legacyvtk.hh>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Wrapper class with vector arithmetics for optimization
     *
     * @tparam Traits traits class for configuration
     */
    template<typename Traits>
      class VectorWrapper
      {
        public:

          using Real  = typename Traits::RF;
          using Index = typename Traits::Index;


        private:

          unsigned int data_size = 0;
          fftw_complex* data = nullptr;

          enum {dim = Traits::dim};

          std::array<Index,dim> extendedCells;
          Index extendedDomainSize = 0;
          MPI_Comm comm = MPI_COMM_WORLD;

        public:

          /**
           * @brief Default constructor
           *
           * Constructs empty wrapper, representing the origin.
           */
          VectorWrapper()
          {}

          /**
           * @brief Constructor
           *
           * Creates vector wrapper from given data backend.
           *
           * @param backend             backend to extract data from
           * @param extendedCells_      number of cells per dimension
           * @param extendedDomainSize_ number of data entries
           * @param comm_               MPI communicator
           */
          template<typename Backend>
            VectorWrapper(const Backend& backend, std::array<Index,dim> extendedCells_,
                Index extendedDomainSize_, MPI_Comm comm_)
            : data_size(backend.localMatrixSize()), extendedCells(extendedCells_),
            extendedDomainSize(extendedDomainSize_), comm(comm_)
        {
          data = FFTW<Real>::alloc_complex(data_size);

          for (Index i = 0; i < data_size; ++i)
          {
            data[i][0] = backend.get(i); 
            data[i][1] = 0.;
          }
        }

          /**
           * @brief Copy constructor
           *
           * @param other other vector wrapper to copy from
           */
          VectorWrapper(const VectorWrapper& other)
            : data_size(other.data_size), extendedCells(other.extendedCells),
            extendedDomainSize(other.extendedDomainSize), comm(other.comm)
        {
          if (other.data)
          {
            data = FFTW<Real>::alloc_complex(data_size);

            for (Index i = 0; i < data_size; ++i)
            {
              data[i][0] = other.data[i][0]; 
              data[i][1] = other.data[i][1];
            }
          }
          else
            data = nullptr;
        }

          /**
           * @brief Move constructor
           *
           * @param other other vector wrapper to move from
           */
          VectorWrapper(VectorWrapper&& other)
            : data_size(std::move(other.data_size)), data(std::move(other.data)),
            extendedCells(std::move(other.extendedCells)),
            extendedDomainSize(std::move(other.extendedDomainSize)), comm(std::move(other.comm))
        {
          other.data = nullptr;
          other.data_size = 0;

        }

          /**
           * @brief Destructor
           */
          ~VectorWrapper()
          {
            if (data)
              FFTW<Real>::free(data);
          }

          /**
           * @brief Assignment operator
           *
           * @param other other vector wrapper to copy from
           *
           * @return reference to current wrapper
           */
          VectorWrapper& operator=(const VectorWrapper& other)
          {
            if (this == &other)
              return *this;

            if (data_size != other.data_size)
            {
              data_size = other.data_size;
              extendedCells = other.extendedCells;
              extendedDomainSize = other.extendedDomainSize;

              if (data)
                FFTW<Real>::free(data);

              if (data_size)
                data = FFTW<Real>::alloc_complex(data_size);
              else
                data = nullptr;
            }

            if (other.data)
              for (Index i = 0; i < data_size; ++i)
              {
                data[i][0] = other.data[i][0];
                data[i][1] = other.data[i][1];
              }

            return *this;
          }

          /**
           * @brief Move assignment operator
           *
           * @param other other vector wrapper to move from
           * 
           * @return reference to current wrapper
           */
          VectorWrapper& operator=(VectorWrapper&& other)
          {
            data_size = std::move(other.data_size);
            data = std::move(other.data);
            extendedCells = std::move(other.extendedCells);
            extendedDomainSize = std::move(other.extendedDomainSize);
            comm = std::move(other.comm);

            other.data_size = 0;
            other.data = nullptr;

            return *this;
          }

          /**
           * @brief Access to underlying data vector
           *
           * @return raw data pointer
           */
          fftw_complex* raw() const
          {
            return data;
          }

          /**
           * @brief Scaled multiply-add
           *
           * Adds a scalar multiple of another vector wrapper.
           *
           * @param other other vector wrapper to add
           * @param alpha scale factor
           */
          void axpy(const VectorWrapper& other, Real alpha)
          {
            if (other.data)
            {
              if (!data)
              {
                data_size = other.data_size;
                extendedCells = other.extendedCells;
                extendedDomainSize = other.extendedDomainSize;
                data = FFTW<Real>::alloc_complex(data_size);

                for (Index i = 0; i < data_size; ++i)
                {
                  data[i][0] = alpha * other.data[i][0];
                  data[i][1] = alpha * other.data[i][1];
                }
              }
              else
                for (Index i = 0; i < data_size; ++i)
                {
                  data[i][0] += alpha * other.data[i][0];
                  data[i][1] += alpha * other.data[i][1];
                }
            }
          }

          /**
           * @brief Multiplication with scalar
           *
           * @param alpha scale factor
           */
          void operator*=(Real alpha)
          {
            if (data)
              for (Index i = 0; i < data_size; ++i)
              {
                data[i][0] *= alpha;  
                data[i][1] *= alpha;  
              }
          }

          /**
           * @brief Vector addition
           *
           * @param other other vector wrapper to add
           */
          void operator+=(const VectorWrapper& other)
          {
            if (other.data)
            {
              if (!data)
              {
                data_size = other.data_size;
                extendedCells = other.extendedCells;
                extendedDomainSize = other.extendedDomainSize;
                data = FFTW<Real>::alloc_complex(data_size);

                for (Index i = 0; i < data_size; ++i)
                {
                  data[i][0] = other.data[i][0];
                  data[i][1] = other.data[i][1];
                }
              }
              else
                for (Index i = 0; i < data_size; ++i)
                {
                  data[i][0] += other.data[i][0];
                  data[i][1] += other.data[i][1];
                }
            }
          }

          /**
           * @brief Vector subtraction
           *
           * @param other other vector wrapper to subtract
           */
          void operator-=(const VectorWrapper& other)
          {
            if (other.data)
            {
              if (!data)
              {
                data_size = other.data_size;
                extendedCells = other.extendedCells;
                extendedDomainSize = other.extendedDomainSize;
                data = FFTW<Real>::alloc_complex(data_size);

                for (Index i = 0; i < data_size; ++i)
                {
                  data[i][0] = -other.data[i][0];
                  data[i][1] = -other.data[i][1];
                }
              }
              else
                for (Index i = 0; i < data_size; ++i)
                {
                  data[i][0] -= other.data[i][0];
                  data[i][1] -= other.data[i][1];
                }
            }
          }

          /**
           * @brief Calculate value at given tuple of indices
           *
           * @tparam Indices data type of indices
           * @tparam Vector  storage type with one entry
           *
           * @param indices  indices for multidimensional array
           * @param vector   location where value should be stored
           */
          template<typename Indices, typename Vector>
            void evaluate(const Indices& indices, Vector& vector) const
            {
              if (data_size == 0)
              {
                vector[0] = 0.;
                return;
              }

              if constexpr (dim == 2)
                vector[0] = data[Index((indices[1]-0.5) * extendedCells[0]
                    + indices[0]-0.5)%extendedDomainSize][0];
              else if constexpr (dim == 3)
                vector[0] = data[Index(((indices[2]-0.5) * extendedCells[1]
                      + indices[1]-0.5) * extendedCells[0] + indices[0]-0.5)%extendedDomainSize][0];
              else
                DUNE_THROW(Dune::Exception,"not implemented");
            }

          /**
           * @brief Perform forward DFT
           */
          void forwardTransform()
          {
            if (data)
            {
              unsigned int flags = FFTW_ESTIMATE;

              ptrdiff_t n[dim];
              for (unsigned int i = 0; i < dim; i++)
                n[i] = extendedCells[dim-1-i];

              typename FFTW<Real>::plan plan_forward = FFTW<Real>::mpi_plan_dft(dim,n,data,
                  data,comm,FFTW_FORWARD,flags);

              if (plan_forward == nullptr)
                DUNE_THROW(Dune::Exception, "failed to create forward plan");

              FFTW<Real>::execute(plan_forward);
              FFTW<Real>::destroy_plan(plan_forward);

              for (Index i = 0; i < data_size; ++i)
              {
                data[i][0] /= extendedDomainSize;
                data[i][1] /= extendedDomainSize;
              }
            }
          }

          /**
           * @brief Perform backward DFT
           */
          void backwardTransform()
          {
            if (data)
            {
              unsigned int flags = FFTW_ESTIMATE;

              ptrdiff_t n[dim];
              for (unsigned int i = 0; i < dim; i++)
                n[i] = extendedCells[dim-1-i];

              typename FFTW<Real>::plan plan_backward = FFTW<Real>::mpi_plan_dft(dim,n,data,
                  data,comm,FFTW_BACKWARD,flags);

              if (plan_backward == nullptr)
                DUNE_THROW(Dune::Exception, "failed to create backward plan");

              FFTW<Real>::execute(plan_backward);
              FFTW<Real>::destroy_plan(plan_backward);
            }
          }

          /**
           * @brief Remove data that is below given (negative of) threshold
           *
           * @param shift      shift applied to data before truncation
           * @param threshold  threshold used for truncation
           * @param multiplier multiplier applied to data before truncation
           * @param logSumExp  apply LogSumExp transformation if true
           */
          unsigned int makePositive(Real shift, Real threshold, Real multiplier = 1., bool logSumExp = false)
          {
            unsigned int negative = 0;

            if (logSumExp)
            {
              for (Index i = 0; i < data_size; ++i)
              {
                if (data[i][0] < -threshold)
                  negative++;

                data[i][0] = std::log1p(std::exp((data[i][0]-shift) * multiplier)) / multiplier;
              }
            }
            else
            {
              for (Index i = 0; i < data_size; ++i)
              {
                if (data[i][0]*multiplier < -threshold)
                  negative++;

                data[i][0] = std::max(data[i][0]-shift,0.);
              }
            }

            return negative;
          }

          /**
           * @brief Mask data based on complement of constraint vector
           *
           * @param constrained vector of constrained indices
           */
          void removeUnconstrained(const std::vector<bool>& constrained)
          {
            if (constrained.size() != data_size)
              DUNE_THROW(Dune::Exception,"size mismatch");

            for (Index i = 0; i < data_size; ++i)
              if (!constrained[i])
                data[i][0] = 0.;
          }

          /**
           * @brief Mask data based on constraint vector
           *
           * @param constrained vector of constrained indices
           */
          void removeConstrained(const std::vector<bool>& constrained)
          {
            if (constrained.size() != data_size)
              DUNE_THROW(Dune::Exception,"size mismatch");

            for (Index i = 0; i < data_size; ++i)
              if (constrained[i])
                data[i][0] = 0.;
          }

          /**
           * @brief Number of entries in vector
           *
           * @return length of data vector
           */
          unsigned int size() const
          {
            return data_size;
          }

          /**
           * @brief Scalar product
           *
           * @param other other vector wrapper to multiply with
           *
           * @return resulting scalar product
           */
          Real operator*(const VectorWrapper& other) const
          {
            if (data_size == 0 || other.data_size == 0)
              return 0.;

            if (data_size != other.data_size)
              DUNE_THROW(Dune::Exception,"size mismatch");

            Real output = 0.;
            for (Index i = 0; i < data_size; ++i)
              output += data[i][0]*other.data[i][0];

            return output;
          }

          /**
           * @brief Minimum across data entries
           *
           * @return minimum data value
           */
          Real min() const
          {
            if (!data)
              return 0.;

            Real minValue = 0.;
            for (Index i = 0; i < data_size; ++i)
            {
              const Real value = data[i][0];
              if (value < minValue)
                minValue = value;
            }

            return minValue;
          }

          /**
           * @brief Maximum norm
           *
           * @return sum of absolute values
           */
          Real inf_norm() const
          {
            if (!data)
              return 0.;

            Real maxValue = 0.;

            for (Index i = 0; i < data_size; ++i)
            {
              const Real absValue = std::abs(data[i][0]);
              if (absValue > maxValue)
                maxValue = absValue;
            }

            return maxValue;
          }

          /**
           * @brief Equality comparison
           *
           * @param other other vector wrapper to compare to
           *
           * @return true iff all data entries are the same
           */
          bool operator==(const VectorWrapper& other) const
          {
            for (Index i = 0; i < data_size; ++i)
              if (data[i][0] != other.data[i][0])
                return false;

            return true;
          }

          /**
           * @brief Inequality comparison
           *
           * @param other other vector wrapper to compare to
           *
           * @return true iff at least one entry differs
           */
          bool operator!=(const VectorWrapper& other) const
          {
            return !operator==(other);
          }
      };

    /**
     * @brief Class representing a conic feasibility problem
     *
     * This class represents an objective function that should be
     * minimized to obtain a point in the intersection of the linear
     * affine space of covariance functions with correct values on
     * the original domain and the cone of covariance functions with
     * non-negative Fourier modes (if it exists). Choices are a)
     * the squared Euclidean distance from the cone, restricted to
     * the affine space, b) the sum of the (squared) Euclidean
     * distances from both cone and affine space, and c) a smooth
     * approximation of the maximum norm distance from the cone,
     * using the LogSumExp function. See the base class in dune-nonlinopt
     * for a description of the class methods.
     *
     * @tparam Traits traits class for configuration
     */
    template<typename Traits>
      class ConeOptimizationProblem
      : public Dune::NonlinOpt::ProblemBase<typename VectorWrapper<Traits>::Real,VectorWrapper<Traits>>
      {
        public:

          enum {spatialDim = Traits::dim};
          using Point = VectorWrapper<Traits>;
          using Real = typename VectorWrapper<Traits>::Real;

        private:

          const Dune::ParameterTree& config;
          Point start;
          const std::vector<bool>& constrained;
          Real shift, threshold;
          unsigned int extendedDomainSize;
          unsigned int localExtendedDomainSize;
          const std::array<unsigned int,spatialDim>& extendedCells;
          const MPI_Comm comm;
          mutable unsigned int iteration = 0;
          mutable unsigned int forward = 0;
          mutable unsigned int backward = 0;

          mutable Point current;
          mutable Real  affineVal = std::numeric_limits<Real>::max();
          mutable Real  coneVal   = std::numeric_limits<Real>::max();
          mutable unsigned int negative = std::numeric_limits<unsigned int>::max();
          mutable Real logSumExpFactor = 0.;

        public:

          ConeOptimizationProblem(const Dune::ParameterTree& config_, const Point& start_,
              const std::vector<bool>& constrained_, Real shift_, Real threshold_, unsigned int extendedDomainSize_,
              unsigned int localExtendedDomainSize_, const std::array<unsigned int,spatialDim>& extendedCells_,
              const MPI_Comm comm_)
            : config(config_), start(start_), constrained(constrained_), shift(shift_), threshold(threshold_),
            extendedDomainSize(extendedDomainSize_), localExtendedDomainSize(localExtendedDomainSize_),
            extendedCells(extendedCells_), comm(comm_), current(zero())
        {
          if (constrained.size() != localExtendedDomainSize)
            DUNE_THROW(Dune::Exception,"size mismatch");
        }

          Real value(const Point& x, bool subsequent = false) const
          {

            if (subsequent)
              return affineVal + coneVal;

            if (!config.template get<bool>("stochastic.stayOnAffine",false))
            {
              current = x;
              current -= start;

              current.removeUnconstrained(constrained);

              affineVal = 0.5 * (current * current);
            }
            else
              affineVal = 0.;

            current = x;
            current.forwardTransform();
            forward++;

            if (config.template get<bool>("stochastic.logSumExp",false))
            {
              negative = 0;
              Real max = std::numeric_limits<Real>::min();
              for (unsigned int i = 0; i < localExtendedDomainSize; i++)
              {
                if (current.raw()[i][0] < -threshold)
                  negative++;
                max = std::max(max,-current.raw()[i][0]);
              }
              if (logSumExpFactor == 0.)
                logSumExpFactor = 1./max;
              Real sum = 0.;
              for (unsigned int i = 0; i < localExtendedDomainSize; i++)
                sum += std::exp(-logSumExpFactor*(current.raw()[i][0] - max));

              coneVal = max + std::log(sum) - std::log(extendedDomainSize);
            }
            else
            {
              current *= -1.;
              negative = current.makePositive(shift,threshold,-1.);

              coneVal = 0.5 * (current * current) * extendedDomainSize;
            }

            return affineVal + coneVal;
          }

          Real dir_deriv(const Point& x, const Point& direction,
              Point& grad, bool subsequent = false) const
          {
            if (!config.template get<bool>("stochastic.stayOnAffine",false))
            {
              if (!subsequent)
              {
                grad = x;
                grad -= start;

                grad.removeUnconstrained(constrained);
              }

              affineVal = 0.5 * (grad * grad);
            }
            else
            {
              affineVal = 0.;

              grad *= 0.;
            }

            current = x;
            current.forwardTransform();
            forward++;

            if (config.template get<bool>("stochastic.logSumExp",false))
            {
              negative = 0;
              Real max = std::numeric_limits<Real>::min();
              for (unsigned int i = 0; i < localExtendedDomainSize; i++)
              {
                if (current.raw()[i][0] < -threshold)
                  negative++;
                max = std::max(max,-current.raw()[i][0]);
              }
              if (logSumExpFactor == 0.)
                logSumExpFactor = 1./max;
              Real sum = 0.;
              for (unsigned int i = 0; i < localExtendedDomainSize; i++)
              {
                sum += std::exp(-logSumExpFactor*(current.raw()[i][0] - max));
                current.raw()[i][0] = logSumExpFactor * std::exp(-logSumExpFactor*current.raw()[i][0]);
                current.raw()[i][1] = 0.;
                if (!std::isfinite(current.raw()[i][0]))
                  current.raw()[i][0] = 0.;
              }
              current *= 1./std::exp(max + std::log(sum));

              current.backwardTransform();
              backward++;

              coneVal = max + std::log(sum) - std::log(extendedDomainSize);
            }
            else
            {
              current *= -1.;
              negative = current.makePositive(shift,threshold,-1.);

              current.backwardTransform();
              backward++;

              coneVal = 0.5 * (current * current);
            }

            grad -= current;

            if (config.template get<bool>("stochastic.stayOnAffine",false))
              grad.removeConstrained(constrained);

            return grad * direction;
          }

          void gradient(const Point& x, Point& grad,
              const Point* const direction = nullptr, Real derivative = 0.) const
          {
            if (!direction)
              dir_deriv(x,x,grad);
          }

          std::pair<Real,Real> value_and_deriv(const Point& x,
              const Point& direction, Point& grad) const
          {
            this->gradient(x,grad);
            return {value(x,true), grad * direction};
          }

          Point zero() const
          {
            return Point();
          }

          unsigned int optimizationStep() const
          {
            return iteration;
          }

          unsigned int forwards() const
          {
            return forward;
          }

          unsigned int backwards() const
          {
            return backward;
          }

          std::size_t dim() const
          {
            return extendedDomainSize;
          }

          unsigned int negatives() const
          {
            return negative;
          }

          void hook(std::size_t iter, const Point& point, Real val,
              const Point& grad, bool extrapolation = false) const
          {
            iteration = iter;
          }

      };

    /**
     * @brief Wrapper class representing the dual of a projection problem.
     *
     * This class represents the dual of the Euclidean projection
     * problem for the intersection between the set of covariance functions
     * with correct values on the original domain and the set of covariance
     * functions with non-negative Fourier modes. This is the intersection
     * of a cone and an affine linear space, and if it is non-empty, then
     * the projection is a viable covariance function on the extended domain.
     * The main benefit of this formulation is the fact that the
     * Lagrange multipliers are used as free variable, and these have the
     * dimension of the original domain, irrespective of the chosen embedding
     * factor. See the base class in dune-nonlinopt for a description of the
     * class methods.
     *
     * @tparam Traits traits class for configuration
     */
    template<typename Traits>
      class DualOptimizationProblem
      : public Dune::NonlinOpt::ProblemBase<typename VectorWrapper<Traits>::Real, VectorWrapper<Traits>>
      {
        public:

          enum {spatialDim = Traits::dim};
          using Point = VectorWrapper<Traits>;
          using Real = typename VectorWrapper<Traits>::Real;

        private:

          const Dune::ParameterTree& config;
          const Point& start, bound;
          const std::vector<bool>& constrained;
          Real shift, threshold;
          unsigned int extendedDomainSize;
          unsigned int localExtendedDomainSize;
          const std::array<unsigned int,spatialDim>& extendedCells;
          const MPI_Comm comm;
          mutable unsigned int iteration = 0;
          mutable unsigned int forward = 0;
          mutable unsigned int backward = 0;

          mutable Point projectedPoint;
          mutable Real  val = std::numeric_limits<Real>::max();

        public:

          DualOptimizationProblem(const Dune::ParameterTree& config_, const Point& start_,
              const Point& bound_, const std::vector<bool>& constrained_, Real shift_, Real threshold_,
              unsigned int extendedDomainSize_, unsigned int localExtendedDomainSize_,
              const std::array<unsigned int,spatialDim>& extendedCells_, const MPI_Comm comm_)
            : config(config_), start(start_), bound(bound_), constrained(constrained_), shift(shift_),
            threshold(threshold_), extendedDomainSize(extendedDomainSize_),
            localExtendedDomainSize(localExtendedDomainSize_), extendedCells(extendedCells_),
            comm(comm_), projectedPoint(zero())
        {
          if (constrained.size() != localExtendedDomainSize)
            DUNE_THROW(Dune::Exception,"size mismatch");
        }

          Real value(const Point& x, bool subsequent = false) const
          {
            if (!subsequent)
            {
              projectedPoint = start;
              projectedPoint += x;

              projectedPoint.forwardTransform();
              forward++;

              projectedPoint.makePositive(shift,threshold);

              val = 0.5*(projectedPoint*projectedPoint) * extendedDomainSize
                - 0.5*(start*start) - (start*x);
            }

            return val;
          }

          Real dir_deriv(const Point& x, const Point& direction,
              Point& grad, bool subsequent = false) const
          {
            this->value(x,subsequent);

            grad = projectedPoint;

            grad.backwardTransform();
            backward++;

            grad -= bound;
            grad.removeUnconstrained(constrained);

            return grad * direction;
          }

          void gradient(const Point& x, Point& grad,
              const Point* const direction = nullptr, Real derivative = 0.) const
          {
            if (!direction)
              dir_deriv(x,x,grad); 
          }

          std::pair<Real,Real> value_and_deriv(const Point& x,
              const Point& direction, Point& grad) const
          {
            this->gradient(x,grad);
            return {value(x,true), grad * direction};
          }

          Point zero() const
          {
            return Point();
          }

          unsigned int optimizationStep() const
          {
            return iteration;
          }

          unsigned int forwards() const
          {
            return forward;
          }

          unsigned int backwards() const
          {
            return backward;
          }

          std::size_t dim() const
          {
            return extendedDomainSize;
          }

          void hook(std::size_t iter, const Point& point, Real val,
              const Point& grad, bool extrapolation = false) const
          {
            iteration = iter;
          }

      };

  }
}

#endif // HAVE_DUNE_NONLINOPT

#endif // DUNE_RANDOMFIELD_OPTPROBLEM_HH
