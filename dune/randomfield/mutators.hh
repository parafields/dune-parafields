#ifndef DUNE_RANDOMFIELD_MUTATORS_HH
#define DUNE_RANDOMFIELD_MUTATORS_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Abstract base class for value transformations
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class MutatorBase
      {
        public:

          /**
           * @brief Apply transformation to given value
           *
           * @param[in,out] value value to transform
           */
          virtual void apply(RF& value) const = 0;

          /**
           * @brief Virtual destructor
           */
          virtual ~MutatorBase() {};
      };

    /**
     * @brief Default Identity mutator that leaves its argument unchanged
     *
     * This is the default mutator, it does nothing to its arguments, i.e.,
     * the generated field is a Gaussian random field.
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class IdentityMutator
      : public MutatorBase<RF>
      {
        public:

          void apply(RF& value) const
          {}
      };

    /**
     * @brief Exponential function mutator for log-normal fields
     *
     * This mutator replaces each value with its exponential, thereby
     * producing a log-normal random field.
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class LogNormalMutator
      : public MutatorBase<RF>
      {
        public:

          void apply(RF& value) const
          {
            value = std::exp(value);
          }
      };

    /**
     * @brief Absolute value function mutator for folded normal fields
     *
     * This mutator returns the absolute value of the random field.
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class FoldedNormalMutator
      : public MutatorBase<RF>
      {
        public:

          void apply(RF& value) const
          {
            value = std::abs(value);
          }
      };

    /**
     * @brief Replaces value with its sign (-1 for negative, +1 for nonnegative)
     *
     * This mutator returns the sign of the values. This can be used
     * to generate random subdomains (by using the two assigned values to define
     * an indicator function).
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class SignMutator
      : public MutatorBase<RF>
      {
        public:

          void apply(RF& value) const
          {
            if (value < 0.)
              value = -1.;
            else
              value = 1.;
          }
      };

    /**
     * @brief Applies the Box-Cox transformation to the data values
     *
     * This mutator applies the Box-Cox transformation to each data value.
     * The parameter lambda can be chosen through the configuration object.
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class BoxCoxMutator
      : public MutatorBase<RF>
      {
        const RF lambda;

        public:

        /**
         * @brief Constructor
         *
         * @param config ParameterTree object containing configuration
         */
        BoxCoxMutator(const Dune::ParameterTree& config)
          : lambda(config.get<RF>("transform.boxCoxLambda"))
        {}

        void apply(RF& value) const
        {
          value = std::pow(value*lambda + 1,-lambda);
        }
      };

    /**
     * @brief Value transform that applies one of several functions to the Gaussian random field
     *
     * @tparam RF data type of field values
     */
    template<typename RF>
      class ValueTransform
      {
        std::shared_ptr<MutatorBase<RF>> mutator;

        public:

        /**
         * @brief Constructor
         *
         * @param config ParameterTree object containing configuration
         */
        ValueTransform(const Dune::ParameterTree& config)
        {
          const std::string transformType = config.get<std::string>("randomField.transform","none");
          if (transformType == "none")
            mutator = std::shared_ptr<IdentityMutator<RF>>(new IdentityMutator<RF>);
          else if (transformType == "logNormal")
            mutator = std::shared_ptr<LogNormalMutator<RF>>(new LogNormalMutator<RF>);
          else if (transformType == "foldedNormal")
            mutator = std::shared_ptr<FoldedNormalMutator<RF>>(new FoldedNormalMutator<RF>);
          else if (transformType == "sign")
            mutator = std::shared_ptr<SignMutator<RF>>(new SignMutator<RF>);
          else if (transformType == "boxCox")
            mutator = std::shared_ptr<BoxCoxMutator<RF>>(new BoxCoxMutator<RF>(config));
          else
            DUNE_THROW(Dune::Exception,"transform type not known");
        }

        /**
         * @brief Apply the chosen mutator, transforming the value
         *
         * @param[in,out] value value to transform
         */
        void apply(RF& value) const
        {
          (*mutator).apply(value);
        }
      };

  }
}

#endif // DUNE_RANDOMFIELD_MUTATORS_HH
