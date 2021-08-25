#ifndef DUNE_RANDOMFIELD_MUTATORS_HH
#define DUNE_RANDOMFIELD_MUTATORS_HH

namespace Dune {
  namespace RandomField {

    template<typename RF>
      class MutatorBase
      {
        public:

          virtual void apply(RF& value) const = 0;

          virtual ~MutatorBase() {};
      };

    /**
     * @brief Default Identity mutator that leaves its argument unchanged
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

    template<typename RF>
      class BoxCoxMutator
      : public MutatorBase<RF>
      {
        const RF lambda;

        public:

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
     */
    template<typename RF>
      class ValueTransform
      {
        std::shared_ptr<MutatorBase<RF>> mutator;

        public:

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

        void apply(RF& value) const
        {
          (*mutator).apply(value);
        }
      };

  }
}

#endif // DUNE_RANDOMFIELD_MUTATORS_HH
