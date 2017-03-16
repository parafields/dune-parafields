// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_MUTATORS_HH
#define	DUNE_RANDOMFIELD_MUTATORS_HH

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
#include<dune/randomfield/mutators.hh>

namespace Dune {
  namespace RandomField {

    template<typename RF>
      class MutatorBase
      {
        public:

          virtual void apply(RF& value) const = 0;
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
        Dune::shared_ptr<MutatorBase<RF> > mutator;

        public:

        ValueTransform(const Dune::ParameterTree& config)
        {
          const std::string transformType = config.get<std::string>("randomField.transform","none");
          if (transformType == "none")
            mutator = std::shared_ptr<IdentityMutator<RF> >(new IdentityMutator<RF>);
          else if (transformType == "logNormal")
            mutator = std::shared_ptr<LogNormalMutator<RF> >(new LogNormalMutator<RF>);
          else if (transformType == "foldedNormal")
            mutator = std::shared_ptr<FoldedNormalMutator<RF> >(new FoldedNormalMutator<RF>);
          else if (transformType == "sign")
            mutator = std::shared_ptr<SignMutator<RF> >(new SignMutator<RF>);
          else if (transformType == "boxCox")
            mutator = std::shared_ptr<BoxCoxMutator<RF> >(new BoxCoxMutator<RF>(config));
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
