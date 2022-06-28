#ifndef DUNE_RANDOMFIELD_CPPRNGBACKEND_HH
#define DUNE_RANDOMFIELD_CPPRNGBACKEND_HH

#include <random>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Standard normal distribution based on C++11 random facilities
     *
     * This class provides a generator for Gaussian random numbers, based
     * on the C++11 random number generator facilities. This is the
     * default RNG if the GSL library wasn't found.
     *
     * @tparam Traits traits class providing data types and definitions
     */
    template<typename Traits>
      class CppRNGBackend
      {
        using RF = typename Traits::RF;

        std::default_random_engine generator;
        std::normal_distribution<RF> normalDist;

        public:

        /**
         * @brief Constructor
         *
         * @param traits object containing parameters and configuration
         */
        CppRNGBackend(const std::shared_ptr<Traits>& traits)
          :
            normalDist(0.,1.)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using CppRNGBackend" << std::endl;
        }

        /**
         * @brief (Re-)initialize random number generator
         *
         * This function puts the random number generator into a
         * known state, which makes it possible to create the same
         * sequence of numbers. Note that this may still lead to
         * different fields if the code is run on a different
         * platform or with a different parallel data distribution
         * (if any).
         *
         * @param seed seed value for the random number generator
         */
        void seed(unsigned int seed)
        {
          generator.seed(seed);
        }

        /**
         * @brief Produce sample from normally distributed random variable
         *
         * Draw a sample from the standard normal distribution.
         *
         * @return generated sample
         */
        RF sample()
        {
          return normalDist(generator);
        }
      };

  }
}

#endif // DUNE_RANDOMFIELD_CPPRNGBACKEND_HH
