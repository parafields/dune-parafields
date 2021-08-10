// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_CPPRNGBACKEND_HH
#define	DUNE_RANDOMFIELD_CPPRNGBACKEND_HH

#include <random>

namespace Dune {
  namespace RandomField {

    /**
     * @brief Standard normal distribution based on C++11 <random> header
     */
    template<typename Traits>
      class CppRNGBackend
      {
        using RF = typename Traits::RF;

        std::default_random_engine generator;
        std::normal_distribution<RF> normalDist;

        public:

        CppRNGBackend<Traits>(const std::shared_ptr<Traits>& traits)
          :
            normalDist(0.,1.)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using CppRNGBackend" << std::endl;
        }

        /**
         * @brief (Re-)initialize random number generator
         */
        void seed(unsigned int seed)
        {
          generator.seed(seed);
        }

        /**
         * @brief Produce sample from normally distributed random variable
         */
        RF sample()
        {
          return normalDist(generator);
        }
      };

  }
}

#endif // DUNE_RANDOMFIELD_CPPRNGBACKEND_HH
