// -*- tab-width: 2; indent-tabs-mode: nil -*-
#ifndef DUNE_RANDOMFIELD_GSLRNGBACKEND_HH
#define	DUNE_RANDOMFIELD_GSLRNGBACKEND_HH

#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif //HAVE_GSL

namespace Dune {
  namespace RandomField {

    /**
     * @brief Standard normal distribtion based on choice of GSL RNGs
     */
    template<typename Traits>
      class GSLRNGBackend
      {
#ifdef HAVE_GSL
        using RF = typename Traits::RF;

        gsl_rng* generator;

        enum Distribution {BoxMuller, RatioMethod, Ziggurat};

        Distribution type;

        public:

        GSLRNGBackend<Traits>(const std::shared_ptr<Traits>& traits)
        {
          if ((*traits).verbose && (*traits).rank == 0)
            std::cout << "using GSLRNGBackend" << std::endl;

          const std::string& rng
            = (*traits).config.template get<std::string>("random.rng","twister");
          if (rng == "twister")
            generator = gsl_rng_alloc(gsl_rng_mt19937);
          else if (rng == "ranlux")
            generator = gsl_rng_alloc(gsl_rng_ranlxd1);
          else if (rng == "tausworthe")
            generator = gsl_rng_alloc(gsl_rng_taus2);
          else if (rng == "gfsr4")
            generator = gsl_rng_alloc(gsl_rng_gfsr4);
          else
            DUNE_THROW(Dune::Exception,
                "GSLRNGBackend accepts \"twister\", \"ranlux\", \"tausworthe\" and \"gfsr4\"as RNGs");

          const std::string& dist
            = (*traits).config.template get<std::string>("random.distribution","ziggurat");

          if (dist == "boxMuller")
            type = BoxMuller;
          else if (dist == "ratioMethod")
            type = RatioMethod;
          else if (dist == "ziggurat")
            type = Ziggurat;
          else
            DUNE_THROW(Dune::Exception,
                "GSLRNGBackend accepts \"boxMuller\", \"ratioMethod\""
                " and \"ziggurat\" as distributions");
        }

        /**
         * @brief (Re-)initialize random number generator
         */
        void seed(unsigned int seed)
        {
          gsl_rng_set(generator,seed);
        }

        /**
         * @brief Produce sample from normally distributed random variable
         */
        RF sample()
        {
          switch (type)
          {
            case BoxMuller:   return gsl_ran_gaussian(generator,1.);
            case RatioMethod: return gsl_ran_gaussian_ratio_method(generator,1.);
            case Ziggurat:    return gsl_ran_gaussian_ziggurat(generator,1.);

            default: DUNE_THROW(Dune::Exception,
                         "unknown distribution type");
          }
        }

        ~GSLRNGBackend<Traits>()
        {
          if (generator != nullptr)
            gsl_rng_free(generator);
        }

#else // HAVE_GSL
        static_assert(false,"GSLRNGBackend requires Gnu Scientific Library (GSL)");
#endif // HAVE_GSL
      };

  }
}

#endif // DUNE_RANDOMFIELD_GSLRNGBACKEND_HH
