add_subdirectory(backends)

#install headers
install(FILES covariance.hh
              fieldtraits.hh
              io.hh
              legacyvtk.hh
              matrix.hh
              mutators.hh
              pngreader.hh
              randomfield.hh
              stochastic.hh
              trend.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/parafields
       )
