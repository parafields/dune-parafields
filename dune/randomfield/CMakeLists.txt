#install headers
install(FILES fieldtraits.hh
              io.hh
              matrix.hh
              randomfield.hh
              stochastic.hh
              trend.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/randomfield
       )
