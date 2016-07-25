dune_symlink_to_source_files(FILES randomfield2d.ini randomfield3d.ini perm.props)

foreach(dim 2 3)
  foreach(invMat 1 0)
    foreach(invRoot 1 0)
        set(name fieldtest_${dim}_${invMat}_${invRoot})
    dune_add_test(SOURCES fieldtest.cc
                  NAME ${name}
                  COMPILE_DEFINITIONS DIMENSION=${dim} INVMAT=${invMat} INVROOT=${invRoot})
#  target_link_dune_default_libraries(${name})
#  add_dune_fftw3_flags(${name})
#  add_dune_hdf5_flags(${name})
    endforeach()
  endforeach()
endforeach()
