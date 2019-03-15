dune_symlink_to_source_files(FILES randomfield1d.ini randomfield2d.ini randomfield3d.ini)

foreach(dim 1 2 3)
  set(name fieldtest_${dim})
  dune_add_test(SOURCES fieldtest.cc
                NAME ${name}
                COMPILE_DEFINITIONS DIMENSION=${dim})
endforeach()
