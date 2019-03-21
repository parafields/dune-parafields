dune_symlink_to_source_files(FILES randomfield1d.ini randomfield2d.ini
  randomfield3d.ini randomfield4d.ini)

foreach(dim 1 2 3 4)
  set(name fieldtest_${dim})
  dune_add_test(SOURCES fieldtest.cc
                NAME ${name}
                COMPILE_DEFINITIONS DIMENSION=${dim})
endforeach()
