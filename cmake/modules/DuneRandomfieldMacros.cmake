# File for module specific CMake tests.

find_package(FFTW3 REQUIRED)
if (FFTW3_FLOAT_FOUND)
  add_definitions(-DHAVE_FFTW3_FLOAT)
endif()
if (FFTW3_DOUBLE_FOUND)
  add_definitions(-DHAVE_FFTW3_DOUBLE)
endif()
if (FFTW3_LONGDOUBLE_FOUND)
  add_definitions(-DHAVE_FFTW3_LONGDOUBLE)
endif()

find_package(HDF5)
if (HDF5_FOUND AND HDF5_IS_PARALLEL)
  add_definitions(-DHAVE_HDF5=1)
  dune_register_package_flags(LIBRARIES ${HDF5_LIBRARIES}
                              INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
else()
  if (HDF5_FOUND)
    message("HDF5 found, but has not been compiled with MPI support.")
    set(HDF5_FOUND FALSE)
  else()
    message("HDF5 has not been found.")
  endif()
  add_definitions(-DHAVE_HDF5=0)
  message("No parallel HDF5, HAVE_HDF5 set to false.")
endif()

find_package(PNG)
if (PNG_FOUND)
  dune_register_package_flags(LIBRARIES ${PNG_LIBRARIES}
                              INCLUDE_DIRS ${PNG_INCLUDE_DIRS})
  add_definitions(-DHAVE_PNG ${PNG_DEFINITIONS})
endif()

find_package(GSL)
if (GSL_FOUND)
  dune_register_package_flags(LIBRARIES ${GSL_LIBRARIES}
                              INCLUDE_DIRS ${GSL_INCLUDE_DIRS})
  add_definitions(-DHAVE_GSL ${GSL_DEFINITIONS})
endif()
