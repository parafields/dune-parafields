# File for module specific CMake tests.

find_package(FFTW3 COMPONENTS double mpi REQUIRED)
include(AddFFTW3Flags)

find_package(HDF5)
if (HDF5_FOUND AND NOT HDF5_IS_PARALLEL)
  message("HDF5 found, but has not been compiled with MPI support.")
endif()

if (HDF5_FOUND AND HDF5_IS_PARALLEL)
  add_definitions(-DHAVE_HDF5=1)
else()
  add_definitions(-DHAVE_HDF5=0)
  message("No parallel HDF5, HAVE_HDF5 set to false.")
endif()

dune_register_package_flags(LIBRARIES ${HDF5_LIBRARIES}
                            INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})

include(AddHDF5Flags)
