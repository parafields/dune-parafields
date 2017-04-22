# File for module specific CMake tests.

find_package(FFTW3 COMPONENTS double mpi REQUIRED)
include(AddFFTW3Flags)

find_package(HDF5 REQUIRED)
if (HDF5_FOUND AND NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "HDF5 has not been compiled with MPI support.")
endif()

dune_register_package_flags(LIBRARIES ${HDF5_LIBRARIES}
                            INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})

include(AddHDF5Flags)
