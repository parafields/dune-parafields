# File for module specific CMake tests.

find_package(FFTW3 REQUIRED)
include(AddFFTW3Flags)

find_package(HDF5 REQUIRED)
include(AddHDF5Flags)
