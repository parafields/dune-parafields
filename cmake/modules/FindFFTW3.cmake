# - Try to find FFTW3.
# Usage: find_package(FFTW3)
#
# Variables used by this module:
# FFTW3_ROOT_DIR - hint for location of FFTW3
# Variables defined by this module:
# FFTW3_FOUND - system has (at least one) parallel FFTW3
# FFTW3_FLOAT_FOUND - system has float parallel FFTW3
# FFTW3_DOUBLE_FOUND - system has double parallel FFTW3
# FFTW3_LONGDOUBLE_FOUND - system has long double parallel FFTW3
# FFTW3_FLOAT_LIB - float version of FFTW3 library
# FFTW3_MPI_FLOAT_LIB - parallel float FFTW3 library
# FFTW3_DOUBLE_LIB - double FFTW3 library
# FFTW3_MPI_DOUBLE_LIB - parallel double FFTW3 library
# FFTW3_LONGDOUBLE_LIB - long double FFTW3 library
# FFTW3_MPI_LONGDOUBLE_LIB - parallel long double FFTW3 library
# FFTW3_INCLUDE_DIR - FFTW3 include directory

# Search for the header files.
find_path(FFTW3_INCLUDE_DIR fftw3.h fftw3-mpi.h
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES include)
mark_as_advanced(FFTW3_INCLUDE_DIR)

# Search for float library.
find_library(FFTW3_FLOAT_LIB fftw3f
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(FFTW3_FLOAT_LIB)
if (FFTW3_FLOAT_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_FLOAT_LIB})
endif()

# Search for parallel float library.
find_library(FFTW3_MPI_FLOAT_LIB fftw3f_mpi
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(FFTW3_MPI_FLOAT_LIB)
if (FFTW3_MPI_FLOAT_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_MPI_FLOAT_LIB})
endif()

set (FFTW3_FLOAT_FOUND (FFW3_FLOAT_LIB AND FFTW3_MPI_FLOAT_LIB))
mark_as_advanced(FFTW3_FLOAT_FOUND)

# Search for double library.
find_library(FFTW3_DOUBLE_LIB fftw3
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(FFTW3_DOUBLE_LIB)
if (FFTW3_DOUBLE_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_DOUBLE_LIB})
endif()

# Search for parallel double library.
find_library(FFTW3_MPI_DOUBLE_LIB fftw3_mpi
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(FFTW3_MPI_DOUBLE_LIB)
if (FFTW3_MPI_DOUBLE_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_MPI_DOUBLE_LIB})
endif()

set (FFTW3_DOUBLE_FOUND (FFW3_DOUBLE_LIB AND FFTW3_MPI_DOUBLE_LIB))
mark_as_advanced(FFTW3_DOUBLE_FOUND)

# Search for long double library.
find_library(FFTW3_LONGDOUBLE_LIB fftw3l
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(FFTW3_LONGDOUBLE_LIB)
if (FFTW3_LONGDOUBLE_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_LONGDOUBLE_LIB})
endif()

# Search for parallel long double library.
find_library(FFTW3_MPI_LONGDOUBLE_LIB fftw3l_mpi
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib lib64)
mark_as_advanced(FFTW3_MPI_LONGDOUBLE_LIB)
if (FFTW3_MPI_LONGDOUBLE_LIB)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARIES} ${FFTW3_MPI_LONGDOUBLE_LIB})
endif()

set (FFTW3_LONGDOUBLE_FOUND (FFW3_LONGDOUBLE_LIB AND FFTW3_MPI_LONGDOUBLE_LIB))
mark_as_advanced(FFTW3_LONGDOUBLE_FOUND)

# Check whether at least pair of libraries was found.
set(FFTW3_LIB (FFTW3_FLOAT_FOUND OR FFTW3_DOUBLE_FOUND OR FFTW3_LONGDOUBLE_FOUND))

if (FFTW3_INCLUDE_DIR AND NOT FFTW3_LIB)
  message(WARN "found FFTW3 include dir, but no parallel libraries")
endif()

# Handle the QUIETLY and REQUIRED arguments and set FFTW3_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 REQUIRED_VAR FFTW3_LIB FFTW3_INCLUDE_DIR)
mark_as_advanced(FFTW3_FOUND)

if (FFTW3_FOUND)
  dune_register_package_flags(LIBRARIES ${FFTW3_LIBRARIES}
    INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
endif()
