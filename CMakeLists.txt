cmake_minimum_required(VERSION 3.11)
project(dune-parafields CXX)

if(NOT (dune-common_DIR OR dune-common_ROOT OR
      "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
    string(REPLACE  ${CMAKE_PROJECT_NAME} dune-common dune-common_DIR
      ${PROJECT_BINARY_DIR})
endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
  ${dune-common_MODULE_PATH})

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()
dune_enable_all_packages()

# Ensure that this repository has been checked out recursively
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/ext/parafields-core/CMakeLists.txt)
  message(FATAL_ERROR "dune-parafields needs to be cloned with --recursive")
endif()

# Add the parafields core library
add_subdirectory(ext/parafields-core)

get_target_property(CORE_COMPILE_DEFINITIONS parafields::parafields INTERFACE_COMPILE_DEFINITIONS)
get_target_property(CORE_LIBRARIES parafields::parafields INTERFACE_LINK_LIBRARIES)
get_target_property(CORE_INCLUDE_DIRECTORIES parafields::parafields INTERFACE_INCLUDE_DIRECTORIES)

dune_register_package_flags(
  COMPILE_DEFINITIONS ${CORE_COMPILE_DEFINITIONS}
  LIBRARIES ${CORE_LIBRARIES}
  INCLUDE_DIRS ${CORE_INCLUDE_DIRECTORIES}
)

add_subdirectory(dune)

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
