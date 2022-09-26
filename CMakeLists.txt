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

# Ensure that this repository has been checked out recursively
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/ext/parafields-core/CMakeLists.txt)
  message(FATAL_ERROR "dune-parafields needs to be cloned with --recursive")
endif()

# Add the parafields core library
add_subdirectory(${PROJECT_SOURCE_DIR}/ext/parafields-core)

# start a dune project with information from dune.module
dune_project()
dune_enable_all_packages()

add_subdirectory(cmake/modules)
add_subdirectory(dune)
add_subdirectory(src)

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
