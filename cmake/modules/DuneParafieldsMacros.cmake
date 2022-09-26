if(NOT "${PROJECT_NAME}" STREQUAL "dune-parafields")
  set(CMAKE_MODULE_PATH ${dune-parafields_DIR}/ext/parafields-core/cmake ${CMAKE_MODULE_PATH})
  set(parafields_ROOT ${dune-parafields_DIR}/ext/parafields-core)
  find_package(parafields CONFIG)

  get_target_property(CORE_COMPILE_DEFINITIONS parafields::parafields INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(CORE_LIBRARIES parafields::parafields INTERFACE_LINK_LIBRARIES)
  get_target_property(CORE_INCLUDE_DIRECTORIES parafields::parafields INTERFACE_INCLUDE_DIRECTORIES)

  dune_register_package_flags(
    COMPILE_DEFINITIONS ${CORE_COMPILE_DEFINITIONS}
    LIBRARIES ${CORE_LIBRARIES}
    INCLUDE_DIRS ${CORE_INCLUDE_DIRECTORIES}
  )
endif()
