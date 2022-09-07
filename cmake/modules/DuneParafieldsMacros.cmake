
# Add the parafields core library
add_subdirectory(${PROJECT_SOURCE_DIR}/ext/parafields-core)

get_target_property(CORE_COMPILE_DEFINITIONS parafields::parafields INTERFACE_COMPILE_DEFINITIONS)
get_target_property(CORE_LIBRARIES parafields::parafields INTERFACE_LINK_LIBRARIES)
get_target_property(CORE_INCLUDE_DIRECTORIES parafields::parafields INTERFACE_INCLUDE_DIRECTORIES)

dune_register_package_flags(
  COMPILE_DEFINITIONS ${CORE_COMPILE_DEFINITIONS}
  LIBRARIES ${CORE_LIBRARIES}
  INCLUDE_DIRS ${CORE_INCLUDE_DIRECTORIES}
)