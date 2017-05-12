# module providing convenience methods for compiling binaries with HDF5 support
#
# Provides the following functions:
#
# add_dune_hdf5_flags(target1 target2...)
#
# adds HDF5 flags to the targets for compilation and linking
function(add_dune_hdf5_flags _targets)
  if(HDF5_FOUND)
    if(HDF5_IS_PARALLEL)
      foreach(_target ${_targets})
        target_link_libraries(${_target} ${HDF5_LIBRARIES} ${HDF5_C_LIBRARIES})
      #  set_property(TARGET ${_target} APPEND PROPERTY COMPILE_FLAGS ${HDF5_DEFINITIONS})
        set_property(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS H5_USE_16_API)
        set_property(TARGET ${_target} APPEND PROPERTY INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
      endforeach(_target ${_targets})
    else(HDF5_IS_PARALLEL)
      message(FATAL_ERROR "HDF5 without parallel IO support found!")
    endif(HDF5_IS_PARALLEL)                                                                                         
  endif(HDF5_FOUND)
endfunction(add_dune_hdf5_flags)
