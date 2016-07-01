# module providing convenience methods for compiling binaries with FFTW3 support
#
# Provides the following functions:
#
# add_dune_fftw3_flags(target1 target2...)
#
# adds FFTW3 flags to the targets for compilation and linking
function(add_dune_fftw3_flags _targets)
  if(FFTW3_FOUND)
    foreach(_target ${_targets})
      target_link_libraries(${_target} ${FFTW3_LIBRARY} -L${CMAKE_PREFIX_PATH}/lib -lfftw3_mpi)
      set_property(TARGET ${_target} APPEND PROPERTY INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIRECTORIES} )
    endforeach(_target ${_targets})
  endif(FFTW3_FOUND)
endfunction(add_dune_fftw3_flags)
