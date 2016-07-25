# What is dune-randomfield?

dune-randomfield provides Gaussian random fields based on
circulant embedding, with the following features:
- exponential, Gaussian and spherical covariance functions
- standard vector calculus operations
- multiplication with covariance matrices, their inverse
  and an approximation of their square root
- optional caching of matrix-vector products
- parallelization based on domain decomposition and MPI
- field input and output based on HDF5

# How to use dune-randomfield

dune-randomfield is written as a Dune module. You can put it as
a requirement into the dune.module file of your own module and
configure/build it through dunecontrol (see the documentation
of dune-common for details).

dune-randomfield requires the external library FFTW3 with MPI
support for parallelized circulant embedding, and the external
library HDF5 with MPI support for parallel file I/O. The tested
versions are:
- fftw-3.3.4
- hdf5-1.8.14

Apart from dune-common, dune-randomfield has no dependencies on
other Dune modules, and can directly be used in other scientific
computing environments.

See the automated tests in the /test subdirectory for basic
usage.

# Where to get help

To get help concerning dune-randomfield, first check the
examples in the /test subfolder. These tests can be built by
issuing *"make build_tests"* in the main folder, and run with
*"make test"* or by calling the executables directly.

If your problem persists, check the bug tracker at

https://gitlab.dune-project.org/oklein/dune-randomfield/issues

or contact the author directly:
* Ole Klein (ole.klein@iwr.uni-heidelberg.de)

# Acknowledgments

The work by Ole Klein is supported by the federal ministry of
education and research of Germany (Bundesministerium für
Bildung und Forschung) and the ministry of science, research
and arts of the federal state of Baden-Württemberg (Ministerium
für Wissenschaft, Forschung und Kunst Baden-Württemberg).
