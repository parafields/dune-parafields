NOTE: ** This is the next generation repository for dune-randomfield
and is currently under construction. For the existing, fully functioning
version of dune-randomfield, see [the Gitlab repository](https://gitlab.dune-project.org/ole.klein/dune-randomfield)**

# What is dune-parafields?

dune-parafields provides Gaussian random fields based on
circulant embedding, with the following features:
- support for random fields of arbitrary dimensionality
- data redistribution and parallel overlap for 1D (processes),
  2D and 3D realizations of random fields
- exponential, Gaussian, Matérn, spherical and cubic
  covariance functions, among others
- axiparallel and full geometric anisotropy as options
- value transforms like log-normal, folded normal, or
  sign function (excursion set)
- standard vector calculus operations
- multiplication with covariance matrices, their inverse
  and an approximation of their square root
- optional caching of matrix-vector products
- parallelization based on domain decomposition and MPI
- optional support for field input and output based on HDF5
- field output based on VTK (Legacy or XML flavor)

# How to use dune-parafields

dune-parafields is written as a Dune module. You can put it as
a requirement into the *dune.module* file of your own module and
configure/build it through dunecontrol (see the documentation
of dune-common for details). dune-parafields can also be used
as a header-only library (just include its paths and those of
dune-common) or a standalone field generator (using the binary in
the /src subfolder).

dune-parafields requires dune-common for configuration and the
external library FFTW3 with MPI support for parallelized circulant
embedding, and can use the external library HDF5 with MPI support
for parallel file I/O if it is found. The tested versions are:
- dune-common-2.5
- fftw-3.3.4
- hdf5-1.8.18

Apart from dune-common, dune-parafields has no dependencies on
other Dune modules, and can directly be used in other scientific
computing environments via the two options mentioned above.

dune-parafields is built by putting its code and that of
dune-common into two subdirectories of an arbitrary folder, e.g.
*$HOME/dune*, and then executing
"dune-common/bin/dunecontrol --opts=\<optsFile\> all",
where \<optsFile\> is an option file like this:

```bash
# subdirectory to use as build-directory
BUILDDIR="$HOME/dune/releaseBuild"
# paths to external software in non-default locations
CMAKE_PREFIX_PATH="$HOME/software"
# options that control compiler verbosity
GXX_WARNING_OPTS="-Wall -pedantic"
# options that control compiler behavior
GXX_OPTS="-march=native -g -O3 -std=c++14"
```

Basic usage instructions for dune-parafields can be found by running
the standalone *fieldgenerator* application with "-h" or "--help" as
argument or by inspecting the automated tests in the /test subdirectory.

# Where to get help

To get help concerning dune-parafields, first check the
implementation of the *fieldgenerator* binary, which provides an
example how dune-parafields could be incorporated into your own
code.

There are also examples in the /test subfolder. These tests can
be built by issuing *"make build_tests"* in the main folder, and
run with *"make test"* or by calling the executables directly.

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
