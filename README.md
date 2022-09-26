# Welcome to dune-parafields?

`dune-parafields` provides the [Dune framework for the numerical solution of PDEs](https://dune-project.org)
with Gaussian random fields based on circulant embedding, with the following features:

* support for random fields of arbitrary dimensionality
* data redistribution and parallel overlap for 1D (processes),
  2D and 3D realizations of random fields
* exponential, Gaussian, Matérn, spherical and cubic
  covariance functions, among others
* axiparallel and full geometric anisotropy as options
* value transforms like log-normal, folded normal, or
  sign function (excursion set)
* standard vector calculus operations
* multiplication with covariance matrices, their inverse
  and an approximation of their square root
* optional caching of matrix-vector products
* parallelization based on domain decomposition and MPI
* optional support for field input and output based on HDF5
* field output based on VTK (Legacy or XML flavor)

## How to use dune-parafields

dune-parafields is written as a Dune module. You can put it as
a requirement into the `dune.module` file of your own module and
configure/build it through `dunecontrol` (see the documentation
of dune-common for details).

dune-parafields uses the [parafields-core library](https://github.com/parafields/parafields-core)
and requires the same dependencies to be present:

* a C++ compiler supporting C++17
* CMake >= 3.11
* an MPI installation
* FFTW3 compiled with MPI support
* [dune-common](https://gitlab.dune-project.org/core/dune-common) (trivially satisfied for the Dune module dune-parafields)

Optionally, `dune-parafields` can also make use of the following
libraries:

* HDF5 with MPI support for I/O
* GSL for additional covariance functions
* PNG for image I/O
* `dune-grid`
* `dune-nonlinopt`
* `dune-pdelab`

Apart from `dune-common, dune-parafields has no dependencies on
other Dune modules, and can directly be used in other scientific
computing environments via the two options mentioned above.

Basic usage instructions for dune-parafields can be found by running
the standalone *fieldgenerator* application with "-h" or "--help" as
argument.

## History of the project

`dune-parafields` was developed by Ole Klein under the name of [dune-randomfield](https://gitlab.dune-project.org/ole.klein/dune-randomfield).
In 2022, the project was refactored into the core C++ library `parafields-core`
and the Dune module `dune-parafields`. The reason behind this refactoring was
the development of the Dune-agnostic Python package [parafields](https://github.com/parafields/parafields)
that is also based on `parafields-core`.

## Acknowledgments

The work by Ole Klein is supported by the federal ministry of
education and research of Germany (Bundesministerium für
Bildung und Forschung) and the ministry of science, research
and arts of the federal state of Baden-Württemberg (Ministerium
für Wissenschaft, Forschung und Kunst Baden-Württemberg).
