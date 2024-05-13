#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/timer.hh>
#include <iostream>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#if HAVE_DUNE_GRID
#include <dune/grid/io/file/vtk.hh>
#include <dune/grid/yaspgrid.hh>
#endif // HAVE_DUNE_GRID
#include <parafields/io.hh>
#include <parafields/randomfield.hh>

/**
 * @brief Types for coordinates and range values
 */
template<typename DF, typename RF, unsigned int dimension>
class GridTraits
{
public:
  enum
  {
    dim = dimension
  };

  using RangeField = RF;
  using Scalar = Dune::FieldVector<RF, 1>;
  using DomainField = DF;
  using Domain = Dune::FieldVector<DF, dim>;
};

#if HAVE_DUNE_GRID
/**
 * @brief Grid helper class for YaspGrid generation (for VTK output)
 */
template<typename GT>
class GridHelper
{
  enum
  {
    dim = GT::dim
  };

  using DF = typename GT::DomainField;

  int levels;
  std::vector<DF> maxExt;
  std::vector<unsigned int> minCells, maxCells;

public:
  GridHelper(const Dune::ParameterTree& config)
  {
    levels = config.get<int>("grid.levels", 1);
    maxExt = config.get<std::vector<DF>>("grid.extensions");
    maxCells = config.get<std::vector<unsigned int>>("grid.cells");

    if (maxExt.size() != maxCells.size())
      throw std::runtime_error{ "cell and extension vectors differ in size" };

    minCells = maxCells;
    for (int i = 0; i < levels - 1; i++)
      for (unsigned int j = 0; j < maxCells.size(); j++) {
        if (minCells[j] % 2 != 0)
          throw std::runtime_error{
            "cannot create enough levels for hierarchical grid, check number "
            "of cells"
          };

        minCells[j] /= 2;
      }
  }

  Dune::FieldVector<DF, dim> L() const
  {
    Dune::FieldVector<DF, dim> Lvector;

    for (unsigned int i = 0; i < dim; i++)
      Lvector[i] = maxExt[i];

    return Lvector;
  }

  std::array<int, dim> N() const
  {
    std::array<int, dim> Nvector;

    for (unsigned int i = 0; i < dim; i++)
      Nvector[i] = minCells[i];

    return Nvector;
  }

  std::bitset<dim> B() const { return std::bitset<dim>(false); }
};
#endif // HAVE_DUNE_GRID

/**
 * @brief Field generation specialized by dimension
 */
template<unsigned int dim>
void
generate(const Dune::ParameterTree& config)
{
  using GridTraits = GridTraits<double, double, dim>;

  const unsigned int seed = config.template get<unsigned int>("input.seed", 0);

  const std::string hdf5Out =
    config.template get<std::string>("output.dune", "");
  const std::string vtkOut = config.template get<std::string>("output.vtk", "");
  const std::string vtkSepOut =
    config.template get<std::string>("output.vtkSeparate", "");
  const std::string legacyVtkOut =
    config.template get<std::string>("output.legacyVtk", "");
  const std::string legacyVtkSepOut =
    config.template get<std::string>("output.legacyVtkSeparate", "");
  if (hdf5Out == "" && vtkOut == "" && vtkSepOut == "" && legacyVtkOut == "" &&
      legacyVtkSepOut == "")
    throw std::runtime_error{
      "no output file given, please specify field (HDF5) or VTK output file "
      "(or both)\nexample: -output.dune field -output.vtk fieldVis"
    };

  parafields::RandomField<GridTraits> field(config);
  if (seed == 0)
    field.generate();
  else
    field.generate(seed);

  unsigned int refine = config.template get<unsigned int>("input.refine", 0);
  for (unsigned int i = 0; i < refine; i++) {
    field.refineMatrix();
    field.refine();
  }

  unsigned int coarsen = config.template get<unsigned int>("input.coarsen", 0);
  for (unsigned int i = 0; i < coarsen; i++) {
    field.coarsenMatrix();
    field.coarsen();
  }

  if (hdf5Out != "")
    field.writeToFile(hdf5Out);
  if (vtkOut != "") {
#if HAVE_DUNE_GRID
    const GridHelper<GridTraits> gh(config);
    Dune::YaspGrid<dim> yaspGrid(gh.L(), gh.N(), gh.B(), 1);
    field.writeToVTK(vtkOut, yaspGrid.leafGridView());
#else  // HAVE_DUNE_GRID
    throw std::runtime_error{
      "unstructured VTK output requires dune-grid and dune-functions"
    };
#endif // HAVE_DUNE_GRID
  }
  if (vtkSepOut != "") {
#if HAVE_DUNE_GRID
    const GridHelper<GridTraits> gh(config);
    Dune::YaspGrid<dim> yaspGrid(gh.L(), gh.N(), gh.B(), 1);
    field.writeToVTKSeparate(vtkSepOut, yaspGrid.leafGridView());
#else  // HAVE_DUNE_GRID
    throw std::runtime_error{
      "unstructured VTK output requires dune-grid and dune-functions"
    };
#endif // HAVE_DUNE_GRID
  }
  if (legacyVtkOut != "")
    field.writeToLegacyVTK(legacyVtkOut + ".vtk");
  if (legacyVtkSepOut != "")
    field.writeToLegacyVTKSeparate(legacyVtkSepOut + ".vtk");
}

/**
 * @brief Field generation specialized by dimension
 */
template<unsigned int dim>
void
generateList(const Dune::ParameterTree& config)
{
  using GridTraits = GridTraits<double, double, dim>;

  const unsigned int seed = config.template get<unsigned int>("input.seed", 0);

  const std::string hdf5Out =
    config.template get<std::string>("output.dune", "");
  const std::string vtkOut = config.template get<std::string>("output.vtk", "");
  const std::string vtkSepOut =
    config.template get<std::string>("output.vtkSeparate", "");
  const std::string legacyVtkOut =
    config.template get<std::string>("output.legacyVtk", "");
  const std::string legacyVtkSepOut =
    config.template get<std::string>("output.legacyVtkSeparate", "");
  if (hdf5Out == "" && vtkOut == "" && vtkSepOut == "" && legacyVtkOut == "" &&
      legacyVtkSepOut == "")
    throw std::runtime_error{
      "no output file given, please specify field (HDF5) or VTK output file "
      "(or both)\nexample: -output.dune field -output.vtk fieldVis"
    };

  parafields::RandomFieldList<GridTraits> field(config);
  if (seed == 0)
    field.generate();
  else
    field.generate(seed);

  unsigned int refine = config.template get<unsigned int>("input.refine", 0);
  for (unsigned int i = 0; i < refine; i++) {
    field.refineMatrix();
    field.refine();
  }

  unsigned int coarsen = config.template get<unsigned int>("input.coarsen", 0);
  for (unsigned int i = 0; i < coarsen; i++) {
    field.coarsenMatrix();
    field.coarsen();
  }

  if (hdf5Out != "")
    field.writeToFile(hdf5Out);
  if (vtkOut != "") {
#if HAVE_DUNE_GRID
    const GridHelper<GridTraits> gh(config);
    Dune::YaspGrid<dim> yaspGrid(gh.L(), gh.N(), gh.B(), 1);
    field.writeToVTK(vtkOut, yaspGrid.leafGridView());
#else  // HAVE_DUNE_GRID
    throw std::runtime_error{
      "unstructured VTK output requires dune-grid and dune-functions"
    };
#endif // HAVE_DUNE_GRID
  }
  if (vtkSepOut != "") {
#if HAVE_DUNE_GRID
    const GridHelper<GridTraits> gh(config);
    Dune::YaspGrid<dim> yaspGrid(gh.L(), gh.N(), gh.B(), 1);
    field.writeToVTKSeparate(vtkSepOut, yaspGrid.leafGridView());
#else  // HAVE_DUNE_GRID
    throw std::runtime_error{
      "unstructured VTK output requires dune-grid and dune-functions"
    };
#endif // HAVE_DUNE_GRID
  }
  if (legacyVtkOut != "")
    field.writeToLegacyVTK(legacyVtkOut);
  if (legacyVtkSepOut != "")
    field.writeToLegacyVTKSeparate(legacyVtkSepOut);
}

/**
 * @brief Generate random field using supplied parameters
 */
void
generateFields(const Dune::MPIHelper& helper,
               const std::string& configFilename,
               int argc,
               char** argv)
{
  Dune::ParameterTree config;
  Dune::ParameterTreeParser parser;
  if (configFilename != "")
    parser.readINITree(configFilename, config);
  parser.readOptions(argc, argv, config);

  std::vector<double> extensions =
    config.get<std::vector<double>>("grid.extensions");
  std::string types = config.get<std::string>("randomField.types", "");

  if (types == "") {
    if (helper.rank() == 0)
      std::cout << "single field mode" << std::endl;

    // no types found, config describes single random field
    if (extensions.size() == 1)
      generate<1>(config);
    else if (extensions.size() == 2)
      generate<2>(config);
    else if (extensions.size() == 3)
      generate<3>(config);
    else if (extensions.size() == 4)
      generate<4>(config);
    else
      throw std::runtime_error{
        "dimension (size of grid.extensions) has to be 1, 2, 3 or 4"
      };
  } else {
    if (helper.rank() == 0)
      std::cout << "field list mode" << std::endl;

    // types declared, config should be for field list
    if (extensions.size() == 1)
      generateList<1>(config);
    else if (extensions.size() == 2)
      generateList<2>(config);
    else if (extensions.size() == 3)
      generateList<3>(config);
    else if (extensions.size() == 4)
      generateList<4>(config);
    else
      throw std::runtime_error{
        "dimension (size of grid.extensions) has to be 1, 2, 3 or 4"
      };
  }
}

/**
 * @brief Print help message for purpose and usage
 */
void
printHelpMessage()
{
  std::cout
    << "This is dune-randomfield, a Gaussian random field generator\n\n"
    << "Usage:\n\n"
    << "fieldgenerator <config file>\n"
    << "    generate field based on provided specifications\n"
    << "    (see below for example files)\n\n"
    << "fieldgenerator\n"
    << "    use default config filename \"randomfield.ini\" if present\n\n"
    << "fieldgenerator <config file> [-<name> <value>]...\n"
    << "    use config file, but add/change option(s) <name> to <value>\n"
    << "    e.g., -grid.cells \"512 512\" -stochastic.variance 5\n\n"
    << "fieldgenerator [-<name> <value>]...\n"
    << "    as above, but using \"randomfield.ini\"\n\n"
    << "fieldgenerator minimal\n"
    << "    print a minimal working example to standard output\n\n"
    << "fieldgenerator full\n"
    << "    print a feature-complete example to standard output\n\n"
    << "fieldgenerator list\n"
    << "    print an example for field lists to standard output\n\n"
    << "fieldgenerator -h | --help | help\n"
    << "    print this help message\n\n"
    << "Script-specific Options:\n\n"
    << "-input.seed <number>\n"
    << "    specify fixed seed for internal random number generator\n\n"
    << "-input.refine <number>\n"
    << "    subdivide cells <number> times before writing native format\n"
    << "    (not honored by VTK output)\n\n"
    << "-input.coarsen <number>\n"
    << "    same as refine, but merge cells instead\n\n"
    << "-output.dune <basename>\n"
    << "    write field in native format (HDF5 + config files)\n\n"
    << "-output.vtk <basename>\n"
    << "    write field in VTK format (XML flavor)\n\n"
    << "-output.vtkSeparate <basename>\n"
    << "    write components of field in VTK (XML flavor)\n\n"
    << "-output.legacyVtk <basename>\n"
    << "    write field in VTK format (legacy ASCII file)\n\n"
    << "-output.legacyVtkSeparate <basename>\n"
    << "    write components of field in VTK (legacy ASCII file)\n\n"
    << std::endl;
}

/**
 * @brief Print minimal working example ini file
 */
void
printMinimalExample()
{
  std::cout << "# minimal example random field config file\n"
            << "\n"
            << "# non-specified options have default values\n"
            << "\n"
            << "# dimension and extent of discretized field\n"
            << "[grid]\n"
            << "# extension per dimension (vector size = dimension)\n"
            << "extensions = 1 1\n"
            << "# number of cells per dimension\n"
            << "cells = 512 512\n"
            << "\n"
            << "# config for stochastic part of field\n"
            << "[stochastic]\n"
            << "# name of covariance structure (variogram)\n"
            << "covariance = exponential\n"
            << "# variance of random field\n"
            << "variance = 1.\n"
            << "# correlation length of random field\n"
            << "corrLength = 0.1" << std::endl;
}

/**
 * @brief Print feature-complete example ini file
 */
void
printFullExample()
{
  std::cout
    << "# full example random field config file\n"
    << "# values are default values unless specified in minimal example\n"
    << "# and demonstration of trend part / anisotropy"
    << "\n"
    << "# dimension and extent of discretized field\n"
    << "[grid]\n"
    << "# extension per dimension (vector size = dimension)\n"
    << "extensions = 1 1\n"
    << "# number of cells per dimension\n"
    << "cells = 512 512\n"
    << "\n"
    << "# general options for random field\n"
    << "[randomField]\n"
    << "# transform applied to Gaussian random field\n"
    << "# possible values:\n"
    << "#     none, logNormal, foldedNormal,\n"
    << "#     sign, boxCox\n"
    << "transform = none\n"
    << "# periodic boundary conditions (1) or not (0)\n"
    << "# sets embedding.factor = 1, i.e., behavior can't be controlled"
       "per\n "
    << "# boundary segment and correlation length must be small enough\n"
    << "periodic = 0\n"
    << "# whether information should be printed (1) or not (0)\n"
    << "verbose = 0\n"
    << "# whether matvecs with inverse covariance matrix are cached\n"
    << "cacheInvMatvec = 1\n"
    << "# whether matvecs with approximate root of inv. cov. matrix are "
       "cached\n"
    << "cacheInvRootMatvec = 0\n"
    << "# Conjugate Gradients iterations for matrix inverse multiplication\n"
    << "cgIterations = 100\n"
    << "\n"
    << "# options for circulant embedding and periodization\n"
    << "[embedding]\n"
    << "# accept approximate results (1) or not (0)\n"
    << "# simply sets negative eigenvalues to zero if they occur\n"
    << "approximate = 0\n"
    << "# threshold for considering eigenvalues as negative\n"
    << "threshold = 1e-14"
    << "# relative size of extended domain (per dimension)\n"
    << "factor = 2\n"
    << "# type of embedding / periodization\n"
    << "# choice of \"classical\" (circulant embedding), or smooth "
       "periodization\n"
    << "# possible values:\n"
    << "#     classical\n"
    << "#     merge\n"
    << "#     fold\n"
    << "#     cofold\n"
    << "periodization = classical\n"
    << "# sigmoid function for merging, resp. smooth max for folding\n"
    << "# smoothstep is better, but requires choice for recursion level\n"
    << "# possible values:\n"
    << "#     smooth\n"
    << "#     smoothstep\n"
    << "sigmoid = smooth\n"
    << "# number of recursions (and therefore smoothness) for smoothstep\n"
    << "# (magic value 99: heuristic for certain covariance functions)\n"
    << "mergeRecursions = 99\n"
    << "foldRecursions = 99\n"
    << "cofoldRecursions = 1\n"
    << "# optimization to find non-negative embedding\n"
    << "# possible values:\n"
    << "#     none\n"
    << "#     coneopt\n"
    << "#     dualopt\n"
    << "optim = none\n"
    << "\n"
    << "# config for stochastic part of field\n"
    << "[stochastic]\n"
    << "# name of covariance structure (variogram)\n"
    << "# either one of the predefined values:"
    << "#     exponential, gammaExponential, separableExponential,\n"
    << "#     matern, matern32, matern52, gaussian, spherical,\n"
    << "#     cauchy, generalizedCauchy, cubic, dampedOscillation,\n"
    << "#     whiteNoise\n"
    << "# or a custom class via \"custom-iso\" or \"custom-aniso\", the "
       "former\n"
    << "# is for isotropic custom covariances, the latter for general ones, "
       "and\n"
    << "# both require passing a custom class as template parameter to "
       "generate\n"
    << "covariance = exponential\n"
    << "# variance of random field\n"
    << "variance = 1.\n"
    << "# choice of anisotropy of variogram\n"
    << "# possible values:\n"
    << "#     none\n"
    << "#     axiparallel\n"
    << "#     geometric\n"
    << "anisotropy = axiparallel\n"
    << "# correlation length of random field\n"
    << "# possible values:\n"
    << "#     \"none\": single value\n"
    << "#     \"axiparallel\": values for each dim\n"
    << "#     \"geometric\": trafo matrix entries (rowwise)\n"
    << "corrLength = 0.1 0.05\n"
    << "\n"
    << "# optional trend components\n"
    << "# (no restriction in number, arguments mandatory if component "
       "present)\n"
    << "\n"
    << "# config for field mean trend component\n"
    << "[mean]\n"
    << "# mean of trend component\n"
    << "mean = 0.5\n"
    << "# variance of trend component\n"
    << "variance = 0.01\n"
    << "# config for field slope trend component (centered with zero mean)\n"
    << "[slope]\n"
    << "# mean of trend component (one per dim)\n"
    << "mean = 0.3 0.7\n"
    << "# variance of trend component (one per dim)\n"
    << "variance = 0.01 0.01\n"
    << "# configs for Gaussian function trend components\n"
    << "[disk0]\n"
    << "# mean of trend component features\n"
    << "# position vector, radius (= \"stdDev\"), height\n"
    << "mean = 0.3 0.6 0.05 5.\n"
    << "# variance of trend component features\n"
    << "variance = 0.01 0.01 1e-3 0.1\n"
    << "[disk1]\n"
    << "# mean of trend component features\n"
    << "# position vector, radius (= \"stdDev\"), height\n"
    << "mean = 0.7 0.2 0.05 4.5\n"
    << "# variance of trend component features\n"
    << "variance = 0.01 0.01 1e-3 0.1\n"
    << "# configs for block zone trend components\n"
    << "[block0]\n"
    << "# mean of trend component features\n"
    << "# center position, extent of block, height\n"
    << "mean = 0.4 0.3 0.1 0.1 2.5\n"
    << "# variance of trend component features\n"
    << "variance = 0. 0. 0. 0. 0.1\n"
    << "[block1]\n"
    << "# mean of trend component features\n"
    << "# center position, extent of block, height\n"
    << "mean = 0.5 0.6 0.2 0.1 -3.\n"
    << "# variance of trend component features\n"
    << "variance = 0.01 0.01 0.01 0.01 0.1\n"
    << std::endl;
}

/**
 * @brief Print example ini file for field lists
 */
void
printListExample()
{
  std::cout << "# example config file for random field lists\n"
            << "\n"
            << "# options for individual fields have to be placed in\n"
            << "# files called <name>.field, where <name> is one of\n"
            << "# the types listed below\n"
            << "\n"
            << "# grid.extensions and grid.cells are default values\n"
            << "# that are replaced if explicitly specified in field\n"
            << "\n"
            << "# dimension and extent of discretized field\n"
            << "[grid]\n"
            << "# extension per dimension (vector size = dimension)\n"
            << "extensions = 1 1\n"
            << "# number of cells per dimension\n"
            << "cells = 512 512\n"
            << "\n"
            << "# general options for random field\n"
            << "[randomField]\n"
            << "# names for fields contained in list\n"
            << "types = permeability conductivity concentration\n"
            << "# number of fields that are mutable (from left in"
               "types\")\n"
            << "# remaining fields stay constant during operations (if"
               "any)\n"
            << "active = 2" << std::endl;
}

int
main(int argc, char** argv)
{
  const Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);

  if (argc != 1) {
    std::string arg(argv[1]);

    if (arg == "minimal") {
      printMinimalExample();
    } else if (arg == "full") {
      printFullExample();
    } else if (arg == "list") {
      printListExample();
    } else if (arg == "-h" || arg == "--help" || arg == "help") {
      printHelpMessage();
    }
    // argument is option
    else if (arg[0] == '-') {
      generateFields(helper, "", argc, argv);
    }
    // argument is filename
    else {
      generateFields(helper, arg, argc, argv);
    }
  } else {
    // no arguments
    // print help message if randomfield.ini is missing
    if (!parafields::fileExists("randomfield.ini"))
      printHelpMessage();
    else
      generateFields(helper, "randomfield.ini", argc, argv);
  }

  return 0;
}
