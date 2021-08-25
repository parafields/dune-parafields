#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions

#include<dune/common/fmatrix.hh>
#include<dune/common/fvector.hh>
#include<dune/randomfield/randomfield.hh>

/**
 * @brief Types for coordinates and range values
 */
template<typename DF, typename RF, unsigned int dimension>
class GridTraits
{
  public:

    enum {dim = dimension};

    using RangeField  = RF;
    using Scalar      = Dune::FieldVector<RF,1>;
    using DomainField = DF;
    using Domain      = Dune::FieldVector<DF,dim>;
};

/**
 * @brief Test effect of (I - M^1/2 M^-1 M^1/2)
 */
  template<typename RandomField>
void diffRootInvRoot(const RandomField& field)
{
  const double fieldNorm = std::sqrt(field * field);
  RandomField copy(field);
  copy.timesMatrixRoot();
  copy.timesInverseMatrix();
  copy.timesMatrixRoot();
  copy -= field;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "root inv root";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M^-1 M^1/2 M^1/2)
 */
  template<typename RandomField>
void diffRootRootInv(const RandomField& field)
{
  const double fieldNorm = std::sqrt(field * field);
  RandomField copy(field);
  copy.timesMatrixRoot();
  copy.timesMatrixRoot();
  copy.timesInverseMatrix();
  copy -= field;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "root root inv";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M^1/2 M^1/2 M^-1)
 */
  template<typename RandomField>
void diffInvRootRoot(const RandomField& field)
{
  const double fieldNorm = std::sqrt(field * field);
  RandomField copy(field);
  copy.timesInverseMatrix();
  copy.timesMatrixRoot();
  copy.timesMatrixRoot();
  copy -= field;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "inv root root";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M M^-1)
 */
  template<typename RandomField>
void diffInvMult(const RandomField& field)
{
  const double fieldNorm = std::sqrt(field * field);
  RandomField copy(field);
  copy.timesInverseMatrix();
  copy.timesMatrix();
  copy -= field;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "inv mult     ";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M^-1 M)
 */
  template<typename RandomField>
void diffMultInv(const RandomField& field)
{
  const double fieldNorm = std::sqrt(field * field);
  RandomField copy(field);
  copy.timesMatrix();
  copy.timesInverseMatrix();
  copy -= field;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "mult inv     ";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Run different matrix multiplication tests
 */
  template<typename GridTraits>
void runTests(Dune::ParameterTree config, std::string covariance)
{
  config["stochastic.covariance"] = covariance;
  Dune::RandomField::RandomField<GridTraits> randomField(config);
  randomField.generate();

  diffRootInvRoot(randomField);
  diffRootRootInv(randomField);
  diffInvRootRoot(randomField);
  diffInvMult(randomField);
  diffMultInv(randomField);
}

  template<unsigned int dim>
void test()
{
  Dune::ParameterTree config;
  Dune::ParameterTreeParser parser;
  parser.readINITree("randomfield"+std::to_string(dim)+"d.ini",config);

  using GridTraits = GridTraits<double,double,dim>;

  std::cout << "--------------" << std::endl;
  std::cout << dim << "D Exponential" << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits>(config,"exponential");
  std::cout << "--------------" << std::endl;
  std::cout << dim << "D Gaussian   " << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits>(config,"gaussian");
  if (dim < 4)
  {
    std::cout << "--------------" << std::endl;
    std::cout << dim << "D Spherical  " << std::endl;
    std::cout << "--------------" << std::endl;
    runTests<GridTraits>(config,"spherical");
  }
}

int main(int argc, char** argv)
{
  try{
    Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
    std::cout << "Hello World! This is dune-randomfield." << std::endl;
    if(Dune::MPIHelper::isFake)
      std::cout<< "This is a sequential program." << std::endl;
    else
      std::cout<<"I am rank "<<helper.rank()<<" of "<<helper.size()
        <<" processes!"<<std::endl;

    static_assert(DIMENSION >= 1 && DIMENSION <= 4, "only dimension 1, 2, 3 and 4 supported!");

    test<DIMENSION>();

    return 0;
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
    return 1;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
    return 1;
  }
}
