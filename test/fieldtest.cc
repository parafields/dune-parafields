// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

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

    typedef RF                        RangeField;
    typedef Dune::FieldVector<RF,1>   Scalar;
    typedef DF                        DomainField;
    typedef Dune::FieldVector<DF,dim> Domain;
};

/**
 * @brief Test effect of (I - M^1/2 M^-1 M^1/2)
 */
template<typename RandomFieldList>
void diffRootInvRoot(const RandomFieldList& fieldList)
{
  const double fieldNorm = std::sqrt(fieldList * fieldList);
  RandomFieldList copy(fieldList);
  copy.timesMatrixRoot();
  copy.timesInverseMatrix();
  copy.timesMatrixRoot();
  copy -= fieldList;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "root inv root";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M^-1 M^1/2 M^1/2)
 */
template<typename RandomFieldList>
void diffRootRootInv(const RandomFieldList& fieldList)
{
  const double fieldNorm = std::sqrt(fieldList * fieldList);
  RandomFieldList copy(fieldList);
  copy.timesMatrixRoot();
  copy.timesMatrixRoot();
  copy.timesInverseMatrix();
  copy -= fieldList;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "root root inv";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M^1/2 M^1/2 M^-1)
 */
template<typename RandomFieldList>
void diffInvRootRoot(const RandomFieldList& fieldList)
{
  const double fieldNorm = std::sqrt(fieldList * fieldList);
  RandomFieldList copy(fieldList);
  copy.timesInverseMatrix();
  copy.timesMatrixRoot();
  copy.timesMatrixRoot();
  copy -= fieldList;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "inv root root";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M M^-1)
 */
template<typename RandomFieldList>
void diffInvMult(const RandomFieldList& fieldList)
{
  const double fieldNorm = std::sqrt(fieldList * fieldList);
  RandomFieldList copy(fieldList);
  copy.timesInverseMatrix();
  copy.timesMatrix();
  copy -= fieldList;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "inv mult     ";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Test effect of (I - M^-1 M)
 */
template<typename RandomFieldList>
void diffMultInv(const RandomFieldList& fieldList)
{
  const double fieldNorm = std::sqrt(fieldList * fieldList);
  RandomFieldList copy(fieldList);
  copy.timesMatrix();
  copy.timesInverseMatrix();
  copy -= fieldList;
  const double diffNorm = std::sqrt(copy * copy);

  std::cout << "mult inv     ";
  std::cout << " norm: " << fieldNorm;
  std::cout << " diffNorm: " << diffNorm;
  std::cout << " opNorm: " << diffNorm/fieldNorm << std::endl;
}

/**
 * @brief Run different matrix multiplication tests
 */
template<typename GridTraits, typename Covariance, bool storeInvMat, bool storeInvRoot>
void runTests(const Dune::ParameterTree& config)
{
  Dune::RandomField::RandomFieldList<GridTraits,Covariance,storeInvMat,storeInvRoot> randomFieldList(config);
  randomFieldList.generate();

  diffRootInvRoot(randomFieldList);
  diffRootRootInv(randomFieldList);
  diffInvRootRoot(randomFieldList);
  diffInvMult(randomFieldList);
  diffMultInv(randomFieldList);
}

/**
 * @brief 2D version of tests
 */
void test2d()
{
  Dune::ParameterTree config;
  Dune::ParameterTreeParser parser;
  parser.readINITree("randomfield2d.ini",config);

  typedef GridTraits<double,double,2> GridTraits;

  std::cout << "--------------" << std::endl;
  std::cout << "2D Exponential" << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits,Dune::RandomField::ExponentialCovariance,INVMAT,INVROOT>(config);
  std::cout << "--------------" << std::endl;
  std::cout << "2D Gaussian   " << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits,Dune::RandomField::GaussianCovariance,   INVMAT,INVROOT>(config);
  std::cout << "--------------" << std::endl;
  std::cout << "2D Spherical  " << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits,Dune::RandomField::SphericalCovariance,  INVMAT,INVROOT>(config);
}

/**
 * @brief 3D version of tests
 */
void test3d()
{
  Dune::ParameterTree config;
  Dune::ParameterTreeParser parser;
  parser.readINITree("randomfield3d.ini",config);

  typedef GridTraits<double,double,3> GridTraits;

  std::cout << "--------------" << std::endl;
  std::cout << "3D Exponential" << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits,Dune::RandomField::ExponentialCovariance,INVMAT,INVROOT>(config);
  std::cout << "--------------" << std::endl;
  std::cout << "3D Gaussian   " << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits,Dune::RandomField::GaussianCovariance,   INVMAT,INVROOT>(config);
  std::cout << "--------------" << std::endl;
  std::cout << "3D Spherical  " << std::endl;
  std::cout << "--------------" << std::endl;
  runTests<GridTraits,Dune::RandomField::SphericalCovariance,  INVMAT,INVROOT>(config);
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

    if (DIMENSION == 2)
      test2d();
    else if (DIMENSION == 3)
      test3d();
    else
      DUNE_THROW(Dune::Exception,"only dimension 2 and 3 supported!");

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
