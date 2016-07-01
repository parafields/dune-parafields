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

template<typename DF, typename RF, unsigned int dimension>
class GridTraits
{
  public:

    enum {dim = dimension};

    //typedef Dune::YaspGrid<dim> Grid;
    //typedef typename Grid::LevelGridView GridView;

    typedef RF                              RangeField;
    typedef Dune::FieldVector<RF,1>         Scalar;
    typedef Dune::FieldVector<RF,dim>       Vector;
    typedef Dune::FieldMatrix<RF,dim,dim>   Tensor;
    typedef DF                              DomainField;
    typedef Dune::FieldVector<DF,dim>       Domain;
    typedef Dune::FieldVector<DF,dim-1>     IntersectionDomain;
    //typedef typename GridView::Traits::template Codim<0>::Entity Element;
    //typedef typename GridView::Intersection Intersection;

};

int main(int argc, char** argv)
{
  try{
    // Maybe initialize MPI
    Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);
    std::cout << "Hello World! This is dune-randomfield." << std::endl;
    if(Dune::MPIHelper::isFake)
      std::cout<< "This is a sequential program." << std::endl;
    else
      std::cout<<"I am rank "<<helper.rank()<<" of "<<helper.size()
        <<" processes!"<<std::endl;

    Dune::ParameterTree config;
    Dune::ParameterTreeParser parser;
    parser.readINITree("randomfield.ini",config);

    typedef GridTraits<double,double,2> GridTraits;
    typedef Dune::RandomField::ExponentialCovariance Covariance;
    
    /*
    GridTraits::Domain center;
    center[0] = 0.5;
    center[1] = 0.5;
    randomFieldList.localize(center,0.15);
    randomFieldList.writeToFile("sample");
    Dune::RandomField::RandomFieldList<GridTraits,Covariance> copy(randomFieldList);
    Dune::RandomField::RandomFieldList<GridTraits,Covariance> copy2(randomFieldList);
    Dune::RandomField::RandomFieldList<GridTraits,Covariance> diff(randomFieldList);
    copy.timesMatrixRoot();
    copy.writeToFile("timesRoot");
    copy.timesMatrixRoot();
    copy.writeToFile("timesRootSquared");
    diff = copy;
    copy.timesInverseMatrix();
    copy.writeToFile("rootRestored");
    copy2.timesMatrix();
    copy2.writeToFile("timesMatrix");
    diff -= copy2;
    diff.writeToFile("matrixDiff");
    copy2.timesInverseMatrix();
    copy2.writeToFile("matrixRestored");
    randomFieldList -= copy;
    randomFieldList.writeToFile("difference");
    */

    std::cout << "------------------------------" << std::endl;
    std::cout << "root inv root" << std::endl;
    std::cout << "------------------------------" << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> randomFieldList(config);
      randomFieldList.generateUncorrelated();
      const double fieldNorm = std::sqrt(randomFieldList * randomFieldList);
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> copy(randomFieldList);
      copy.timesMatrixRoot();
      copy.timesInverseMatrix();
      copy.timesMatrixRoot();
      randomFieldList -= copy;
      const double diffNorm = std::sqrt(randomFieldList * randomFieldList);
      std::cout << "norm: " << fieldNorm << " diffNorm: " << diffNorm << " opNorm: " << diffNorm/fieldNorm << std::endl;
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "root root inv" << std::endl;
    std::cout << "------------------------------" << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> randomFieldList(config);
      randomFieldList.generateUncorrelated();
      const double fieldNorm = std::sqrt(randomFieldList * randomFieldList);
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> copy(randomFieldList);
      copy.timesMatrixRoot();
      copy.timesMatrixRoot();
      copy.timesInverseMatrix();
      randomFieldList -= copy;
      const double diffNorm = std::sqrt(randomFieldList * randomFieldList);
      std::cout << "norm: " << fieldNorm << " diffNorm: " << diffNorm << " opNorm: " << diffNorm/fieldNorm << std::endl;
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "root inv root local" << std::endl;
    std::cout << "------------------------------" << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> randomFieldList(config);
      randomFieldList.generateUncorrelated();
      GridTraits::Domain center;
      center[0] = 0.5;
      center[1] = 0.5;
      randomFieldList.localize(center,0.15);
      const double fieldNorm = std::sqrt(randomFieldList * randomFieldList);
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> copy(randomFieldList);
      copy.timesMatrixRoot();
      copy.timesInverseMatrix();
      copy.timesMatrixRoot();
      randomFieldList -= copy;
      const double diffNorm = std::sqrt(randomFieldList * randomFieldList);
      std::cout << "norm: " << fieldNorm << " diffNorm: " << diffNorm << " opNorm: " << diffNorm/fieldNorm << std::endl;
    }

    std::cout << "------------------------------" << std::endl;
    std::cout << "root root inv local" << std::endl;
    std::cout << "------------------------------" << std::endl;
    for (unsigned int i = 0; i < 10; i++)
    {
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> randomFieldList(config);
      randomFieldList.generateUncorrelated();
      GridTraits::Domain center;
      center[0] = 0.5;
      center[1] = 0.5;
      randomFieldList.localize(center,0.15);
      const double fieldNorm = std::sqrt(randomFieldList * randomFieldList);
      Dune::RandomField::RandomFieldList<GridTraits,Covariance> copy(randomFieldList);
      copy.timesMatrixRoot();
      copy.timesMatrixRoot();
      copy.timesInverseMatrix();
      randomFieldList -= copy;
      const double diffNorm = std::sqrt(randomFieldList * randomFieldList);
      std::cout << "norm: " << fieldNorm << " diffNorm: " << diffNorm << " opNorm: " << diffNorm/fieldNorm << std::endl;
    }

    return 0;
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
