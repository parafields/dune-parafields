#ifndef DUNE_RANDOMFIELD_LEGACYVTK
#define DUNE_RANDOMFIELD_LEGACYVTK

template<typename Traits>
class LegacyVTKWriter
{
    mutable std::ofstream file;
    mutable bool preambleWritten;

    std::vector<typename Traits::RF> extensions;
    std::vector<unsigned int>        cells;
    std::vector<typename Traits::RF> spacing;
    std::vector<typename Traits::RF> origin;
    unsigned int dataPoints;

  public:

    LegacyVTKWriter(const Dune::ParameterTree& config, const std::string& fileName, const MPI_Comm& comm = MPI_COMM_WORLD)
      : file(fileName+".vtk",std::ofstream::trunc), preambleWritten(false),
      extensions(config.template get<std::vector<typename Traits::RF> >("grid.extensions")),
      cells     (config.template get<std::vector<unsigned int>        >("grid.cells"))
    {
      //DUNE_THROW(Dune::Exception,"prevent parallel runs");
      // VTK is always 3D
      if (extensions.size() < 2)
        extensions.push_back(0.);
      if (extensions.size() < 3)
        extensions.push_back(0.);
      
      if (cells.size() < 2)
        cells.push_back(1);
      if (cells.size() < 3)
        cells.push_back(1);

      for (unsigned int i = 0; i < cells.size(); i++)
      {
        spacing.push_back(extensions[i]/cells[i]);
        origin.push_back(-0.5 * extensions[i]/cells[i]);
        //origin.push_back(0.);
      }

      dataPoints = 1;
      for (unsigned int i = 0; i < cells.size(); i++)
        dataPoints *= cells[i];
    }

    template<typename Field>
      void writeScalarData(const std::string& dataName, const Field& field) const
      {
        if (!preambleWritten)
          writePreamble();

          file << "SCALARS " << dataName << " float 1\n"
            << "LOOKUP_TABLE default\n";

          typename Traits::DomainType domain;
          for (unsigned int i = 0; i < Traits::dim; i++)
            domain[i] = origin[i];
          typename Traits::RangeType value;
          for (unsigned int i = 0; i < cells[2]; i++)
          {
            if (Traits::dim == 3)
              domain[2] += spacing[2];
            if (Traits::dim >= 2)
              domain[1] = origin[1];
            
            for (unsigned int j = 0; j < cells[1]; j++)
            {
              if (Traits::dim >= 2)
                domain[1] += spacing[1];
              domain[0] = origin[0];

              for (unsigned int k = 0; k < cells[0]; k++)
              {
                domain[0] += spacing[0];
                field.evaluate(domain,value);
                file << value[0] << "\n";
              }
            }
          }
      }

  private:

    void writePreamble() const
    {
      file << "# vtk DataFile Version 2.0\n"
        << "dune-randomfield VTK output\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << cells[0] << " " << cells[1] << " " << cells[2] << "\n"
        << "SPACING " << spacing[0] << " " << spacing[1] << " " << spacing[2] << "\n"
        << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << "\n\n"
        << "POINT_DATA " << dataPoints << "\n"
        << std::endl;

      preambleWritten = true;
    }

};

#endif // DUNE_RANDOMFIELD_LEGACYVTK
