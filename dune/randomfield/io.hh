#ifndef DUNE_RANDOMFIELD_IO_HH
#define DUNE_RANDOMFIELD_IO_HH

#include<fstream>

#if HAVE_HDF5
#include<hdf5.h>

/// @todo get rid of this
#define HDF5_DATA_TYPE H5T_IEEE_F64LE  //define for 64 bit machine
//#define HDF5_DATA_TYPE H5T_IEEE_F32LE //define for 32 bit machine
#endif // HAVE_HDF5

namespace Dune{
  namespace RandomField{

    /**
     * @brief Check if file exists
     */
    bool fileExists(std::string filename)
    {
      std::ifstream testfile(filename);
      return testfile.good();
    }

#if HAVE_HDF5 // only define IO methods if HDF5 present
    // HDF5-internal data type specifiers
    template<typename> hid_t hdf5Type;
    template<> const hid_t hdf5Type<float>       = H5T_NATIVE_FLOAT;
    template<> const hid_t hdf5Type<double>      = H5T_NATIVE_DOUBLE;
    template<> const hid_t hdf5Type<long double> = H5T_NATIVE_LDOUBLE;

    /// @todo take care of missing files
    /**
     * @brief Read data from an HDF5 file (parallel)
     */
    template<typename RF, unsigned int dim>
      void readParallelFromHDF5(
          std::vector<RF>& local_data,
          const std::array<unsigned int,dim>& local_count,
          const std::array<unsigned int,dim>& local_offset,
          const MPI_Comm& communicator,
          const std::string& data_name,
          const std::string& data_filename
          )
      {
        // setup file access template with parallel IO access
        hid_t access_pList = H5Pcreate(H5P_FILE_ACCESS);
        assert(access_pList > -1);

        herr_t status;
        MPI_Info mpiInfo = MPI_INFO_NULL;
        status = H5Pset_fapl_mpio(access_pList,communicator,mpiInfo);
        assert(status > -1);

        // open the file for reading
        hid_t file_id = H5Fopen(data_filename.c_str(),H5F_ACC_RDONLY,access_pList);
        assert(file_id > -1);

        // Release file-access template
        status = H5Pclose(access_pList);
        assert(status > -1);

        // open the dataset
        hid_t dataset_id = H5Dopen(file_id,data_name.c_str(),H5P_DEFAULT);
        assert(dataset_id > -1);

        // get the dataspace
        hid_t dataspace_id = H5Dget_space(dataset_id);
        assert(dataspace_id > -1);

        // some needed variables
        hsize_t dimData;
        hsize_t* dims;

        // get the dimension
        dimData = H5Sget_simple_extent_ndims(dataspace_id);
        assert(dimData == dim);

        // get the size of the data structure
        dims = (hsize_t*)malloc(dim * sizeof (hsize_t));
        status = H5Sget_simple_extent_dims(dataspace_id,dims,0);
        assert(status > -1);

        //set the local, offset, and count as hsize_t, which is needed by the HDF5 routines
        hsize_t local_size = 1;
        hsize_t offset[dim],count[dim];
        for(unsigned int i=0; i < dim; i++ )
        {
          local_size      *=  local_count [i];
          offset[dim-i-1]  =  local_offset[i];
          count [dim-i-1]  =  local_count [i];
        }

        // create the memory space, if something needs to be read on this processor
        hid_t memspace_id = 0;
        if(local_size != 0)
          memspace_id = H5Screate_simple(1,&local_size,NULL);

        //select the hyperslab
        status = H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,NULL,count,NULL);
        assert(status > -1);

        //resize the return data
        local_data.resize(local_size);

        // set up the collective transfer properties list
        hid_t xferPropList = H5Pcreate(H5P_DATASET_XFER);
        assert(xferPropList > -1);

        // finally the reading from the file, only if something needes to be read
        if(local_size != 0)
        {
          status = H5Dread(dataset_id,
              hdf5Type<RF>,
              memspace_id,
              dataspace_id,
              xferPropList,
              &(local_data[0])
              );
          assert(status > -1);
        }

        // close the identifiers
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        if(local_size != 0) //this identifier only exists if somethings needs to be read
          H5Sclose(memspace_id);
        free(dims);
        status = H5Fclose( file_id );
        assert( status > -1 );
      }

    /// @todo take care of missing files
    /**
     * @brief Write data to an HDF5 file (parallel)
     */
    template<typename RF, unsigned int dim>
      void writeParallelToHDF5(
          const std::array<unsigned int,dim>& global_dim,
          const std::vector<RF>& data,
          const std::array<unsigned int,dim>& local_count,
          const std::array<unsigned int,dim>& local_offset,
          const MPI_Comm& communicator,
          const std::string& data_name,
          const std::string& data_filename
          )
      {
        //Info variable needed for HDF5
        MPI_Info mpiInfo = MPI_INFO_NULL;
        herr_t status;

        // Set up file access property list with parallel I/O access
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id,communicator,mpiInfo);
        assert(plist_id > -1);

        // Create a new file using default properties.
        hid_t file_id = H5Fcreate(data_filename.c_str(),
            H5F_ACC_TRUNC,
            H5P_DEFAULT,
            plist_id
            );
        assert(file_id > -1);
        H5Pclose(plist_id);

        // set the global size of the grid into a vector of type hsize_t (needed for HDF5 routines)
        hsize_t global_dim_HDF5[dim];
        for(unsigned int i = 0; i < dim; i++)
          global_dim_HDF5[dim-i-1] = global_dim[i];

        // set the count and offset in the different dimensions (determine the size of the hyperslab)
        // (in hsize_t format, needed for HDF5 routines)
        hsize_t count[dim], offset[dim];
        for(unsigned int i = 0; i < dim; i++)
        {
          count[dim-i-1]  = local_count [i];
          offset[dim-i-1] = local_offset[i];
        }

        // define the total size of the local data
        hsize_t nAllLocalCells = 1;
        for (unsigned int i = 0; i < dim; i++)
          nAllLocalCells *= count[i];

        // Create the dataspace for the dataset.
        hid_t filespace = H5Screate_simple(dim,global_dim_HDF5,NULL);
        assert(filespace > -1);

        // Create the dataset with default properties and close filespace.
        hid_t dset_id = H5Dcreate(file_id,data_name.c_str(),HDF5_DATA_TYPE,
            filespace,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
        H5Sclose(filespace);
        assert(dset_id > -1);

        //get the memoryspace (but only if something needs to be written on this processor!)
        hid_t memspace_id;
        if(nAllLocalCells != 0)
        {  // -> otherwise HDF5 warning, because of writing nothing!
          memspace_id = H5Screate_simple(dim,count,NULL);
          assert(memspace_id > -1);
        }

        // Select hyperslab in the file
        filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace,H5S_SELECT_SET,offset,NULL,count,NULL);

        // Create property list for collective dataset write.
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id,H5FD_MPIO_COLLECTIVE);

        // finally write the data to the disk
        // even if nothing should be written H5Dwrite needs to be called!
        if(nAllLocalCells != 0)
        { // -> otherwise HDF5 warning, because of writing nothing!
          status = H5Dwrite(dset_id,hdf5Type<RF>,memspace_id,filespace,plist_id,&(data[0]));
          assert(status > -1);
        }
        else
        { // IMPORTANT. otherwise the H5Dwrite() blocks!
          status = H5Dwrite(dset_id,hdf5Type<RF>,0,filespace,plist_id,&(data[0]));
          assert(status > -1);
        }

        // Close the property list
        status = H5Pclose(plist_id);
        assert(status > -1);

        // Close the filespace
        status = H5Sclose(filespace);
        assert(status > -1);

        //if something written close the memspace
        if(nAllLocalCells != 0)
        {
          status = H5Sclose(memspace_id);
          assert(status > -1);
        }

        // Close the dataset
        status = H5Dclose(dset_id);
        assert(status > -1);

        // Close the file
        status = H5Fclose(file_id);
        assert(status > -1);

        //propably not needed. because the H5Dwrite blocks anyway
        MPI_Barrier(communicator);
      }
#endif //HAVE_HDF5

    template<typename RF, unsigned int dim>
      bool writeToXDMF(
          const std::array<unsigned int,dim>& cells,
          const std::array<RF,dim>& extensions,
          const std::string& fileName
          )
      {
        if (dim > 4)
          return false;

        std::ofstream file(fileName+".xdmf",std::ofstream::trunc);

        file
          << "<?xml version=\"1.0\" ?>\n"
          << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
          << "<Xdmf Version=\"2.0\">\n"
          << " <Domain>\n";

        if (dim < 4)
        {
          file
            << "  <Grid Name=\"StructuredGrid\">\n"
            << "   <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"";
          for (unsigned int i = 0; i < dim; i++)
            file << cells[dim-(i+1)] << " ";
          file
            << "\"/>\n"
            << "   <Geometry GeometryType=\"origin_dxdydz\">\n"
            << "    <DataItem Dimensions=\"3\">\n"
            << "     0. 0. 0.\n"
            << "    </DataItem>\n"
            << "    <DataItem Dimensions=\"3\">\n"
            << "     ";
          for (unsigned int i = 0; i < 3-dim; i++)
            file << extensions[0]/cells[0] << " "; // additional entries to visualize 1D and 2D files
          for (unsigned int i = 0; i < dim; i++)
            file << extensions[i]/cells[i] << " ";
          file
            << "\n"
            << "    </DataItem>\n"
            << "   </Geometry>\n"
            << "   <Attribute Name=\""
            << fileName
            << "\" Center=\"Cell\">\n"
            << "    <DataItem Dimensions=\"";
          for (unsigned int i = 0; i < dim; i++)
            file << cells[dim-(i+1)] << " ";
          file
            << "\" Format=\"HDF\">\n"
            << "     " << fileName+".stoch.h5" << ":/stochastic\n"
            << "    </DataItem>\n"
            << "   </Attribute>\n"
            << "  </Grid>\n";
        }
        else if (dim == 4)
        {
          file
            << "  <Grid Name=\"GridTime\" GridType=\"Collection\""
            << " CollectionType=\"Temporal\">\n";
          for (unsigned int j = 0; j < cells[dim-1]; j++)
          {
            file
              << "   <Grid Name=\"StructuredGrid\">\n"
              << "    <Time Value=\"" << j << "\"/>\n"
              << "    <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"";
            for (unsigned int i = 0; i < dim-1; i++)
              file << cells[dim-(i+2)] << " ";
            file
              << "\"/>\n"
              << "    <Geometry GeometryType=\"origin_dxdydz\">\n"
              << "     <DataItem Dimensions=\"3\">\n"
              << "      0. 0. 0.\n"
              << "     </DataItem>\n"
              << "     <DataItem Dimensions=\"3\">\n"
              << "      ";
            for (unsigned int i = 0; i < dim-1; i++)
              file << extensions[i]/cells[i] << " ";
            file
              << "\n"
              << "     </DataItem>\n"
              << "    </Geometry>\n"
              << "    <Attribute Name=\""
              << fileName
              << "\" Center=\"Cell\">\n"
              << "     <DataItem ItemType=\"HyperSlab\" Dimensions=\"";
            for (unsigned int i = 0; i < dim-1; i++)
              file << cells[dim-(i+2)] << " ";
            file
              << "1\">\n"
              << "      <DataItem Dimensions=\"3 4\">\n"
              << "       " << j << " 0 0 0\n"
              << "       1 1 1 1\n"
              << "       1 ";
            for (unsigned int i = 0; i < dim-1; i++)
              file << cells[dim-(i+2)] << " ";
            file
              << "\n"
              << "      </DataItem>\n"
              << "      <DataItem Dimensions=\"";
            for (unsigned int i = 0; i < dim; i++)
              file << cells[dim-(i+1)] << " ";
            file
              << "\" Format=\"HDF\">\n"
              << "       " << fileName+".stoch.h5" << ":/stochastic\n"
              << "      </DataItem>\n"
              << "     </DataItem>\n"
              << "    </Attribute>\n"
              << "   </Grid>\n";
          }
          file
            << "  </Grid>\n";
        }

        file
          << " </Domain>\n"
          << "</Xdmf>"
          << std::endl;

        return true;
      }

  }
}

#endif // DUNE_RANDOMFIELD_IO_HH
