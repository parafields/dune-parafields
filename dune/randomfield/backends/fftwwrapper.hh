#ifndef DUNE_RANDOMFIELD_FFTWWRAPPER_HH
#define DUNE_RANDOMFIELD_FFTWWRAPPER_HH

namespace Dune {
  namespace RandomField {

    /**
     * @brief Fallback class for error message when requested FFTW3 library wasn't found
     *
     * This class is instanciated if a certain data type can't be used with FFTW3,
     * either because FFTW doesn't support this data type in general, or because it
     * wasn't compiled with support for it. This triggers a human-readable error message.
     *
     * @tparam RF data type for FFTW data entries
     */
    template<typename RF>
      class FFTW
      {
        static_assert(std::is_same<RF,void>::value,
            "FFTW library for requested data type not found");
      };

#if HAVE_FFTW3_FLOAT
    /**
     * @brief Wrapper class for float parallel FFTW3 library
     *
     * This class serves as a namespace with template argument of sorts, wrapping
     * the float functions of the FFTW3 library, so that they can be selected
     * using the data type as a template parameter.
     */
    template<>
      class FFTW<float>
      {
        public:

          using complex  = fftwf_complex;
          using plan     = fftwf_plan;
          using r2r_kind = fftwf_r2r_kind;

          // allocation and deallocation

          //! @brief Allocation of storage for real numbers
          static float* alloc_real(ptrdiff_t size)
          {
            return fftwf_alloc_real(size);
          }

          //! @brief Allocation of storage for complex numbers
          static fftwf_complex* alloc_complex(ptrdiff_t size)
          {
            return fftwf_alloc_complex(size);
          }

          //! @brief Deallocation of allocated storage
          template<typename Ptr>
            static void free(Ptr& ptr)
            {
              fftwf_free(ptr);
            }

          // data layout queries

          //!@brief Determine array length and offset in distributed dimension
          static ptrdiff_t mpi_local_size(unsigned int dim, const ptrdiff_t* n, MPI_Comm comm,
              ptrdiff_t* localN0, ptrdiff_t* local0Start)
          {
            return fftwf_mpi_local_size(dim,n,comm,localN0,local0Start);
          }

          //!@brief Determine array length and offset in distributed dimension, 1D case
          static ptrdiff_t mpi_local_size_1d(const ptrdiff_t n0, MPI_Comm comm, int direction, unsigned int flags,
              ptrdiff_t* localN0, ptrdiff_t* local0Start, ptrdiff_t* localN02, ptrdiff_t* local0Start2)
          {
            return fftwf_mpi_local_size_1d(n0,comm,direction,flags,localN0,local0Start,localN02,local0Start2);
          }

          //!@brief Determine array length and offset in distributed dimension, transposed case
          static ptrdiff_t mpi_local_size_transposed(unsigned int dim, const ptrdiff_t* n, MPI_Comm comm,
              ptrdiff_t* localN0, ptrdiff_t* local0Start, ptrdiff_t* localN0Trans, ptrdiff_t* local0StartTrans)
          {
            return fftwf_mpi_local_size_transposed(dim,n,comm,localN0,local0Start,localN0Trans,local0StartTrans);
          }

          //!@brief Determine array length and offset in distributed dimension, second transposed case
          static ptrdiff_t mpi_local_size_many_transposed(unsigned int dim, const ptrdiff_t* n, ptrdiff_t howmany,
              ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm, ptrdiff_t* localN0, ptrdiff_t* local0Start,
              ptrdiff_t* localN0Trans, ptrdiff_t* local0StartTrans)
          {
            return fftwf_mpi_local_size_many_transposed(dim,n,howmany,block0,block1,comm,
                localN0,local0Start,localN0Trans,local0StartTrans);
          }

          // plan creation

          //! @brief Generate discrete Fourier transform plan
          static fftwf_plan mpi_plan_dft(unsigned int dim, const ptrdiff_t* n, fftwf_complex* data1, fftwf_complex* data2,
              MPI_Comm comm, int direction, unsigned int flags)
          {
            return fftwf_mpi_plan_dft(dim,n,data1,data2,comm,direction,flags);
          }

          //! @brief Generate real-to-complex discrete Fourier transform plan
          static fftwf_plan mpi_plan_dft_r2c(unsigned int dim, const ptrdiff_t* n, float* data1, fftwf_complex* data2,
              MPI_Comm comm, unsigned int flags)
          {
            return fftwf_mpi_plan_dft_r2c(dim,n,data1,data2,comm,flags);
          }

          //! @brief Generate complex-to-real discrete Fourier transform plan
          static fftwf_plan mpi_plan_dft_c2r(unsigned int dim, const ptrdiff_t* n, fftwf_complex* data1, float* data2,
              MPI_Comm comm, unsigned int flags)
          {
            return fftwf_mpi_plan_dft_c2r(dim,n,data1,data2,comm,flags);
          }

          //! @brief Generate discrete cosine / sine transform plan
          static fftwf_plan mpi_plan_r2r(unsigned int dim, const ptrdiff_t* n, float* data1, float* data2,
              MPI_Comm comm, r2r_kind* kinds, unsigned int flags)
          {
            return fftwf_mpi_plan_r2r(dim,n,data1,data2,comm,kinds,flags);
          }

          //! @brief Generate discrete cosine / sine transform plan, second version
          static fftwf_plan mpi_plan_many_r2r(unsigned int dim, const ptrdiff_t* n, ptrdiff_t howmany, ptrdiff_t block0,
              ptrdiff_t block1, float* data1, float* data2, MPI_Comm comm, r2r_kind* kinds, unsigned int flags)
          {
            return fftwf_mpi_plan_many_r2r(dim,n,howmany,block0,block1,data1,data2,comm,kinds,flags);
          }

          // plan execution and destruction

          //! @brief Perform discrete transform
          static void execute(fftwf_plan& plan)
          {
            fftwf_execute(plan);
          }

          //! @brief Clean up generated plan
          static void destroy_plan(fftwf_plan& plan)
          {
            fftwf_destroy_plan(plan);
          }

          // wisdom storage

          //! @brief Read in optimized DFT configuration
          static void import_wisdom_from_filename(const char* filename)
          {
            fftwf_import_wisdom_from_filename(filename);
          }

          //! @brief Communicate optimized DFT configuration in parallel case
          static void mpi_broadcast_wisdom(MPI_Comm comm)
          {
            fftwf_mpi_broadcast_wisdom(comm);
          }

          //! @brief Receive optimized DFT configuration in parallel case
          static void mpi_gather_wisdom(MPI_Comm comm)
          {
            fftwf_mpi_gather_wisdom(comm);
          }

          //! @brief Write out optimized DFT configuration
          static void export_wisdom_to_filename(const char* filename)
          {
            fftwf_export_wisdom_to_filename(filename);
          }
      };
#endif // HAVE_FFTW3_FLOAT

#if HAVE_FFTW3_DOUBLE
    /**
     * @brief Wrapper class for double parallel FFTW3 library
     *
     * This class serves as a namespace with template argument of sorts, wrapping
     * the double functions of the FFTW3 library, so that they can be selected
     * using the data type as a template parameter.
     */
    template<>
      class FFTW<double>
      {
        public:

          using complex  = fftw_complex;
          using plan     = fftw_plan;
          using r2r_kind = fftw_r2r_kind;

          // allocation and deallocation

          //! @brief Allocation of storage for real numbers
          static double* alloc_real(ptrdiff_t size)
          {
            return fftw_alloc_real(size);
          }

          //! @brief Allocation of storage for complex numbers
          static fftw_complex* alloc_complex(ptrdiff_t size)
          {
            return fftw_alloc_complex(size);
          }

          //! @brief Deallocation of allocated storage
          template<typename Ptr>
            static void free(Ptr& ptr)
            {
              fftw_free(ptr);
            }

          // data layout queries

          //!@brief Determine array length and offset in distributed dimension
          static ptrdiff_t mpi_local_size(unsigned int dim, const ptrdiff_t* n, MPI_Comm comm,
              ptrdiff_t* localN0, ptrdiff_t* local0Start)
          {
            return fftw_mpi_local_size(dim,n,comm,localN0,local0Start);
          }

          //!@brief Determine array length and offset in distributed dimension, 1D case
          static ptrdiff_t mpi_local_size_1d(const ptrdiff_t n0, MPI_Comm comm, int direction, unsigned int flags,
              ptrdiff_t* localN0, ptrdiff_t* local0Start, ptrdiff_t* localN02, ptrdiff_t* local0Start2)
          {
            return fftw_mpi_local_size_1d(n0,comm,direction,flags,localN0,local0Start,localN02,local0Start2);
          }

          //!@brief Determine array length and offset in distributed dimension, transposed case
          static ptrdiff_t mpi_local_size_transposed(unsigned int dim, const ptrdiff_t* n, MPI_Comm comm,
              ptrdiff_t* localN0, ptrdiff_t* local0Start, ptrdiff_t* localN0Trans, ptrdiff_t* local0StartTrans)
          {
            return fftw_mpi_local_size_transposed(dim,n,comm,localN0,local0Start,localN0Trans,local0StartTrans);
          }

          //!@brief Determine array length and offset in distributed dimension, second transposed case
          static ptrdiff_t mpi_local_size_many_transposed(unsigned int dim, const ptrdiff_t* n, ptrdiff_t howmany,
              ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm, ptrdiff_t* localN0, ptrdiff_t* local0Start,
              ptrdiff_t* localN0Trans, ptrdiff_t* local0StartTrans)
          {
            return fftw_mpi_local_size_many_transposed(dim,n,howmany,block0,block1,comm,
                localN0,local0Start,localN0Trans,local0StartTrans);
          }

          // plan creation

          //! @brief Generate discrete Fourier transform plan
          static fftw_plan mpi_plan_dft(unsigned int dim, const ptrdiff_t* n, fftw_complex* data1, fftw_complex* data2,
              MPI_Comm comm, int direction, unsigned int flags)
          {
            return fftw_mpi_plan_dft(dim,n,data1,data2,comm,direction,flags);
          }

          //! @brief Generate real-to-complex discrete Fourier transform plan
          static fftw_plan mpi_plan_dft_r2c(unsigned int dim, const ptrdiff_t* n, double* data1, fftw_complex* data2,
              MPI_Comm comm, unsigned int flags)
          {
            return fftw_mpi_plan_dft_r2c(dim,n,data1,data2,comm,flags);
          }

          //! @brief Generate complex-to-real discrete Fourier transform plan
          static fftw_plan mpi_plan_dft_c2r(unsigned int dim, const ptrdiff_t* n, fftw_complex* data1, double* data2,
              MPI_Comm comm, unsigned int flags)
          {
            return fftw_mpi_plan_dft_c2r(dim,n,data1,data2,comm,flags);
          }

          //! @brief Generate discrete cosine / sine transform plan
          static fftw_plan mpi_plan_r2r(unsigned int dim, const ptrdiff_t* n, double* data1, double* data2,
              MPI_Comm comm, r2r_kind* kinds, unsigned int flags)
          {
            return fftw_mpi_plan_r2r(dim,n,data1,data2,comm,kinds,flags);
          }

          //! @brief Generate discrete cosine / sine transform plan, second version
          static fftw_plan mpi_plan_many_r2r(unsigned int dim, const ptrdiff_t* n, ptrdiff_t howmany, ptrdiff_t block0,
              ptrdiff_t block1, double* data1, double* data2, MPI_Comm comm, r2r_kind* kinds, unsigned int flags)
          {
            return fftw_mpi_plan_many_r2r(dim,n,howmany,block0,block1,data1,data2,comm,kinds,flags);
          }

          // plan execution and destruction

          //! @brief Perform discrete transform
          static void execute(fftw_plan& plan)
          {
            fftw_execute(plan);
          }

          //! @brief Clean up generated plan
          static void destroy_plan(fftw_plan& plan)
          {
            fftw_destroy_plan(plan);
          }

          // wisdom storage

          //! @brief Read in optimized DFT configuration
          static void import_wisdom_from_filename(const char* filename)
          {
            fftw_import_wisdom_from_filename(filename);
          }

          //! @brief Communicate optimized DFT configuration in parallel case
          static void mpi_broadcast_wisdom(MPI_Comm comm)
          {
            fftw_mpi_broadcast_wisdom(comm);
          }

          //! @brief Receive optimized DFT configuration in parallel case
          static void mpi_gather_wisdom(MPI_Comm comm)
          {
            fftw_mpi_gather_wisdom(comm);
          }

          //! @brief Write out optimized DFT configuration
          static void export_wisdom_to_filename(const char* filename)
          {
            fftw_export_wisdom_to_filename(filename);
          }
      };
#endif // HAVE_FFTW3_DOUBLE

#if HAVE_FFTW3_LONGDOUBLE
    /**
     * @brief Wrapper class for long double parallel FFTW3 library
     *
     * This class serves as a namespace with template argument of sorts, wrapping
     * the long double functions of the FFTW3 library, so that they can be selected
     * using the data type as a template parameter.
     */
    template<>
      class FFTW<long double>
      {
        public:

          using complex  = fftwl_complex;
          using plan     = fftwl_plan;
          using r2r_kind = fftwl_r2r_kind;

          // allocation and deallocation

          //! @brief Allocation of storage for real numbers
          static long double* alloc_real(ptrdiff_t size)
          {
            return fftwl_alloc_real(size);
          }

          //! @brief Allocation of storage for complex numbers
          static fftwl_complex* alloc_complex(ptrdiff_t size)
          {
            return fftwl_alloc_complex(size);
          }

          //! @brief Deallocation of allocated storage
          template<typename Ptr>
            static void free(Ptr& ptr)
            {
              fftwl_free(ptr);
            }

          // data layout queries

          //!@brief Determine array length and offset in distributed dimension
          static ptrdiff_t mpi_local_size(unsigned int dim, const ptrdiff_t* n, MPI_Comm comm,
              ptrdiff_t* localN0, ptrdiff_t* local0Start)
          {
            return fftwl_mpi_local_size(dim,n,comm,localN0,local0Start);
          }

          //!@brief Determine array length and offset in distributed dimension, 1D case
          static ptrdiff_t mpi_local_size_1d(const ptrdiff_t n0, MPI_Comm comm, int direction, unsigned int flags,
              ptrdiff_t* localN0, ptrdiff_t* local0Start, ptrdiff_t* localN02, ptrdiff_t* local0Start2)
          {
            return fftwl_mpi_local_size_1d(n0,comm,direction,flags,localN0,local0Start,localN02,local0Start2);
          }

          //!@brief Determine array length and offset in distributed dimension, transposed case
          static ptrdiff_t mpi_local_size_transposed(unsigned int dim, const ptrdiff_t* n, MPI_Comm comm,
              ptrdiff_t* localN0, ptrdiff_t* local0Start, ptrdiff_t* localN0Trans, ptrdiff_t* local0StartTrans)
          {
            return fftwl_mpi_local_size_transposed(dim,n,comm,localN0,local0Start,localN0Trans,local0StartTrans);
          }

          //!@brief Determine array length and offset in distributed dimension, second transposed case
          static ptrdiff_t mpi_local_size_many_transposed(unsigned int dim, const ptrdiff_t* n, ptrdiff_t howmany,
              ptrdiff_t block0, ptrdiff_t block1, MPI_Comm comm, ptrdiff_t* localN0, ptrdiff_t* local0Start,
              ptrdiff_t* localN0Trans, ptrdiff_t* local0StartTrans)
          {
            return fftwl_mpi_local_size_many_transposed(dim,n,howmany,block0,block1,comm,
                localN0,local0Start,localN0Trans,local0StartTrans);
          }

          // plan creation

          //! @brief Generate discrete Fourier transform plan
          static fftwl_plan mpi_plan_dft(unsigned int dim, const ptrdiff_t* n, fftwl_complex* data1, fftwl_complex* data2,
              MPI_Comm comm, int direction, unsigned int flags)
          {
            return fftwl_mpi_plan_dft(dim,n,data1,data2,comm,direction,flags);
          }

          //! @brief Generate real-to-complex discrete Fourier transform plan
          static fftwl_plan mpi_plan_dft_r2c(unsigned int dim, const ptrdiff_t* n, long double* data1, fftwl_complex* data2,
              MPI_Comm comm, unsigned int flags)
          {
            return fftwl_mpi_plan_dft_r2c(dim,n,data1,data2,comm,flags);
          }

          //! @brief Generate complex-to-real discrete Fourier transform plan
          static fftwl_plan mpi_plan_dft_c2r(unsigned int dim, const ptrdiff_t* n, fftwl_complex* data1, long double* data2,
              MPI_Comm comm, unsigned int flags)
          {
            return fftwl_mpi_plan_dft_c2r(dim,n,data1,data2,comm,flags);
          }

          //! @brief Generate discrete cosine / sine transform plan
          static fftwl_plan mpi_plan_r2r(unsigned int dim, const ptrdiff_t* n, long double* data1, long double* data2,
              MPI_Comm comm, r2r_kind* kinds, unsigned int flags)
          {
            return fftwl_mpi_plan_r2r(dim,n,data1,data2,comm,kinds,flags);
          }

          //! @brief Generate discrete cosine / sine transform plan, second version
          static fftwl_plan mpi_plan_many_r2r(unsigned int dim, const ptrdiff_t* n, ptrdiff_t howmany, ptrdiff_t block0,
              ptrdiff_t block1, long double* data1, long double* data2, MPI_Comm comm, r2r_kind* kinds, unsigned int flags)
          {
            return fftwl_mpi_plan_many_r2r(dim,n,howmany,block0,block1,data1,data2,comm,kinds,flags);
          }

          // plan execution and destruction

          //! @brief Perform discrete transform
          static void execute(fftwl_plan& plan)
          {
            fftwl_execute(plan);
          }

          //! @brief Clean up generated plan
          static void destroy_plan(fftwl_plan& plan)
          {
            fftwl_destroy_plan(plan);
          }

          // wisdom storage

          //! @brief Read in optimized DFT configuration
          static void import_wisdom_from_filename(const char* filename)
          {
            fftwl_import_wisdom_from_filename(filename);
          }

          //! @brief Communicate optimized DFT configuration in parallel case
          static void mpi_broadcast_wisdom(MPI_Comm comm)
          {
            fftwl_mpi_broadcast_wisdom(comm);
          }

          //! @brief Receive optimized DFT configuration in parallel case
          static void mpi_gather_wisdom(MPI_Comm comm)
          {
            fftwl_mpi_gather_wisdom(comm);
          }

          //! @brief Write out optimized DFT configuration
          static void export_wisdom_to_filename(const char* filename)
          {
            fftwl_export_wisdom_to_filename(filename);
          }
      };
#endif // HAVE_FFTW3_LONGDOUBLE

  }
}

#endif // DUNE_RANDOMFIELD_FFTWWRAPPER_HH
