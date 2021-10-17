cmake_minimum_required (VERSION 3.6)
project (exafmm CXX)
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")#-openmp:llvm -openmp:experimental")
endif()

# cmake settings
set (CMAKE_VERBOSE_MAKEFILE on)
set (CMAKE_COLOR_MAKEFILE on)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake-modules")

# check openmp
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
  message (FATAL_ERROR "The compiler does not support OpenMP.")
endif()

# check BLAS and LAPACK
find_package(OpenBLAS CONFIG REQUIRED)
find_package(clapack CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(docopt CONFIG REQUIRED)

# targets
include_directories (${PROJECT_SOURCE_DIR}/include)

add_executable (fmm_laplace tests/fmm_laplace.cpp)
add_executable (fmm_helmholtz tests/fmm_helmholtz.cpp)
add_executable (fmm_modified_helmholtz tests/fmm_modified_helmholtz.cpp)

target_link_libraries (fmm_laplace PRIVATE OpenBLAS::OpenBLAS
                                   f2c lapack
                                   FFTW3::fftw3
								   docopt)

target_link_libraries (fmm_helmholtz PRIVATE OpenBLAS::OpenBLAS
                                   f2c lapack
                                   FFTW3::fftw3
								   docopt)

target_link_libraries (fmm_modified_helmholtz PRIVATE OpenBLAS::OpenBLAS
                                   f2c lapack
                                   FFTW3::fftw3
								   docopt)
