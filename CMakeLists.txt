cmake_minimum_required (VERSION 3.6)
project (exafmm CXX)
set(CMAKE_CXX_STANDARD 17)

# Libraries
find_package(OpenMP REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(docopt CONFIG REQUIRED)

# targets
include_directories (${PROJECT_SOURCE_DIR}/include)

add_executable (fmm_laplace tests/fmm_laplace.cpp)
add_executable (fmm_helmholtz tests/fmm_helmholtz.cpp)
add_executable (fmm_modified_helmholtz tests/fmm_modified_helmholtz.cpp)

target_link_libraries (fmm_laplace PRIVATE Eigen3::Eigen
                                   FFTW3::fftw3
								   docopt
								   PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries (fmm_helmholtz PRIVATE Eigen3::Eigen
                                   FFTW3::fftw3
								   docopt
								   PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries (fmm_modified_helmholtz PRIVATE Eigen3::Eigen
                                   FFTW3::fftw3
								   docopt
								   PUBLIC OpenMP::OpenMP_CXX)
