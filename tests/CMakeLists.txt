
find_package(docopt CONFIG REQUIRED)
add_executable (fmm_laplace fmm_laplace.cpp)
add_executable (fmm_helmholtz fmm_helmholtz.cpp)
add_executable (fmm_modified_helmholtz fmm_modified_helmholtz.cpp)

target_link_libraries (fmm_laplace PRIVATE exafmm docopt)
target_link_libraries (fmm_helmholtz PRIVATE exafmm docopt)
target_link_libraries (fmm_modified_helmholtz PRIVATE exafmm docopt)