cmake_minimum_required (VERSION 3.6)
project (exafmm CXX)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_EXAMPLES "Builds example binaries (not quite tests)" ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

# Required packages for ExaFmm library.
find_package(OpenMP REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)

# ExaFmm library:
add_library(exafmm INTERFACE)
target_include_directories(exafmm
    INTERFACE 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    link_libraries(exafmm m)   # Maths std library.
endif()
link_libraries(exafmm OpenMP::OpenMP_CXX Eigen3::Eigen FFTW3::fftw3)

# Make the files appear nicely in Visual Studio.
if (MSVC)
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	file (GLOB EXAFMM_INCLUDE "include/ExaFMM/*")
	source_group("ExaFmm_source" FILES ${EXAFMM_INCLUDE})
	add_custom_target(ExaFmm_source_ SOURCES ${EXAFMM_INCLUDE})
	target_include_directories(exafmm INTERFACE tools/natvis/Eigen.natvis)
endif()

# Create example binaries
if(BUILD_EXAMPLES)
	add_subdirectory("tests")
endif()
