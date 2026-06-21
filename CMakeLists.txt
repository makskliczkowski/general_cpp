cmake_minimum_required(VERSION 3.23)
project(GenUtils VERSION 1.2.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# general_cpp: utility library (logging, IO, lattices, linear algebra,
# random, statistics)! Hello, it's a grab bag of useful stuff. 
# The core is STL-only and always builds; Armadillo, HDF5, MKL are optional backends. 
#
# On macOS the Accelerate framework provides BLAS/LAPACK and MKL is never required.
# Environment variables honoured: 
# - ARMADILLO_INCL_DIR, 
# - MKL_INCL_DIR,
# - MKL_LIB_DIR, 
# - HDF5_INCL_DIR, 
# - HDF5_LIB_DIR 
# (see README.md).
# ---------------------------------------------------------------------------

include(GNUInstallDirs)

# Definition of options
option(GENUTILS_USE_ARMADILLO "Enable the Armadillo linear-algebra backend" ON)
option(GENUTILS_USE_HDF5      "Enable HDF5 file support"                    ON)
option(GENUTILS_USE_MKL       "Enable Intel MKL acceleration (non-Apple)"   OFF)
option(GENUTILS_BUILD_SHARED  "Build genutils as a shared library"          OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Sources. 
# STL-only units form the always-buildable core; the Armadillo
# units are added only when the backend is enabled.
# ---------------------------------------------------------------------------
set(GENUTILS_CORE_SOURCES
    cpp/common.cpp
    cpp/directories.cpp
    cpp/exceptions.cpp
    cpp/maths.cpp
    cpp/ml.cpp
    cpp/signatures.cpp
    cpp/slurm.cpp
    cpp/str.cpp
    cpp/time.cpp
    cpp/ui.cpp
    cpp/Lattices/lattice_symmetry.cpp
    cpp/LinearAlgebra/lin_alg.cpp
)

set(GENUTILS_ARMA_SOURCES
    cpp/Lattices/lattices.cpp
    cpp/Lattices/square.cpp
    cpp/Lattices/triangular.cpp
    cpp/Lattices/honeycomb.cpp
    cpp/Lattices/hexagonal.cpp
    cpp/LinearAlgebra/ode.cpp
    cpp/LinearAlgebra/pfaffian.cpp
    cpp/LinearAlgebra/preconditioners.cpp
    cpp/LinearAlgebra/solvers.cpp
    cpp/Random/montecarlo.cpp
)

set(GENUTILS_SOURCES ${GENUTILS_CORE_SOURCES})
if(GENUTILS_USE_ARMADILLO)
    list(APPEND GENUTILS_SOURCES ${GENUTILS_ARMA_SOURCES})
endif()

if(GENUTILS_BUILD_SHARED)
    add_library(genutils SHARED ${GENUTILS_SOURCES})
else()
    add_library(genutils STATIC ${GENUTILS_SOURCES})
endif()

# Add an alias for convenient usage from other CMake projects via find_package(GenUtils).
add_library(genutils::genutils ALIAS genutils)

# ---------------------------------------------------------------------------
# C++20, include directories. The public include directory is the same as the source directory
# ---------------------------------------------------------------------------

target_compile_features(genutils PUBLIC cxx_std_20)
target_include_directories(genutils PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(NOT GENUTILS_USE_ARMADILLO)
    target_compile_definitions(genutils PUBLIC GENUTILS_NO_ARMADILLO)
endif()

# ---------------------------------------------------------------------------
# Armadillo: a source tree pointed to by ARMADILLO_INCL_DIR takes precedence
# (header-only, no wrapper); otherwise find_package. On macOS Accelerate
# supplies BLAS/LAPACK; elsewhere BLAS/LAPACK (or MKL) is linked.
# ---------------------------------------------------------------------------
set(GENUTILS_ARMA_AVAILABLE OFF)
if(GENUTILS_USE_ARMADILLO)
    set(_genutils_arma_env "$ENV{ARMADILLO_INCL_DIR}")
    unset(_genutils_arma_include)
    if(_genutils_arma_env)
        if(EXISTS "${_genutils_arma_env}/include/armadillo")
            set(_genutils_arma_include "${_genutils_arma_env}/include")
        elseif(EXISTS "${_genutils_arma_env}/armadillo")
            set(_genutils_arma_include "${_genutils_arma_env}")
        endif()
    endif()

    if(_genutils_arma_include)
        target_compile_definitions(genutils PUBLIC
            GENUTILS_HAS_ARMADILLO ARMA_DONT_USE_WRAPPER)
        target_include_directories(genutils PUBLIC "${_genutils_arma_include}")
        if(APPLE)
            target_link_libraries(genutils PUBLIC "-framework Accelerate")
            set(GENUTILS_ARMA_AVAILABLE ON)
        else()
            find_package(BLAS QUIET)
            find_package(LAPACK QUIET)
            if(BLAS_FOUND AND LAPACK_FOUND)
                target_link_libraries(genutils PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
                set(GENUTILS_ARMA_AVAILABLE ON)
            else()
                message(WARNING "ARMADILLO_INCL_DIR set but BLAS/LAPACK not found; Armadillo disabled.")
            endif()
        endif()
        if(GENUTILS_ARMA_AVAILABLE)
            message(STATUS "GenUtils Armadillo (source tree): ${_genutils_arma_include}")
        endif()
    else()
        find_package(Armadillo QUIET)
        if(ARMADILLO_FOUND)
            target_compile_definitions(genutils PUBLIC GENUTILS_HAS_ARMADILLO)
            target_include_directories(genutils PUBLIC ${ARMADILLO_INCLUDE_DIRS})
            target_link_libraries(genutils PUBLIC ${ARMADILLO_LIBRARIES})
            set(GENUTILS_ARMA_AVAILABLE ON)
            message(STATUS "GenUtils Armadillo (system): ${ARMADILLO_INCLUDE_DIRS}")
        endif()
    endif()

    if(NOT GENUTILS_ARMA_AVAILABLE)
        message(WARNING "GENUTILS_USE_ARMADILLO=ON but Armadillo not found; backend disabled.")
    endif()
endif()

# ---------------------------------------------------------------------------
# HDF5 (optional). find_package first, then HDF5_INCL_DIR / HDF5_LIB_DIR.
# ---------------------------------------------------------------------------
if(GENUTILS_USE_HDF5)
    find_package(HDF5 QUIET COMPONENTS C)
    if(HDF5_FOUND)
        target_compile_definitions(genutils PUBLIC GENUTILS_HAS_HDF5 ARMA_USE_HDF5)
        target_include_directories(genutils PUBLIC ${HDF5_INCLUDE_DIRS})
        target_link_libraries(genutils PUBLIC ${HDF5_LIBRARIES})
        message(STATUS "GenUtils HDF5: ${HDF5_INCLUDE_DIRS}")
    else()
        set(_genutils_hdf5_inc "$ENV{HDF5_INCL_DIR}")
        if(_genutils_hdf5_inc AND EXISTS "${_genutils_hdf5_inc}/hdf5.h")
            find_library(GENUTILS_HDF5_LIB NAMES hdf5 HINTS "$ENV{HDF5_LIB_DIR}")
            if(GENUTILS_HDF5_LIB)
                target_compile_definitions(genutils PUBLIC GENUTILS_HAS_HDF5 ARMA_USE_HDF5)
                target_include_directories(genutils PUBLIC "${_genutils_hdf5_inc}")
                target_link_libraries(genutils PUBLIC "${GENUTILS_HDF5_LIB}")
                message(STATUS "GenUtils HDF5 (env): ${_genutils_hdf5_inc}")
            else()
                message(STATUS "GenUtils HDF5 requested but library not found; disabled.")
            endif()
        else()
            message(STATUS "GenUtils HDF5 requested but not found; disabled.")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# MKL (optional, non-Apple acceleration). Honours MKL_INCL_DIR / MKL_LIB_DIR.
# ---------------------------------------------------------------------------
if(GENUTILS_USE_MKL AND NOT APPLE)
    set(_genutils_mkl_inc "$ENV{MKL_INCL_DIR}")
    if(_genutils_mkl_inc AND EXISTS "${_genutils_mkl_inc}/mkl.h")
        target_compile_definitions(genutils PUBLIC GENUTILS_HAS_MKL ARMA_USE_MKL_ALLOC ARMA_USE_MKL_TYPES)
        target_include_directories(genutils PUBLIC "${_genutils_mkl_inc}")
        find_library(GENUTILS_MKL_CORE NAMES mkl_core HINTS "$ENV{MKL_LIB_DIR}")
        if(GENUTILS_MKL_CORE)
            target_link_libraries(genutils PUBLIC "${GENUTILS_MKL_CORE}")
        endif()
        message(STATUS "GenUtils MKL: ${_genutils_mkl_inc}")
    else()
        message(STATUS "GENUTILS_USE_MKL set but MKL_INCL_DIR not found; MKL disabled.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Threads + OpenMP.
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
target_link_libraries(genutils PUBLIC Threads::Threads)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(genutils PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(genutils PROPERTIES
    VERSION ${PROJECT_VERSION}
    DEBUG_POSTFIX "_debug"
)

include(CTest)
if(BUILD_TESTING)
    add_executable(genutils_cli_tests tests/cli_arguments.cpp)
    target_link_libraries(genutils_cli_tests PRIVATE genutils::genutils)
    add_test(NAME genutils.cli COMMAND genutils_cli_tests)

    add_executable(genutils_statistics_tests tests/statistics.cpp)
    target_link_libraries(genutils_statistics_tests PRIVATE genutils::genutils)
    add_test(NAME genutils.statistics COMMAND genutils_statistics_tests)
endif()

# ---------------------------------------------------------------------------
#! EOF
# ---------------------------------------------------------------------------
