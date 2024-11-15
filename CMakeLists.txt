cmake_minimum_required(VERSION 3.23.0)
project(GenUtils VERSION 1.1.0 LANGUAGES CXX)

# Detect platform
if (WIN32)
    set(OS_PLATFORM "windows")
    message(STATUS "Compiling on Windows")
elseif (UNIX)
    set(OS_PLATFORM "linux")
    message(STATUS "Compiling on Linux")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Set compilers
if (OS_PLATFORM STREQUAL "windows")
    set(CMAKE_C_COMPILER "$ENV{CC}")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}")
else()
    find_program(ICPX_COMPILER icpx)
    if (ICPX_COMPILER)
        set(CMAKE_C_COMPILER icx)
        set(CMAKE_CXX_COMPILER icpx)
        message(STATUS "Using Intel compilers")
    else()
        set(CMAKE_C_COMPILER "/usr/bin/gcc")
        set(CMAKE_CXX_COMPILER "/usr/bin/g++")
        message(STATUS "Using GCC compilers")
    endif()
endif()

message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set variables for source and include directories
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/cpp")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src")

# Find source files (.cpp) and header files (.h or .hpp) recursively
file(GLOB_RECURSE SOURCES "${SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS "${INCLUDE_DIR}/*.h" "${INCLUDE_DIR}/*.hpp")

##########################################################################

# Find the path of the MKL include directory using an environment variable
find_path(MKL_INCL_DIR
    NAMES "mkl.h"
    PATHS "$ENV{MKL_INCL_DIR}"
    DOC "Path to MKL include directory"
)

# Find the path of the MKL library directory using an environment variable
find_path(MKL_LIB_DIR
    NAMES "mkl_core.lib" "libmkl_core.so"
    PATHS "$ENV{MKL_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/opt/intel/mkl/lib"
    DOC "Path to MKL library directory"
)

# Find the path of the HDF5 include directory using an environment variable or common paths
find_path(HDF5_INCL_DIR
    NAMES "hdf5.h"
    PATHS "$ENV{HDF5_INCL_DIR}" "/usr/include/hdf5/serial" "/usr/local/include/hdf5"
    DOC "Path to HDF5 include directory"
)

# Find the path of the HDF5 library directory using an environment variable or common paths
find_path(HDF5_LIB_DIR
    NAMES "hdf5.lib" "libhdf5.so"
    PATHS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial"
    DOC "Path to HDF5 library directory"
)

# Find the path of the HDF5 library directory
find_library(HDF5_LIBRARIES NAMES hdf5 HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")
find_library(HDF5_HL_LIBRARIES NAMES hdf5_hl HINTS "$ENV{HDF5_LIB_DIR}" "/usr/lib" "/usr/local/lib" "/usr/lib/x86_64-linux-gnu/hdf5/serial")

# Find the path of the Armadillo include directory
find_path(ARMADILLO_INCL_DIR
    NAMES "armadillo"
    PATHS "$ENV{ARMADILLO_INCL_DIR}" "/usr/include" "/usr/local/include"
    DOC "Path to Armadillo include directory"
)

##########################################################################

# Print the paths found
message(STATUS "MKL include directory: ${MKL_INCL_DIR}")
message(STATUS "MKL library directory: ${MKL_LIB_DIR}")
message(STATUS "Armadillo include directory: ${ARMADILLO_INCL_DIR}")
message(STATUS "HDF5 include directory: ${HDF5_INCL_DIR}")
message(STATUS "HDF5 library directory: ${HDF5_LIB_DIR}")

# Include directories
include_directories(
    ${INCLUDE_DIR}
    ${MKL_INCL_DIR}
    ${ARMADILLO_INCL_DIR}
    ${HDF5_INCL_DIR}
)

# Specify HDF5 library path directly if needed
link_directories("${HDF5_LIB_DIR}")
link_directories("${MKL_LIB_DIR}")
link_directories("${ARMADILLO_LIB_DIR}")

##########################################################################

# Define the library target
add_library(genutils SHARED ${SOURCES} ${HEADERS})

# Link libraries for threading or platform-specific needs
set(LIBRARIES)
if (OS_PLATFORM STREQUAL "linux")
    find_library(PTHREAD_LIBRARY pthread)
    find_library(DL_LIBRARY dl)
    list(APPEND LIBRARIES ${PTHREAD_LIBRARY} ${DL_LIBRARY})
endif()

# Compiler settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Platform-specific flags
if (OS_PLATFORM STREQUAL "linux")
    if (CMAKE_CXX_COMPILER STREQUAL "icpx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qopenmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
    endif()
elseif (OS_PLATFORM STREQUAL "windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# Output library settings
set_target_properties(genutils PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "genutils"
    DEBUG_POSTFIX "_debug"
    RELEASE_POSTFIX ""
)

# Link platform-specific libraries and additional libraries (MKL, Armadillo, HDF5)
target_link_libraries(genutils PRIVATE ${LIBRARIES} ${ARMADILLO_LIBRARIES} ${MKL_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})