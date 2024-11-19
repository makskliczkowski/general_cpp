
# GenUtils C++ Project

This repository is a C++ project that involves linear algebra utilities, random number generation, and other utility functions used in quantum systems simulations. Below you will find instructions on how to set up the environment, configure the project, and the necessary libraries and environmental variables.

This library provides a set of tools and functions to solve eigenvalue problems in quantum mechanics using C++.

## Features
Features

- Linear Algebra Utilities: Includes solvers, preconditioners, and tools for handling generalized matrices.
- Random Number Generators: High-quality pseudo-random number generation, including implementations like xoshiro.
- Lattice Structures: Support for defining and working with square and hexagonal lattice configurations.
- Templated Design: Enables flexible integration and extensibility for a variety of use cases.
- Helper Functions: For operations such as string manipulation, directory handling, and mathematical utilities.
- Scalable and Efficient: Utilizes modern C++ techniques and libraries (e.g., Armadillo, Intel MKL) for high performance.

## TODO

- Add support to build and distribute as a shared library.
- Improve and extend the documentation.
- Implement automated tests.
- Move some of the files to .cpp sources
- Extend linear algebra utilities
- Extend lattice functionality
- Efficiency improvements
- Add suport to plain MKL and add more generalized templates

## Installation

To install the library, clone the repository and build it using CMake:

### From the QuantumEigenSolver

The repository is already integrated as a submodule in the [QuantumEigenSolver](https://github.com/makskliczkowski/QuantumEigenSolver) project. To use it within QuantumEigenSolver, simply clone the main repository with submodules enabled:
```sh
git clone --recurse-submodules https://github.com/makskliczkowski/QuantumEigenSolver.git
```
and 
```sh
cd QuantumEigenSolver
mkdir build
cd build
cmake ..
make
```
### From here as a shared library

#### Required Libraries

Before compiling the project, make sure you have the following libraries installed:

- **Intel MKL (Math Kernel Library)**:
    - Used for optimized mathematical operations and parallel computing.
- **HDF5**:
    - Used for high-performance storage of large datasets.
- **Armadillo**:
    - C++ library for linear algebra and scientific computing.

#### Installing Libraries

If these libraries are not already installed, you can follow the instructions below to install them on Linux. You may need to adjust commands for other platforms.

#### Intel MKL
1. **Install Intel oneAPI Toolkit**: You can download the Intel oneAPI toolkit that includes MKL from the [Intel website](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html).
2. **Installation for Linux**:
    ```bash
    sudo apt-get install intel-oneapi-mkl
    ```

#### HDF5
1. **Install HDF5** on Linux:
    ```bash
    sudo apt-get install libhdf5-dev
    ```

#### Armadillo
1. **Install Armadillo** on Linux:
    ```bash
    sudo apt-get install libarmadillo-dev
    ```

#### Environmental Variables

The following environment variables need to be set in your system to help the build system find the required libraries and include directories. You can set these in your shell configuration file (e.g., `.bashrc` or `.zshrc` for Linux/macOS) or manually before building.

##### 1. **MKL_INCL_DIR**
- **Description**: Path to the Intel MKL include directory.
- **Example**:
    ```bash
    export MKL_INCL_DIR=/opt/intel/oneapi/mkl/latest/include
    ```

##### 2. **MKL_LIB_DIR**
- **Description**: Path to the Intel MKL library directory.
- **Example**:
    ```bash
    export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64
    ```

##### 3. **HDF5_INCL_DIR**
- **Description**: Path to the HDF5 include directory.
- **Example**:
    ```bash
    export HDF5_INCL_DIR=/usr/include/hdf5/serial
    ```

##### 4. **HDF5_LIB_DIR**
- **Description**: Path to the HDF5 library directory.
- **Example**:
    ```bash
    export HDF5_LIB_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial
    ```

##### 5. **ARMADILLO_INCL_DIR**
- **Description**: Path to the Armadillo include directory.
- **Example**:
    ```bash
    export ARMADILLO_INCL_DIR=/usr/include
    ```

| Variable           | Description                | Example Path                                   |
|--------------------|----------------------------|-----------------------------------------------|
| `MKL_INCL_DIR`     | Intel MKL include directory | `/opt/intel/oneapi/mkl/latest/include`         |
| `MKL_LIB_DIR`      | Intel MKL library directory | `/opt/intel/oneapi/mkl/latest/lib/intel64`     |
| `HDF5_INCL_DIR`    | HDF5 include directory      | `/usr/include/hdf5/serial`                    |
| `HDF5_LIB_DIR`     | HDF5 library directory      | `/usr/lib/x86_64-linux-gnu/hdf5/serial`       |
| `ARMADILLO_INCL_DIR` | Armadillo include directory | `/usr/include`                                |

#### Verify the Setup

To verify that the necessary environmental variables are set, you can check each variable by running:

```bash
echo $MKL_INCL_DIR
echo $MKL_LIB_DIR
echo $HDF5_INCL_DIR
echo $HDF5_LIB_DIR
echo $ARMADILLO_INCL_DIR
```
```plaintext
│   ├── src/                   # Source files from the external library
│   │   ├── binary.h
│   │   ├── lin_alg.h
│   │   ├── plotter.h
│   │   ├── lattices.h
│   │   ├── xoshiro_pp.h
│   │   ├── Include/
│   │   │   ├── random.h
│   │   │   ├── str.h
│   │   │   ├── directories.h
│   │   │   ├── linalg/
│   │   │   │   ├── generalized_matrix.h
│   │   │   │   ├── diagonalizers.h
│   │   │   └── exceptions.h
│   │   ├── flog.h
│   │   ├── Lattices/
│   │   │   ├── square.h
│   │   │   └── hexagonal.h
│   │   ├── UserInterface/ui.h
│   │   └── common.h
│   └── cpp/
│       ├── time.cpp
│       ├── signatures.cpp
│       ├── exceptions.cpp
│       ├── str.cpp
│       ├── ui.cpp
│       ├── LinearAlgebra/
│       │   ├── preconditioners.cpp
│       │   ├── pfaffian.cpp
│       │   ├── Solvers/
│       │   │   ├── solvers_pseudo.cpp
│       │   │   ├── solvers_direct.cpp
│       │   │   ├── solvers_minres.cpp
│       │   │   ├── solvers_arma.cpp
│       │   │   ├── solvers_minresqlp.cpp
│       │   │   ├── solvers_arnoldi.cpp
│       │   │   └── solvers_cg.cpp
│       │   └── solvers.cpp
│       ├── directories.cpp
│       ├── Lattices/
│       │   ├── square.cpp
│       │   └── hexagonal.cpp
│       ├── common.cpp
│       └── maths.cpp
```
## Usage
Integrate the headers and source files into your project, link against the required libraries, and include the headers as needed. For example:
```cpp
#include "src/Include/linalg/generalized_matrix.h"
#include "src/Include/lin_alg.h"
#include "src/Lattices/square.h"
#include "src/Include/random.h"

int main() {
    // test the solvers
    auto _eps 				= 1e-13;
    auto _max_iter 			= 1000;
    auto _reg 				= 1e-15;

    auto _preconditionerType = -1;
    LOGINFO("Using real now...", LOG_TYPES::TRACE, 50, 'x', 0);
    // real 
    {
        algebra::Solvers::General::Tests::solve_test_multiple<double, true>(_eps, _max_iter, _reg, _preconditionerType, false);
        LOGINFO(5);
        // make random
        algebra::Solvers::General::Tests::solve_test_multiple<double, true>(_eps, _max_iter, _reg, _preconditionerType, true);
        LOGINFO(5);
    }
    // add preconditioner
    _preconditionerType = 1;
    // real with preconditioner
    {
        algebra::Solvers::General::Tests::solve_test_multiple<double, true>(_eps, _max_iter, _reg, _preconditionerType, false);
        LOGINFO(5);
        // make random with preconditioner
        algebra::Solvers::General::Tests::solve_test_multiple<double, true>(_eps, _max_iter, _reg, _preconditionerType, true);
        LOGINFO(5);
    }
    // Your simulation code here
}
```

## Documentation

Comprehensive documentation is in progress and will be added under the docs/ directory.

## Contributing

Contributions are welcome! Please submit issues or pull requests to improve the library. Guidelines will be outlined in a CONTRIBUTING.md file (coming soon).
## License

This project is licensed under the MIT License. See the `LICENSE` file for more details. !NOT YET IMPLEMENTED!

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainer at [maksymilian.kliczkowski@pwr.edu.pl] or [maxgrom97@gmail.com]. 
