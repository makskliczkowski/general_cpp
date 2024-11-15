
# GenUtils C++ Project

This repository is a C++ project that involves linear algebra utilities, random number generation, and other utility functions used in quantum systems simulations. Below you will find instructions on how to set up the environment, configure the project, and the necessary libraries and environmental variables.

This library provides a set of tools and functions to solve eigenvalue problems in quantum mechanics using C++.

## Features

- Efficient algorithms for solving eigenvalue problems
- Support for various matrix types and sizes
- Easy-to-use API for integrating into your projects
- Easy-to-use tools for various .cpp implementations
- Easy-to-translate methods for other languages
- Heavily templated

## TODO

- Implement this as a shared library

## Installation

To install the library, clone the repository and build it using CMake:

### From the QuantumEigenSolver
```sh
git clone https://github.com/makskliczkowski/QuantumEigenSolver.git
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

#### Verify the Setup

To verify that the necessary environmental variables are set, you can check each variable by running:

```bash
echo $MKL_INCL_DIR
echo $MKL_LIB_DIR
echo $HDF5_INCL_DIR
echo $HDF5_LIB_DIR
echo $ARMADILLO_INCL_DIR
```
```
src/
├── user_interface/
│   ├── ui_check_eth.cpp
│   ├── ui_check_nqs.cpp
│   ├── ui_check_quadratic.cpp
│   └── ui_check_symmetries.cpp
├── LinearAlgebra/
│   ├── Solvers/
│   │   ├── solvers_cg.cpp
│   │   ├── solvers_minres.cpp
│   │   └── solvers_minresqlp.cpp
│   ├── preconditioners.cpp
│   └── pfaffian.cpp
├── Lattices/
│   ├── hexagonal.cpp
│   └── square.cpp
├── nqs.cpp
└── operator_parser.cpp
```
## Usage


## Documentation

Detailed documentation is available in the `docs` directory. You can also find examples and API references.

## Contributing

Contributions are welcome! Please read the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainer at [your-email@example.com].
