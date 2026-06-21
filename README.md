
# GenUtils C++ Project

GenUtils is a consumer-independent C++20 scientific utility library. It provides
linear algebra, iterative solvers, random-number generation, statistics, lattice
and graph data structures, IO, and runtime helpers. Public APIs operate on
matrices, vectors, callables, indices, and generic containers; application models,
simulation state, and domain-specific control flow belong in consumer projects.

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

### Building standalone

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces a static `genutils` library (`genutils::genutils`). Consumers link
the target and inherit its include directories and backend definitions
automatically:

```cmake
add_subdirectory(general_cpp)
target_link_libraries(my_app PRIVATE genutils::genutils)
```

#### Command-line parsing

`genutils::cli::Arguments` is a backend-independent C++20 parser for command-line
tokens and configuration files. It supports `--key=value`, `--key value`, short
or long flags, repeated options, negative numeric values, positional arguments,
quoted file values, comments, and explicit typed conversion errors. It does not
create directories, generate random values, or substitute defaults after invalid
input.

```cpp
#include <UserInterface/ui.h>

const auto arguments = genutils::cli::Arguments::from_argv(argc, argv);
const auto threads = arguments.get_or<std::size_t>("threads", 1);
const auto verbose = arguments.flag("verbose");
```

#### Backends are optional features

All third-party backends are optional and selected by CMake feature flags;
the core builds with none of them. On macOS the Accelerate framework supplies
BLAS/LAPACK and **MKL is never required**.

| Option | Default | Effect | Definition |
|---|---|---|---|
| `GENUTILS_USE_ARMADILLO` | ON | Armadillo linear-algebra backend | `GENUTILS_HAS_ARMADILLO` |
| `GENUTILS_USE_HDF5` | ON | HDF5 dataset IO | `GENUTILS_HAS_HDF5` |
| `GENUTILS_USE_MKL` | OFF | Intel MKL acceleration (non-Apple) | `GENUTILS_HAS_MKL` |
| `GENUTILS_BUILD_SHARED` | OFF | build a shared instead of static library | - |

When a requested backend is not found the build prints a status line and
continues with the feature disabled rather than failing.

- **Armadillo** (recommended): header-only linear algebra. Either point
  `ARMADILLO_INCL_DIR` at an unpacked source tree (no wrapper library needed)
  or install a system package found by `find_package(Armadillo)`.
- **HDF5** (optional): high-performance storage of large datasets.
- **Intel MKL** (optional, non-Apple): optimized BLAS/LAPACK and allocators.

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

All of these are **optional** - the build finds system packages on its own and
disables any backend it cannot locate. Set them only to point at a non-standard
install (e.g. an unpacked Armadillo source tree). They can go in your shell
profile (`.bashrc` / `.zshrc`) or be exported before configuring.

| Variable | Backend | When needed |
|---|---|---|
| `ARMADILLO_INCL_DIR` | Armadillo | Path to an unpacked Armadillo source tree (root containing `include/armadillo`, or the dir containing `armadillo` itself). Skipped if a system Armadillo is found. |
| `HDF5_INCL_DIR` / `HDF5_LIB_DIR` | HDF5 | Fallback when `find_package(HDF5)` fails. |
| `MKL_INCL_DIR` / `MKL_LIB_DIR` | MKL | Only with `-DGENUTILS_USE_MKL=ON` on non-Apple platforms. |

```bash
# Armadillo source tree (header-only, no wrapper library required)
export ARMADILLO_INCL_DIR=/path/to/armadillo-15.2.7
# HDF5 fallback (Linux example)
export HDF5_INCL_DIR=/usr/include/hdf5/serial
export HDF5_LIB_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial
# MKL (non-Apple, only with GENUTILS_USE_MKL=ON)
export MKL_INCL_DIR=/opt/intel/oneapi/mkl/latest/include
export MKL_LIB_DIR=/opt/intel/oneapi/mkl/latest/lib/intel64
```

> macOS uses the Accelerate framework for BLAS/LAPACK; MKL is never required.

## Usage


## Documentation

Detailed documentation is available in the `docs` directory. You can also find examples and API references.

## Contributing

Contributions are welcome! Please read the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainer at [your-email@example.com].
