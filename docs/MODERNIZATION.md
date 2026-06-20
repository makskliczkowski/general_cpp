# general_cpp modernization roadmap

Goal: bring general_cpp to the organization and feature set of the Python
reference package `pyqusolver/.../general_python`, modern + fast + memory
efficient for scientific computing, with a clean one-directional
dependency from cpqusolver.

Reference layout (general_python):

- `common/`   : binary, directories, display, flog, hdf5man, memory, parsers, timer
- `maths/`    : math_utils, random, statistics
- `algebra/`  : backend_linalg, eigen, solvers, preconditioners, ode, ran_matrices
- `lattices/` : lattice, square, triangular, honeycomb, hexagonal, graph, tools

Hard rules:

- cpqusolver depends on general_cpp; general_cpp NEVER includes cpqusolver.
- Backends (Armadillo, MKL, HDF5) are optional features behind compile
  flags; the STL-only core always builds standalone.
- No backend `#define`s baked into library headers (config is the build's
  job, not the header's).
- Matrix-free first: solvers/diagonalizers accept a matvec callable so the
  matrix need not be materialized.
- Headers carry declarations + templates only; non-template bodies live in
  `.cpp`.
- Deterministic-vs-entropy seeding is explicit in the RNG API.

## Milestones

- G-M1 Build system: modern CMake, optional MKL/HDF5/Armadillo, STL core
  builds standalone, proper `genutils` target with usage requirements;
  cpqusolver links the target. Env vars documented in README. STATUS: in progress.
- G-M2 common.h decomposition: split arma typedefs out
  (`algebra/arma_aliases`), STL types stay, move bodies to .cpp, stop
  baking ARMA_USE_* / HDF5 defines into `lin_alg.h`.
- G-M3 `common/` reorg: binary, directories, flog, hdf5man, timer, str,
  exceptions, signatures - clean header+cpp per concern, mirroring Python.
- G-M4 `maths/`: math_utils, random (deterministic vs entropy seeding via
  xoshiro), statistics.
- G-M5 `algebra/`: GeneralizedMatrix + arma backend, eigen diagonalizers,
  iterative solvers, preconditioners, ode, ran_matrices; matrix-free matvec.
- G-M6 `lattices/`: SoA neighbor storage; neighbors, transformations,
  connectivity, adjacency matrices, boundary fluxes, sublattices with
  multiparticity, kspace; chain/triangular/graph types.
- G-M7 Integration + docs: one-directional dependency gate, READMEs with
  every env var, consolidation, audit vs Python for missing functionality.

## Verification

The full general_cpp standalone build (this machine: Armadillo source tree +
HDF5, no MKL) plus the cpqusolver test suite (Debug/Release/no-arma) gate
every milestone. A milestone is done only when both build clean and the
relevant tests pass.
