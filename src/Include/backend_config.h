/******************************************************************************
 *
 *  @file src/backend_config.h
 *  @brief Centralized Armadillo / BLAS / HDF5 / MKL backend configuration.
 *
 *  Project : general_cpp
 *  Author  : Maksymilian Kliczkowski
 *
 *  All third-party numeric-backend `#define`s live here, in one place, and
 *  defer to the build system. 
 * 
 *  The build communicates its choices through opt-out macros, for example:
 *    - GENUTILS_NO_ARMADILLO : do not pull Armadillo at all.
 *    - GENUTILS_NO_HDF5      : build Armadillo without HDF5 support.
 *    - GENUTILS_HAS_MKL      : opt in to MKL allocators/types (non-Apple).
 *
 *  Including this header is the single supported way to bring in Armadillo;
 *  include it before any `<armadillo>` use.
 *
 ******************************************************************************/

#pragma once
#ifndef GENUTILS_BACKEND_CONFIG_H
#define GENUTILS_BACKEND_CONFIG_H

#ifndef GENUTILS_NO_ARMADILLO

// --- LAPACK / exception handling (always on for the Armadillo backend) ------
#ifndef ARMA_USE_LAPACK
#	define ARMA_USE_LAPACK          // Armadillo's LAPACK support is always on for this project.
#endif
#ifndef ARMA_PRINT_EXCEPTIONS
#	define ARMA_PRINT_EXCEPTIONS    // Armadillo will print exceptions to stderr by default...
#endif

// --- MKL allocators/types: non-Apple, opt-in via GENUTILS_HAS_MKL -----------
#if !defined(__APPLE__) && defined(GENUTILS_HAS_MKL)
#	ifndef ARMA_USE_MKL_ALLOC
#		define ARMA_USE_MKL_ALLOC
#	endif
#	ifndef ARMA_USE_MKL_TYPES
#		define ARMA_USE_MKL_TYPES
#	endif
#endif

// --- threading: Armadillo's own OpenMP stays off (callers manage threads) ---
// TODO: Validate if we always want this off, or if we should allow opt-in via the build system.
#ifndef ARMA_DONT_USE_OPENMP
#	define ARMA_DONT_USE_OPENMP
#endif

// --- HDF5: on unless the build opts out --------------------------------------
#if !defined(GENUTILS_NO_HDF5)
#	ifndef ARMA_USE_HDF5
#		define ARMA_USE_HDF5
#	endif
#	ifndef DH5_USE_110_API
#		define DH5_USE_110_API
#	endif
#	ifndef D_HDF5USEDLL_
#		define D_HDF5USEDLL_
#	endif
#endif

// --- warning suppression -----------------------------------------------------
#ifndef ARMA_ALLOW_FAKE_GCC
#	define ARMA_ALLOW_FAKE_GCC
#endif
#ifndef ARMA_DONT_PRINT_CXX11_WARNING
#	define ARMA_DONT_PRINT_CXX11_WARNING
#endif
#ifndef ARMA_DONT_PRINT_CXX03_WARNING
#	define ARMA_DONT_PRINT_CXX03_WARNING
#endif
#ifndef ARMA_DONT_PRINT_FAST_MATH_WARNING
#	define ARMA_DONT_PRINT_FAST_MATH_WARNING
#endif

// --- finally, pull in Armadillo ------------------------------------------------
#include <armadillo>

#endif // !GENUTILS_NO_ARMADILLO

#endif // !GENUTILS_BACKEND_CONFIG_H

// ------------------------------------------------------------------------------
//! EOF
// ------------------------------------------------------------------------------
