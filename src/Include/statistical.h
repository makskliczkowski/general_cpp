#pragma once
#include "../common.h"
// armadillo flags:
#define ARMA_USE_LAPACK             
#define ARMA_PRINT_EXCEPTIONS
//#define ARMA_BLAS_LONG_LONG                                                                 // using long long inside LAPACK call
//#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
//#define ARMA_DONT_USE_WRAPPER
//#define ARMA_USE_SUPERLU
//#define ARMA_USE_ARPACK 
#define ARMA_USE_MKL_ALLOC
#define ARMA_USE_MKL_TYPES
#define ARMA_WARN_LEVEL 1
#define ARMA_DONT_USE_OPENMP
#define ARMA_USE_HDF5
////#define ARMA_USE_OPENMP
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

// ######################################################## binning ########################################################

/*
* @brief bin the data to calculate the correlation time approximation
* @param seriesData data to be binned
* @param bins the vector to save the average into
* @param binSize the size of a given single bin
*/
template<typename _T>
inline void binning(const _T& seriesData, _T& bins, uint binSize) {
	if (binSize * bins.size() > seriesData.size()) throw "Cannot create bins of insufficient elements";
	for (int i = 0; i < bins.size(); i++)
		bins[i] = arma::mean(seriesData.subvec(binSize * i, binSize * (i + 1) - 1));
}


// ######################################################## statistical meassures ########################################################

/*
* @brief Approximate the correlation error
* @param bins the bin average of the data
* @returns approximation of a statistical correlation error
*/
template<typename _T>
inline _T correlationError(const arma::Col<_T>& bins) {
	return arma::real(sqrt(arma::var(bins) / bins.size()));
}

/*
* @brief for an std::vector calculates the standard deviation
*/
template <typename _T>
_T stddev(const v_1d<_T>& v)
{
	_T mean = std::accumulate(v.begin(), v.end(), T(0.0)) / T(v.size());
	v_1d<_T> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(), [mean](T x) { return x - mean; });
	_T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), T(0.0));
	return std::sqrt(sq_sum / cpx(v.size()));
}