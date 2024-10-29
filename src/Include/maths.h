#pragma once
#include <iostream>
#ifndef MATH_H
#define MATH_H

#include <cmath>
#include <complex>

#define BETWEXT(val, lower, upper)	val >= lower && val <= upper
#define BETWEEN(val, lower, upper)	val > lower && val < upper

#ifdef __has_include
// matrix base class concepts
#	if __has_include(<concepts>)
		#include <concepts>
		#include <type_traits>
		template<typename _T>
		concept HasDoubleType = std::is_base_of<double, _T>::value								|| 
								std::is_base_of<long double, _T>::value							||
								std::is_base_of<float, _T>::value;

		template<typename _T>
		concept HasIntType	  = std::is_base_of<short, _T>::value								||
								std::is_base_of<unsigned short, _T>::value						||
								std::is_base_of<int, _T>::value									|| 
								std::is_base_of<unsigned int, _T>::value						||
								std::is_base_of<long, _T>::value								||
								std::is_base_of<unsigned long, _T>::value						||
								std::is_base_of<long long, _T>::value							||
								std::is_base_of<unsigned long long, _T>::value;
#	endif
#endif
/*******************************
* Contains the possible methods
* for using math in simulation.
*******************************/

// #################################################################

/*
* @brief Check the sign of a value
* @param val value to be checked
* @return sign of a variable
*/
template <typename T1, typename T2>
inline T1 sgn(T2 val) {
	return (T2(0) < val) - (val < T2(0));
}

// #################################################################

/*
* @brief Placeholder for non-complex values
* @param _r casting the complex value to the correct type
*/
template <typename _T>
inline _T toType(double _r, double _i = 0) {
	return _T(_r, _i);
}
template <>

inline double toType(double _r, double _i) {
	return _r;
}

// #################################################################

/*
* @brief Defines an euclidean modulo denoting also the negative sign
* @param a left side of modulo
* @param b right side of modulo
* @return euclidean a%b
* @link https://en.wikipedia.org/wiki/Modulo_operation
*/
template <typename _T>
inline _T modEUC(_T a, _T b)
{
	_T m = a % b;
	if (m < 0) m = (b < 0) ? m - b : m + b;
	return m;
}

// ###############################################################################

/*
* @brief Given a sum of squares measurement and sum of the measurements
* returns the normalized variance
* @param _av2 sum of unnormalized squares of values
* @param _av sum of unnormalized values
* @param _norm number of measurements
* @returns a normalized variance
*/
template<typename _T>
inline _T variance(_T _av2, _T _av, int _norm)
{
	return std::sqrt((_av2 / _norm - _av * _av) / _norm);
}

// ###############################################################################

namespace Math
{

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Calculates the complex exponential of a number exp(i * x)
	* @param _val value in the exponential
	* @returns exponential of the number
	*/
	template <typename _T, typename _T2>
	inline _T2 expI(_T _val)
	{
		return std::exp(std::complex<double>(0, 1) * _val);
	}

	template <>
	inline double expI<double, double>(double _val)
	{
		return std::real(std::exp(std::complex<double>(0, 1) * _val));
	}
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	/*
	* @brief Calculates a safe exponential function for the given value - checks if the value is too large
	* @param _val value to be exponentiated
	* @returns exponential of the value
	*/
	template <typename _T>
	inline _T expS(_T _val)
	{
    	constexpr double max_exp = 50; // Limit for the exponent to avoid overflow
		if constexpr (std::is_same<_T, std::complex<double>>::value) {
			// Separate real and imaginary parts
			double real_part = std::real(_val);
			double imag_part = std::imag(_val);

			// Cap the real part to avoid overflow
			if (real_part > max_exp) {
				real_part = max_exp;
#ifdef _DEBUG
				std::cerr << "Warning: Exponential overflow, capping real part to " << max_exp << std::endl;
#endif
			}
			else if (real_part < -max_exp) {
				real_part = -max_exp;
#ifdef _DEBUG
				std::cerr << "Warning: Exponential overflow, capping real part to " << -max_exp << std::endl;
#endif
			}
				
			return std::exp(real_part) * std::polar(1.0, imag_part);
		} else {
			// For non-complex types, cap the value
			if (_val > max_exp) 
				return std::exp(max_exp);
			else if (_val < -max_exp)
				return std::exp(-max_exp);
			
			return std::exp(_val);
		}
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Truncates a number at the given precision in base 10
	* @param _val value to be truncated
	* @returns truncated value
	*/
	template <typename _T, uint _P>
	inline _T trunc(_T _val)
	{
		return std::round(_val * (_P + 1)) / (_P + 1);
	}
}

// ###############################################################################


#include <future>
#include <vector>
#include <atomic>
#include <functional>

#ifdef _DEBUG
    #define PRAG_PARALLEL_FOR
    #define PRAG_PARALLEL_FOR_TH(_thNum)
#else
    #define PRAG_PARALLEL_FOR _Pragma("omp parallel for")
    #define PRAG_PARALLEL_FOR_TH(_thNum) _Pragma("omp parallel for num_threads(" #_thNum ")")
#endif

namespace Threading
{
	/*
	* @brief Creates futures for a class member function with arguments and returns the results.
	* 
	* @tparam ClassType Type of the class instance.
	* @tparam _R Return type of the function.
	* @tparam _F Type of the member function. The function necessarlily has to
	* 	take the following arguments: size_t, size_t, std::atomic<size_t>&, size_t, _Args... and return _R.
	*	the first two arguments are the start and stop indices of the iteration, the third is the total number of elements,
	*	the fourth is the thread number, and the rest are the additional arguments.
	* @tparam _Args Types of the additional arguments for the function.
	* 
	* @param _instance Pointer to the class instance.
	* @param _totalIterator Atomic iterator for the total number of elements.
	* @param _thNum Number of threads to be used.
	* @param _isParallel Indicates whether execution is parallel.
	* @param _iterSize Total size of the iteration.
	* @param f Member function pointer.
	* @param args Arguments for the member function.
	* @returns A vector containing results of type _R.
	*/
	template <typename ClassType, typename _R, typename _F, typename ..._Args>
	inline std::vector<_R> createFutures(ClassType* _instance,							// instance of the class
										std::atomic<size_t>& _totalIterator,			// iterator for the total number of elements
										size_t _thNum,									// number of threads		
										bool _isParallel,								// is parallel				
										size_t _iterSize,								// size of the iteration			
										_F &&f, _Args &&...args)						// function and arguments		 
					
	{
		if (!_isParallel)
			_thNum = 1;
		
		// reserve the threads
		std::vector<std::future<_R>> _futures;
		_futures.reserve(_thNum);

		for(auto _ithread = 0; _ithread < _thNum; ++_ithread)
		{
			// get the subvectors
			size_t _startElem	= _ithread * _iterSize / _thNum;
			size_t _stopElem	= (_ithread + 1) * _iterSize / _thNum;

			// Validate indices
			if (_startElem >= _iterSize || _stopElem > _iterSize)
				throw std::out_of_range("Thread indices out of range");

			_futures.push_back(
				std::async(
					[=, &_totalIterator, &_instance]() mutable { 
						// Use mutable lambda to avoid const issues
						// Call the member function with all arguments
						return (_instance->*f)(_startElem, 
												_stopElem, 
												std::ref(_totalIterator), 
												_ithread,
												std::forward<_Args>(args)...);
					}
				)
			);
		}

		std::vector<_R> _out;
		for (auto& future : _futures) {
			try {
				auto _res = future.get();
				_out.push_back(_res);
			} catch (const std::exception& e) {
				std::cerr << "Exception while getting future result: " << e.what() << std::endl;
			}
		}
		return _out;
	}


};


#endif