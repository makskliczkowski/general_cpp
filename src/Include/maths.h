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
template <typename T1, typename T2 = T1>
inline T1 sgn(T2 val) {
	if (val == T2(0.0)) 
		return T1(0.0);
	return (T2(0) < T2(val)) - (T2(val) < T2(0));
}

template <>
inline double sgn(double val) {
	if (val == 0.0) 
		return 0.0;
	return (0.0 < val) - (val < 0.0);
}

template <> 
inline std::complex<double> sgn(std::complex<double> val) {
	if (val == std::complex<double>(0.0)) 
		return std::complex<double>(0.0, 0.0);
	return val / std::abs(val);
}

template <> 
inline std::complex<float> sgn(std::complex<float> val) {
	if (val == std::complex<float>(0.0)) 
		return std::complex<float>(0.0, 0.0);
	return val / std::abs(val);
}

template <> 
inline std::complex<long double> sgn(std::complex<long double> val) {
	if (val == std::complex<long double>(0.0)) 
		return std::complex<long double>(0.0, 0.0);
	return val / std::abs(val);
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

template <typename _T>
typename std::enable_if<std::is_integral<_T>::value, _T>::type
modEUC(_T a, _T b);

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

namespace algebra {
	// ################################################################## CONJUGATE #####################################################################

	template <typename _T>
	inline auto conjugate(_T x)														-> _T		{ return std::conj(x); };
	template <>
	inline auto conjugate(double x)													-> double	{ return x; };
	template <>
	inline auto conjugate(float x)													-> float	{ return x; };
	template <>
	inline auto conjugate(int x)													-> int		{ return x; };

	template <typename _T>
	inline auto real(_T x)															-> double	{ return std::real(x); };
	template <>
	inline auto real(double x)														-> double	{ return x; };

	template <typename _T>
	inline auto imag(_T x)															-> double	{ return std::imag(x); };
	template <>
	inline auto imag(double x)														-> double	{ return 0.0; };
	
	template <typename _T>
	inline auto norm(_T x)															-> double	{ return algebra::real(x * algebra::conjugate(x)); };	
	template <>
	inline auto norm(double x)														-> double	{ return x * x; };
	template <typename _T, typename ... _Ts>
	inline auto norm(_T x, _Ts... y)												-> double	{ return algebra::norm(x) + algebra::norm(y...); };
	template <typename ..._Ts>
	inline auto norm(_Ts... y)														-> double	{ return algebra::norm(y...); };
	
	// -----------------------------------------------------------------------------------------------------------------------------------------
    /** 
    * @brief Check the maximum value of a set of values of a given type.
    * @param x first value
    * @param y... rest of the values
    * @returns maximum value
    */
    template <typename _T, typename... _Ts>
    inline auto maximum(_T x, _Ts... y) -> double
    {
        if constexpr (std::is_same_v<_T, std::complex<double>>) {
            return std::max({std::real(x), std::real(y)...});
        } else {
            return std::max({x, y...});
        }
    }
    
    /**
    * @brief Check the minimum value of a set of values of a given type.
    * @param x first value
    * @param y... rest of the values
    * @returns minimum value
    */
    template <typename _T, typename... _Ts>
    inline auto minimum(_T x, _Ts... y) -> double
    {
        if constexpr (std::is_same_v<_T, std::complex<double>>) {
            return std::min({std::real(x), std::real(y)...});
        } else {
            return std::min({static_cast<double>(x), static_cast<double>(y)...});
        }
    }

	// -----------------------------------------------------------------------------------------------------------------------------------------

	template <typename _T, typename _T2>
	inline bool gr(_T x, _T2 y)														{ return x > y; };
	template <typename _T>
	inline bool gr(_T x, _T y)														{ return x > y; };
	template <>
	inline bool gr(std::complex<double> x, std::complex<double> y)					{ return std::real(x) > std::real(y); };
	template <>
	inline bool gr(double x, std::complex<double> y)								{ return x > std::real(y); };
	template <>
	inline bool gr(std::complex<double> x, double y)								{ return std::real(x) > y; };

	template <typename _T1, typename _T2>
	inline bool ls(_T1 x, _T2 y)													{ return x > y; };
	template <typename _T>
	inline bool ls(_T x, _T y)														{ return x < y; };
	template <>
	inline bool ls(std::complex<double> x, std::complex<double> y)					{ return std::real(x) < std::real(y); };
	template <>
	inline bool ls(double x, std::complex<double> y)								{ return x < std::real(y); };
	template <>
	inline bool ls(std::complex<double> x, double y)								{ return std::real(x) < y; };

	template <typename _T1, typename _T2>
	inline bool eq(_T1 x, _T2 y)													{ return x == y; };
	template <typename _T>
	inline bool eq(_T x, _T y)														{ return x == y; };
	template <>
	inline bool eq(std::complex<double> x, std::complex<double> y)					{ return std::abs(x - y) < 1e-10; };
	template <>
	inline bool eq(double x, std::complex<double> y)								{ return std::abs(x - std::real(y)) < 1e-10; };
	template <>
	inline bool eq(std::complex<double> x, double y)								{ return std::abs(std::real(x) - y) < 1e-10; };
	
	template <typename _T1, typename _T2>
	inline bool neq(_T1 x, _T2 y)													{ return x != y; };
	template <typename _T>
	inline bool neq(_T x, _T y)														{ return x != y; };
	template <>
	inline bool neq(std::complex<double> x, std::complex<double> y)					{ return std::abs(x - y) > 1e-10; };
	template <>
	inline bool neq(double x, std::complex<double> y)								{ return std::abs(x - std::real(y)) > 1e-10; };
	template <>
	inline bool neq(std::complex<double> x, double y)								{ return std::abs(std::real(x) - y) > 1e-10; };

	template <typename _T1, typename _T2>
	inline bool geq(_T1 x, _T2 y)													{ return x >= y; };
	template <typename _T>
	inline bool geq(_T x, _T y)														{ return x >= y; };
	template <>
	inline bool geq(std::complex<double> x, std::complex<double> y)					{ return std::real(x) >= std::real(y); };
	template <>
	inline bool geq(double x, std::complex<double> y)								{ return x >= std::real(y); };
	template <>
	inline bool geq(std::complex<double> x, double y)								{ return std::real(x) >= y; };

	template <typename _T1, typename _T2>
	inline bool leq(_T1 x, _T2 y)													{ return x <= y; };
	template <typename _T>
	inline bool leq(_T x, _T y)														{ return x <= y; };
	template <>
	inline bool leq(std::complex<double> x, std::complex<double> y)					{ return std::real(x) <= std::real(y); };
	template <>
	inline bool leq(double x, std::complex<double> y)								{ return x <= std::real(y); };
	template <>
	inline bool leq(std::complex<double> x, double y)								{ return std::real(x) <= y; };
	
	// ###################################################################### CAST #####################################################################

	template <typename _T, typename _Tin>
	inline auto cast(_Tin x)														-> _T								{ return static_cast<_T>(x); };
	template <typename _T>
	inline auto cast(std::complex<double> x)										-> _T								{ return x; };
	template <>
	inline auto cast<double>(std::complex<double> x)								-> double							{ return std::real(x); };

};


#endif