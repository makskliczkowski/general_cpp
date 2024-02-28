#pragma once
#include <cmath>
#include <complex>
// matrix base class concepts

#ifdef __has_include
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
								std::is_base_of<unsigned long long, _T>::value					||
								std::is_base_of<uint_fast16_t, _T>::value						||
								std::is_base_of<uint_fast32_t, _T>::value;
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

namespace MATH
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

}