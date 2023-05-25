#pragma once
#include <cmath>
#include <complex>

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
*/
template <typename _T>
inline _T toType(double _r, double _i) {
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

// #################################################################

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