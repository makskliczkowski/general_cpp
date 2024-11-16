#include "../src/Include/maths.h"


// #################################################################################################################################################

/*
* @brief Defines an euclidean modulo denoting also the negative sign
* @param a left side of modulo
* @param b right side of modulo
* @return euclidean a % b
* @link https://en.wikipedia.org/wiki/Modulo_operation
*/
template <typename _T>
typename std::enable_if<std::is_integral<_T>::value, _T>::type
modEUC(_T a, _T b)
{
    _T m = a % b;
    if (m < 0) m = (b < 0) ? m - b : m + b;
    return m;
}
// template specializations 
template int modEUC(int a, int b);
template long modEUC(long a, long b);
template long long modEUC(long long a, long long b);
template unsigned int modEUC(unsigned int a, unsigned int b);
template unsigned long modEUC(unsigned long a, unsigned long b);
template unsigned long long modEUC(unsigned long long a, unsigned long long b);

// #################################################################################################################################################

/**
* @brief Algebraic operations and functions for the library...
*/
namespace algebra
{



    // #################################################################################################################################################

};