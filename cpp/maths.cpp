#include "../src/Include/maths.h"

// #################################################################################################################################################

/**
* @brief Algebraic operations and functions for the library...
*/
namespace algebra
{

    /** 
    * @brief Check the maximum value of a set of values of a given type.
    * @param x first value
    * @param y... rest of the values
    * @returns maximum value
    */
    template <typename _T, typename... _Ts>
    auto maximum(_T x, _Ts... y) -> double
    {
        if constexpr (std::is_same_v<_T, std::complex<double>>) {
            return std::max({std::abs(x), std::abs(y)...});
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
    auto minimum(_T x, _Ts... y) -> double
    {
        if constexpr (std::is_same_v<_T, std::complex<double>>) {
            return std::max({std::abs(x), std::abs(y)...});
        } else {
            return std::max({static_cast<double>(x), static_cast<double>(y)...});
        }
    }

    // #################################################################################################################################################

};