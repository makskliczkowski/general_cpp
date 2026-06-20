/******************************************************************************
 *
 *  @file src/common/algorithms.h
 *  @brief Modern, optimized search and utility algorithms.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @copyright  : (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <cstddef>

/**
 * @brief Performs a binary search on a sorted container, returning the 0-based index.
 * @details Leverages std::lower_bound for compiler-optimized binary search.
 * @param container The sorted container to search.
 * @param low Starting index of the search range.
 * @param high Ending index of the search range (inclusive).
 * @param elem The element to search for.
 * @returns Index of the element if found, otherwise -1.
 */
template <typename Container, typename T>
inline long long binarySearch(const Container& arr, std::size_t low, std::size_t high, const T& elem) 
{
    if (std::empty(arr) || low > high || high >= std::size(arr))
        return -1;

    auto first  = std::begin(arr) + low;
    auto last   = std::begin(arr) + high + 1;
    auto it     = std::lower_bound(first, last, elem);

    if (it != last && *it == elem)
        return std::distance(std::begin(arr), it);

    return -1;
}

/**
 * @brief Overload for binarySearch on the entire sorted container.
 * @param container The sorted container to search.
 * @param elem The element to search for.
 * @returns Index of the element if found, otherwise -1.
 */
template <typename Container, typename T>
inline long long binarySearch(const Container& arr, const T& elem) 
{
    if (std::empty(arr))
        return -1;
    return binarySearch(arr, 0, std::size(arr) - 1, elem);
}

// ------------------------------------------------------------------------------
// Overloads for binarySearchDouble on the entire sorted container.
// ------------------------------------------------------------------------------

/**
 * @brief Specialization of binarySearch for double elements using a tolerance.
 * @param container The sorted container to search.
 * @param low Starting index of the search range.
 * @param high Ending index of the search range (inclusive).
 * @param elem The element to search for.
 * @param tolerance The tolerance for floating-point comparison.
 * @returns Index of the element if found, otherwise -1.
 */
template <typename Container>
inline long long binarySearchDouble(const Container& arr, std::size_t low, std::size_t high, double elem, double tolerance = 1e-12) 
{
    
    if (std::empty(arr) || low > high || high >= std::size(arr))
        return -1;

    auto first  = std::begin(arr) + low;
    auto last   = std::begin(arr) + high + 1;
    auto it     = std::lower_bound(first, last, elem - tolerance);
    
    if (it != last && std::abs(*it - elem) <= tolerance)
        return std::distance(std::begin(arr), it);
    
    return -1;
}


/**
 * @brief Overload for binarySearchDouble on the entire sorted container.
 * @tparam Container The type of the container to search.
 * @param arr The sorted container to search.
 * @param elem The element to search for.
 * @param tolerance The tolerance for floating-point comparison.
 * @returns Index of the element if found, otherwise -1.
 */
template <typename Container>
inline long long binarySearchDouble(const Container& arr, double elem, double tolerance = 1e-12) 
{
    if (std::empty(arr))
        return -1;
    return binarySearchDouble(arr, 0, std::size(arr) - 1, elem, tolerance);
}

template <typename Container>
inline long long binarySearch(const Container& arr, std::size_t low, std::size_t high, double elem) 
{
    return binarySearchDouble(arr, low, high, elem);
}

template <typename Container>
inline long long binarySearch(const Container& arr, double elem) 
{
    return binarySearchDouble(arr, elem);
}

#endif // !ALGORITHMS_H

// -----------------------------------------------------------------------------
//! EOF
// -----------------------------------------------------------------------------
