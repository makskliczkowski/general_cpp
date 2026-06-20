/******************************************************************************
 *
 *  @file src/Include/bit_ops.h
 *  @brief Generic word-level bit operations on 64-bit integers.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @details Convention-free: functions act on bit indices only; any site-to-bit
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef BIT_OPS_H
#define BIT_OPS_H

/**
* @brief Generic word-level bit operations on 64-bit integers.
* @details Convention-free: functions act on bit indices only; any site-to-bit
* mapping is the caller's responsibility. 
*
* Header-only, STL-only, constexpr and noexcept throughout - usable in hot loops.
*/

#include <bit>
#include <cstddef>
#include <cstdint>

// -----------------------------------------------------------------------------

namespace BitOps
{
	// #########################################################################
	// Population counts and parity
	// #########################################################################

	/**
	* @brief Number of set bits in a word.
	* @param value input word
	* @returns popcount in [0, 64]
	*/
	[[nodiscard]] constexpr std::size_t popcount_64(uint64_t value) noexcept
	{
		return static_cast<std::size_t>(std::popcount(value));
	}

	/**
	* @brief Parity of a word (popcount mod 2).
	* @param value input word
	* @returns true when an odd number of bits is set
	*/
	[[nodiscard]] constexpr bool parity_64(uint64_t value) noexcept
	{
		return (std::popcount(value) & 1) != 0;
	}

	/**
	* @brief Whether exactly one bit is set (power of two).
	* Just checks if nonzero and popcount == 1, without branching.
	* Optimized to a single instruction on many platforms...
	* @param value input word
	* @returns true for single-bit words (false for zero)
	*/
	[[nodiscard]] constexpr bool has_single_bit_64(uint64_t value) noexcept
	{
		return std::has_single_bit(value);
	}

	// #########################################################################
	// Counting leading/trailing zeros and bit ranks
	// #########################################################################

	/**
	* @brief Index of the lowest set bit (count of trailing zeros).
	* @param value input word (must be nonzero)
	* @returns position in [0, 63]; 64 when value is zero
	* @note In binary 0b...0001 has 0 trailing zeros, 0b...0010 has 1, and 0 has 64.
	* Count from right to left...
	*/
	[[nodiscard]] constexpr std::size_t trailing_zeros_64(uint64_t value) noexcept
	{
		return static_cast<std::size_t>(std::countr_zero(value));
	}

	/**
	* @brief Number of leading zero bits (above the highest set bit).
	* @param value input word (must be nonzero)
	* @returns position in [0, 64]; 64 when value is zero
	* @note In binary 0x0000'0000'0000'0001 has 63 leading zeros, 0x8000'0000'0000'0000 has 0, and 0 has 64.
	* Count from left to right...
	*/
	[[nodiscard]] constexpr std::size_t leading_zeros_64(uint64_t value) noexcept
	{
		return static_cast<std::size_t>(std::countl_zero(value));
	}

	// #########################################################################
	// Bit popcounts in windows and ranks of set bits
	// #########################################################################

	/**
	* @brief Number of set bits strictly below index `bit` (bit in [0, 64]) - from right to left.
	* The Jordan-Wigner prefix sign source: popcount of value & low_mask(bit).
	* @param value input word
	* @param bit exclusive upper index of the counted window
	* @returns popcount of the low `bit` bits
	*/
	[[nodiscard]] constexpr std::size_t prefix_popcount_64(uint64_t value, std::size_t bit) noexcept
	{
		if (bit >= 64)
			return static_cast<std::size_t>(std::popcount(value));

		// move the mask to selected bit and subtract 1 to get low_mask(bit) = 0b...000111...111 with `bit` ones
		return static_cast<std::size_t>(std::popcount(value & ((uint64_t(1) << bit) - 1)));
	}

	/**
	* @brief Index of the `n`-th set bit (0-based), scanning from low to high.
	* @param value input word
	* @param n     rank of the wanted set bit
	* @returns bit index, or 64 when fewer than n+1 bits are set
	*/
	[[nodiscard]] constexpr std::size_t nth_set_bit_64(uint64_t value, std::size_t n) noexcept
	{
		while (value != 0)
		{
			const std::size_t bit = static_cast<std::size_t>(std::countr_zero(value));
			if (n == 0)
				return bit;
			--n;
			value &= value - 1;
		}
		return 64;
	}

	// #########################################################################
	// Bit masks
	// #########################################################################

	/**
	* @brief Mask with the lowest `bits` bits set (bits in [0, 64]).
	* @param bits number of low bits to set
	* @returns mask value, e.g. low_mask_64(4) = 0b1111
	*/
	[[nodiscard]] constexpr uint64_t low_mask_64(std::size_t bits) noexcept
	{
		if (bits >= 64)
			return ~uint64_t(0);
		return bits == 0 ? uint64_t(0) : ((uint64_t(1) << bits) - 1);
	}

	// #########################################################################
	// Bit reversal and rotation
	// #########################################################################

	/**
	* @brief Reverse all 64 bits (bit i -> bit 63 - i) via SWAR exchanges.
	* @param value input word
	* @returns bit-reversed word
	*/
	[[nodiscard]] constexpr uint64_t reverse_bits_64(uint64_t value) noexcept
	{
		// predefined masks for SWAR (SIMD Within A Register) bit-reversal algorithm
		// it just performs a sequence of parallel bit swaps: adjacent bits, then pairs, then nibbles, bytes, etc.
		value = ((value >>  1) & 0x5555555555555555ULL) | ((value & 0x5555555555555555ULL) <<  1);
		value = ((value >>  2) & 0x3333333333333333ULL) | ((value & 0x3333333333333333ULL) <<  2);
		value = ((value >>  4) & 0x0F0F0F0F0F0F0F0FULL) | ((value & 0x0F0F0F0F0F0F0F0FULL) <<  4);
		value = ((value >>  8) & 0x00FF00FF00FF00FFULL) | ((value & 0x00FF00FF00FF00FFULL) <<  8);
		value = ((value >> 16) & 0x0000FFFF0000FFFFULL) | ((value & 0x0000FFFF0000FFFFULL) << 16);
		return (value >> 32) | (value << 32);
	}

	/**
	* @brief Reverse the lowest `bits` bits (bit i -> bit bits - 1 - i).
	* Bits at index >= bits are discarded.
	* @param value input word
	* @param bits  active window width (1 <= bits <= 64)
	* @returns reversed window, canonical (high bits zero)
	*/
	[[nodiscard]] constexpr uint64_t reverse_bits_64(uint64_t value, std::size_t bits) noexcept
	{
		return reverse_bits_64(value) >> (64 - bits);
	}

	// #########################################################################
	// Bit rotation
	// #########################################################################

	/**
	* @brief Rotate the lowest `bits` bits toward higher indices:
	* bit i -> bit (i + shift) mod bits. Bits at index >= bits are discarded.
	* This is translation of bits in a circular buffer of width `bits`, so the shift is reduced mod bits.
	* @example 0x110011 rotated left by 2 bits with bits=6 gives 0x001111, as the two leftmost bits 11 are moved to the right end of the 6-bit window.
	* @param value input word
	* @param bits active window width (1 <= bits <= 64)
	* @param shift rotation distance (any value, reduced mod bits)
	* @returns rotated window, canonical (high bits zero)
	*/
	[[nodiscard]] constexpr uint64_t rotate_left_64(uint64_t value, std::size_t bits, std::size_t shift) noexcept
	{
		const uint64_t mask	= low_mask_64(bits);
		value				&= mask;
		shift				%= bits;
		if (shift == 0)
			return value;
		return ((value << shift) | (value >> (bits - shift))) & mask;
	}
} // namespace BitOps

#endif //! BIT_OPS_H

// -----------------------------------------------------------------------------
//! EOF
// -----------------------------------------------------------------------------
