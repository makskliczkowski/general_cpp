/******************************************************************************
 *
 *  @file src/lattices/lattice_symmetry.h
 *  @brief Lattice symmetry group permutations and Cayley tables.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @details Generates point, translation and space groups for various lattices.
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef LATTICE_SYMMETRY_H
#define LATTICE_SYMMETRY_H

#include <vector>
#include <cstdint>
#include <cstddef>

namespace LatticeSym
{
	using Perm = std::vector<uint32_t>;

	std::vector<Perm> translation_perms(std::size_t Lx, std::size_t Ly, std::size_t sites_per_cell);
	std::vector<Perm> point_group_perms_square(std::size_t Lx, std::size_t Ly);
	std::vector<Perm> space_group_perms(std::size_t Lx, std::size_t Ly, std::size_t sites_per_cell, bool full_point_group);
	std::vector<std::vector<uint32_t>> cayley_table(const std::vector<Perm>& perms);
}

#endif // !LATTICE_SYMMETRY_H
