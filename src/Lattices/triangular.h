/******************************************************************************
 *
 *  @file src/lattices/triangular.h
 *  @brief Triangular lattice class declaration (2D Bravais).
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef TRIANGULAR_H
#define TRIANGULAR_H

#ifndef LATTICE_H
#include "lattices.h"
#endif

class TriangularLattice : public Lattice 
{
private:
	int Lx = 1;
	int Ly = 1;
	int Lz = 1;

public:
	~TriangularLattice()
	{
		LOGINFOG(this->get_info() + " is destroyed.", LOG_TYPES::INFO, 3);
	}
	TriangularLattice() = default;
	TriangularLattice(int Lx, int Ly = 1, int _BC = 0);

	// GETTERS
	arma::vec getRealVec(int x, int y, int z) const override {
		return x * this->a1 + y * this->a2 + z * this->a3;
	}
	uint get_SitesPerCell() const override { return 1; };
	int get_Lx() const override { return this->Lx; }
	int get_Ly() const override { return this->Ly; }
	int get_Lz() const override { return this->Lz; }
	int getNorm(int x, int y, int z) const override { return this->spatialNorm[x][y][z]; }
	int get_nn(int site, direction d) const override;

	// FORWARD GETTERS
	v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint> p) const override { return this->nnForward; }
	v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint> p) const override { return this->nnnForward; }
	uint get_nn_ForwardNum(int site, int num) const override { return this->nnForward[num]; }
	uint get_nnn_ForwardNum(int site, int num) const override { return this->nnnForward[num]; }

	// CALCULATORS
	void calculate_nn(bool pbcx, bool pbcy, bool pbcz) override;
	void calculate_nnn(bool pbcx, bool pbcy, bool pbcz) override;
	void calculate_coordinates() override;

	// SYMMETRIES (Default / basic 2D Bravais implementation)
	std::tuple<int, int, int> getNumElems() override {
		return std::make_tuple(2 * this->Lx - 1, 2 * this->Ly - 1, 2 * this->Lz - 1);
	}
	t_3d<int> get_sym_pos(int x, int y, int z) override {
		return std::make_tuple(x + Lx - 1, y + Ly - 1, z + Lz - 1);
	}
	t_3d<int> get_sym_pos_inv(int x, int y, int z) override {
		return std::make_tuple(x - (Lx - 1), y - (Ly - 1), z - (Lz - 1));
	}
	bool symmetry_checker(int xx, int yy, int zz) override {
		return true;
	}

private:
	void calculate_kVec() override;
	void calculate_rVec() override;
};

#endif // !TRIANGULAR_H
