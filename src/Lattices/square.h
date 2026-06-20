/******************************************************************************
 *
 *  @file src/lattices/square.h
 *  @brief Square lattice class declaration (2D Bravais / 1D chain / 3D cubic).
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef LATTICE_H
#include "lattices.h"
#endif

#ifndef SQUARE_H
#define SQUARE_H

class SquareLattice : public Lattice 
{
private:
	bool symmetry	= false;
	int Lx			= 1;
	int Ly			= 1;
	int Lz			= 1;

	double a		= 1;
	double b		= 1;
	double c		= 1;

public:
	// CONSTRUCTORS
	~SquareLattice()
	{
		LOGINFOG(this->get_info() + " is destroyed.", LOG_TYPES::INFO, 3);
	}
	SquareLattice() = default;
	SquareLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);

	// GETTERS
	arma::vec getRealVec(int x, int y, int z) const override { return { a * x, b * y, c * z }; };
	uint get_SitesPerCell() const override { return 1; };
	int get_Lx() const override { return this->Lx; };
	int get_Ly() const override { return this->Ly; };
	int get_Lz() const override { return this->Lz; };
	int getNorm(int x, int y, int z) const override { return this->spatialNorm[x][y][z]; };
	int get_nn(int lat_site, direction d) const override;

	// FORWARD GETTERS
	v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint> p) const override { return this->nnForward; };
	v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint> p) const override { return this->nnnForward; };
	uint get_nn_ForwardNum(int site, int num) const override { return this->nnForward[num]; };
	uint get_nnn_ForwardNum(int site, int num) const override { return this->nnnForward[num]; };

	// CALCULATORS
	void calculate_nn(bool pbcx, bool pbcy, bool pbcz) override final;
	void calculate_nnn(bool pbcx, bool pbcy, bool pbcz) override final;
	void calculate_coordinates() override;

	// SYMMETRIES
	std::tuple<int, int, int> getNumElems() override {
		if (!this->symmetry)
			return std::make_tuple(2 * this->Lx - 1, 2 * this->Ly - 1, 2 * this->Lz - 1);

		switch (this->_BC)
		{
		case 0:
			return std::make_tuple(this->Lx / 2, this->Ly / 2, this->Lz / 2);
		default:
			return std::make_tuple(this->Lx, this->Ly, this->Lz);
		}
	}

	t_3d<int> get_sym_pos(int x, int y, int z) override {
		if (!this->symmetry)
			return std::make_tuple(x + Lx - 1, y + Ly - 1, z + Lz - 1);
		else
			return std::make_tuple(x, y, z);
	}

	t_3d<int> get_sym_pos_inv(int x, int y, int z) override {
		if (!this->symmetry)
			return std::make_tuple(x - (Lx - 1), y - (Ly - 1), z - (Lz - 1));
		else
			return std::make_tuple(x, y, z);
	}

	bool symmetry_checker(int xx, int yy, int zz) override {
		return
			(xx <= this->Lx / 2 && xx >= 0) &&
			(yy <= this->Ly / 2 && yy >= 0) &&
			(zz <= this->Lz / 2 && zz >= 0);
	};

private:
	void calculate_kVec() override;
	void calculate_rVec() override;
};

#endif // !SQUARE_H
