/******************************************************************************
 *
 *  @file src/lattices/honeycomb.h
 *  @brief Honeycomb lattice class declaration.
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

#ifndef HONEYCOMB_H
#define HONEYCOMB_H

class Honeycomb : public Lattice 
{
private:
    int Lx, Ly, Lz;

    double a = 1, c = 1;
public:
    ~Honeycomb()            { LOGINFOG(this->get_info() + " is destroyed.", LOG_TYPES::DEBUG, 3); }
    Honeycomb()             = default;
    Honeycomb(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0); 

    // GETTERS
    int get_Lx()                                        const override { return this->Lx; };
    int get_Ly()                                        const override { return this->Ly; };
    int get_Lz()                                        const override { return this->Lz; };
    int getNorm(int x, int y, int z)                    const override { return this->spatialNorm[x][y][z]; };
    int get_nn(int _site, direction d)                  const override;
    arma::vec getRealVec(int x, int y, int z)           const override;
    uint get_Sublattice(uint site)                      const override { return site % 2; };
    uint get_SitesPerCell()                             const override { return 2; };

    // GETTERS NEIGHBORS
	v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint>)	const override { if (this->dim == 1 || site % 2 == 0) return { 0 }; else return { 1, 2 }; };
	v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint>)	const override { if (this->dim == 1 || site % 2 == 0) return { 0 }; else return { 1, 2 }; };
	uint get_nn_ForwardNum(int site, int num)			const override { return this->nnForward[num]; };
	uint get_nnn_ForwardNum(int site, int num)			const override { return this->nnnForward[num]; };

    // CALCULATORS
    void calculate_nn(bool pbcx, bool pbcy, bool pbcz)  override final;
    void calculate_nnn(bool pbcx, bool pbcy, bool pbcz) override final {};
    void calculate_coordinates()                        override final;

    // SYMMETRIES
	t_3d<int> getNumElems() override 
	{
		return std::make_tuple(2 * this->Lx - 1, 4 * this->Ly - 1, 2 * this->Lz - 1);
	}

	t_3d<int> get_sym_pos(int x, int y, int z) override 
	{
		return std::make_tuple(x + Lx - 1, y + 2 * Ly - 1, z + Lz - 1);
	}

	t_3d<int> get_sym_pos_inv(int x, int y, int z) override 
	{
		return std::make_tuple(x - (Lx - 1), y - (2 * Ly - 1), z - (Lz - 1));
	}

	bool symmetry_checker(int xx, int yy, int zz) override 
	{
		return true;
	};
private:
	void calculate_kVec() override;
	void calculate_rVec() override;
public:
    v_1d<uint> get_flux_sites(int X, int Y, int Z) const override final;
};
#endif // !HONEYCOMB_H
