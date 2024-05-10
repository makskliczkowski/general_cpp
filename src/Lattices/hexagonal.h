#pragma once
#ifndef LATTICE_H
#include "../lattices.h"
#endif // !LATTICE_H


// -------------------------------------------------------- HEXAGONAL LATTICE --------------------------------------------------------
#ifndef HEXAGONAL_H
#define HEXAGONAL_H

class HexagonalLattice : public Lattice {
private:
	// elementary cells numbering
	int Lx;																												// spatial x-length
	int Ly;																												// spatial y-length
	int Lz;																												// spatial z-length

	// lattice parameters
	double a = 1;
	double c = 1;


public:
	// CONSTRUCTORS
	~HexagonalLattice() {
		LOGINFOG(this->get_info() + " is destroyed.", LOG_TYPES::INFO, 3);
	}
	HexagonalLattice() = default;
	HexagonalLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);											// general constructor

	// GETTERS
	int get_Lx()														const override { return this->Lx; };
	int get_Ly()														const override { return this->Ly; };
	int get_Lz()														const override { return this->Lz; };
	int getNorm(int x, int y, int z)								const override { return this->spatialNorm[x][y][z]; };
	int get_nn(int lat_site, direction d)						const override;
	arma::vec getRealVec(int x, int y, int z)					const override;

	// ----------------------- GETTERS NEI
	v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint>)		const override { if (this->dim == 1 || site % 2 == 0) return { 0 }; else return { 1, 2 }; };
	v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint>)	const override { if (this->dim == 1 || site % 2 == 0) return { 0 }; else return { 1, 2 }; };
	uint get_nn_ForwardNum(int site, int num)					const override { return nnForward[num]; };
	uint get_nnn_ForwardNum(int site, int num)				const override { return nnnForward[num]; };

	// CALCULATORS
	// --- nn --- 
	void calculate_nn_pbc()											override final;
	void calculate_nn_obc()											override final;
	void calculate_nn_mbc()											override final;
	void calculate_nn_sbc()											override final;
	// --- nnn --- 
	void calculate_nnn_pbc()										override final;
	void calculate_nnn_obc()										override final;
	// --- coords --- 
	void calculate_coordinates()									override final;

	// SYMMETRIES
	t_3d<int> getNumElems() override 
	{
		return std::make_tuple(2 * this->Lx - 1, 4 * this->Ly - 1, 2 * this->Lz - 1);
	}

	t_3d<int> getSymPos(int x, int y, int z) override 
	{
		return std::make_tuple(x + Lx - 1, y + 2 * Ly - 1, z + Lz - 1);
	}

	t_3d<int> getSymPosInv(int x, int y, int z) override 
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
};


#endif // ! HEXAGONAL_H