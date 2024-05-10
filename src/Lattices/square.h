#pragma once
#ifndef LATTICE_H
#include "../lattices.h"
#endif // !LATTICE_H

// -------------------------------------------------------- SQUARE LATTICE --------------------------------------------------------

#ifndef SQUARE_H
#define SQUARE_H
class SquareLattice : public Lattice 
{
private:
	bool symmetry	= false;		// if we shall include symmetry in saving greens
	int Lx			= 1;			// spatial x-length
	int Ly			= 1;			// spatial y-length
	int Lz			= 1;			// spatial z-length

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
	SquareLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);							// general constructor

	// GETTERS
	arma::vec getRealVec(int x, int y, int z)				const override { return { a * x, b * y, c * z }; };
	int get_Lx()											const override { return this->Lx; };
	int get_Ly()											const override { return this->Ly; };
	int get_Lz()											const override { return this->Lz; };
	int getNorm(int x, int y, int z)						const override { return this->spatialNorm[x][y][z]; };
	int get_nn(int lat_site, direction d)					const override;

	// ----------------------- GETTERS NEI
	v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint> p)	const override { return this->nnForward; };
	v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint> p)	const override { return this->nnnForward; };
	uint get_nn_ForwardNum(int site, int num)				const override { return this->nnForward[num]; };
	uint get_nnn_ForwardNum(int site, int num)				const override { return this->nnnForward[num]; };

	// ----------------------- CALCULATORS
	// --- nn ---
	void calculate_nn_pbc() override;
	void calculate_nn_obc() override;
	void calculate_nn_mbc() override;
	void calculate_nn_sbc() override;
	// --- nnn --- 
	void calculate_nnn_pbc() override;
	void calculate_nnn_obc() override;
	// --- coords --- 
	void calculate_coordinates() override;

	// ----------------------- SYMMETRIES

	/*
	* @brief Get the number of elements in the lattice for each dimension
	*/
	std::tuple<int, int, int> getNumElems() override {
		if (!this->symmetry)
			return std::make_tuple(2 * this->Lx - 1, 2 * this->Ly - 1, 2 * this->Lz - 1);

		switch (this->_BC)
		{
		case 0:
			return std::make_tuple(this->Lx / 2, this->Ly / 2, this->Lz / 2);
			break;
		default:
			return std::make_tuple(this->Lx, this->Ly, this->Lz);
			break;
		}
	}

	/*
	* @brief Out of a given coordinates, get the value of element in bigger than zero form
	*/
	t_3d<int> getSymPos(int x, int y, int z) override {
		if (!this->symmetry)
			return std::make_tuple(x + Lx - 1, y + Ly - 1, z + Lz - 1);
		else
			return std::make_tuple(x, y, z);
	}

	/*
	* @brief Out of a given bigger than zero coordinates, 
	* get the value of element in bigger than zero form
	*/
	t_3d<int> getSymPosInv(int x, int y, int z) override {
		if (!this->symmetry)
			return std::make_tuple(x - (Lx - 1), y - (Ly - 1), z - (Lz - 1));
		else
			return std::make_tuple(x, y, z);
	}

	/*
	* @brief Check the lattice symmetry around zero for the square lattice
	* @param xx - symmetrized form of x
	* @param yy - symmetrized form of y
	* @param zz - symmetrized form of z
	*/
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