/*
* The implementation of the honeycomb lattice is based on the Lattice class.
* The honeycomb lattice is a 2D lattice with a hexagonal structure. The lattice
* has two sublattices, A and B, with the A sublattice sites at the center of the
* hexagons and the B sublattice sites at the vertices of the hexagons.
* The honeycomb lattice has two nearest neighbor vectors, nn1 and nn2, and
* four next nearest neighbor vectors, nnn1, nnn2, nnn3, and nnn4.
* It is based upon ladder lattice with the following structure:
*      A B A B A B
*     B A B A B A
*    A B A B A B
*   B A B A B A
*  A B A B A B
* B A B A B A
* @url Check the geometry from Fig. 2 of the paper https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.013160
*/

#ifndef LATTICE_H
    #include "../lattices.h"
#endif

#ifndef HONEYCOMB_H
#define HONEYCOMB_H

// ####################################################################################################

class Honeycomb : public Lattice {
private:
    int Lx, Ly, Lz;

    // lattice parameters
    double a = 1, c = 1;    // lattice parameters - a and c define the lattice vectors as a(1,0) and c(1/2,sqrt(3)/2)
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

    // GETTERS NEIGHBORS
	v_1d<uint> get_nn_ForwardNum(int site, v_1d<uint>)	const override { if (this->dim == 1 || site % 2 == 0) return { 0 }; else return { 1, 2 }; };
	v_1d<uint> get_nnn_ForwardNum(int site, v_1d<uint>)	const override { if (this->dim == 1 || site % 2 == 0) return { 0 }; else return { 1, 2 }; };
	uint get_nn_ForwardNum(int site, int num)			const override { return this->nnForward[num]; };
	uint get_nnn_ForwardNum(int site, int num)			const override { return this->nnnForward[num]; };

    // CALCULATORS
    void calculate_nn(bool pbcx, bool pbcy, bool pbcz);
    void calculate_nnn(bool pbcx, bool pbcy, bool pbcz)                 {};
    // --- nn ---
    void calculate_nn_pbc()                             override final  { this->calculate_nn(true, true, true);     };
    void calculate_nn_obc()                             override final  { this->calculate_nn(false, false, false);  };
    void calculate_nn_mbc()                             override final  { this->calculate_nn(true, false, false);   };
    void calculate_nn_sbc()                             override final  { this->calculate_nn(false, true, false);   };
    // --- nnn ---
    void calculate_nnn_pbc()                            override final  { this->calculate_nnn(true, true, true);     };
    void calculate_nnn_obc()                            override final  { this->calculate_nnn(false, false, false);  };
    // --- coords ---
    void calculate_coordinates()                        override final;

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
#endif // !HONEYCOMB_H
