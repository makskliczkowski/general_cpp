/*******************************
* Contains the possible methods
* for general lattice class.
*******************************/

#ifndef COMMON_H
	#include "common.h"
#endif

#ifndef LATTICE_H
#define LATTICE_H

// ########################################################	GENERAL LATTICE ########################################################

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
enum LatticeTypes { SQ, HEX, HON };						//%
														//%
BEGIN_ENUM(LatticeTypes)								//%
{														//%
	DECL_ENUM_ELEMENT(SQ),								//%
	DECL_ENUM_ELEMENT(HEX)								//%
	DECL_ENUM_ELEMENT(HON)
}														//%
END_ENUM(LatticeTypes);									//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
enum BoundaryConditions {	PBC = 0, OBC = 1,			//%
							MBC = 2, SBC = 3 };			//%
BEGIN_ENUM(BoundaryConditions)							//%
{														//%
	DECL_ENUM_ELEMENT(PBC),								//%
	DECL_ENUM_ELEMENT(OBC),								//%
	DECL_ENUM_ELEMENT(MBC),								//%
	DECL_ENUM_ELEMENT(SBC)								//%
}														//%
END_ENUM(BoundaryConditions);							//%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Pure virtual lattice class, it will allow to distinguish between different geometries in the models
*/
class Lattice {
protected:
	// ----------------------- LATTICE PARAMETERS
	BoundaryConditions _BC	= BoundaryConditions::PBC;		// boundary conditions 0 = PBC, 1 = OBC, 2 = MBC [PBC->x;OBC->y, OBC->z], 3 = SBC [OBC->x, PBC->y, OBC->z],
	LatticeTypes type_		= LatticeTypes::SQ;				// enum type of the lattice
	std::string type		= "";							// type of the lattice
	
	unsigned int dim		= 1;							// the dimensionality of the lattice 1,2,3
	unsigned int Ns			= 1;							// number of lattice sites 
	
	// --- nn --- 
	v_2d<int> nn;											// vector of the nearest neighbors (for a given site, it creates a vector of all nn)
	v_2d<int> nnF;											// vector of the nearest neighbors (for a given site, it creates a vector of all nn) - forward only
	v_1d<uint> nnForward;									// number of nearest neighbors forward (not to include all the connections twice)
	
	// --- nnn --- 
	v_2d<int> nnn;											// vector of the next nearest neighbors (for a given site, it creates a vector of all nnn)
	v_2d<int> nnnF;											// vector of the next nearest neighbors (for a given site, it creates a vector of all nnn) - forward only
	v_1d<uint> nnnForward;									// number of nearest neighbors forward (not to include all the connections twice)
	
	// --- coords ---
	v_2d<int> coord;										// vector of real coordiates allowing to get the distance between lattice points
	v_3d<int> spatialNorm;									// norm for averaging over all spatial sites

	// reciprocal vectors
	arma::vec a1, a2, a3;									// base vectors of the lattice - to be used in the derived classes
	arma::vec b1, b2, b3;									// reciprocal vectors of the lattice - to be used in the derived classes
	arma::mat kVec;											// allowed values of k - to be used in the lattice derived classes
	arma::mat rVec;											// allowed values of r - to be used in the lattice derived classes
	arma::Mat<cpx> dft_;									// DFT matrix
public:
	enum direction { X, Y, Z };

	virtual ~Lattice() 										{ LOGINFOG("General lattice is destroyed.", LOG_TYPES::DEBUG, 3); };

	// ---------------------- VIRTUAL GETTERS ----------------------
	virtual int get_Lx()									const = 0;
	virtual int get_Ly()									const = 0;
	virtual int get_Lz()									const = 0;
	auto getSiteDifference(t_3d<int> i, uint j)				const ->t_3d<int>;
	auto getSiteDifference(uint i, uint j)					const ->t_3d<int>;
	auto getSiteDistance(uint i, uint j)					const -> double;

	// -------------------------- GETTERS --------------------------
	virtual arma::vec getRealVec(int x, int y, int z)		const = 0;
	virtual int getNorm(int x, int y, int z)				const = 0;

	// -------------------------- FORWARD
	virtual uint get_nn_ForwardNum(int site, int num)		const = 0;
	virtual uint get_nnn_ForwardNum(int site, int num)		const = 0;
	virtual v_1d<uint> get_nn_ForwardNum(int, v_1d<uint>)	const = 0;													// with placeholder returns vector of nn
	virtual v_1d<uint> get_nnn_ForwardNum(int, v_1d<uint>)	const = 0;													// with placeholder returns vector of nnn
	virtual int get_nn(int site, direction d)				const = 0;													// retruns nn in a given direction x 
	virtual int get_nnf(int site, int n) 					const { return this->nnF[site][n]; };						// returns the forward nn
	
	// ------------------------ GETTERS NEI ------------------------
	auto get_nn_ForwardNum(int site)						const -> uint	{ return (uint)this->nnForward.size(); };	// with no placeholder returns number of nn
	auto get_nnn_ForwardNum(int site)						const -> uint	{ return (uint)this->nnnForward.size(); };	// with no placeholder returns number of nnn
	auto get_nn(int site, int nei_num)						const -> int	{ return this->nn[site][nei_num]; };		// returns given nearest nei at given lat site
	auto get_nnn(int site, int nei_num)						const -> int	{ return this->nnn[site][nei_num]; };		// returns given next nearest nei at given lat site
	auto get_nn(int site)									const -> uint	{ return (uint)this->nn[site].size(); };	// returns the number of nn
	auto get_nnn(int site)									const -> uint  	{ return (uint)this->nnn[site].size(); };	// returns the number of nnn
	auto get_nei(int lat_site, int corr_len)				const -> int;

	// ----------------------- GETTERS OTHER -----------------------
	const arma::Mat<cpx>& get_DFT()							const	{ return this->dft_; };								// returns the DFT matrix
	BoundaryConditions get_BC()								const	{ return this->_BC; };								// returns the boundary conditions
	LatticeTypes get_Type()									const	{ return this->type_; };							// returns the type of the lattice as a string
	std::string get_type()									const	{ return this->type; };								// returns the type of the lattice as a string

	arma::mat get_kVec()									const	{ return this->kVec; };								// returns all k vectors in the RBZ
	arma::subview_row<double> get_kVec(uint row)					{ return this->kVec.row(row); };					// returns the given k vector row

	arma::mat get_rVec()									const	{ return this->rVec; };								// returns all r vectors in the RBZ
	arma::subview_row<double> get_rVec(uint row)					{ return this->rVec.row(row); };					// returns the given r vector row

	v_3d<int> get_spatial_norm()							const	{ return this->spatialNorm; };						// returns the spatial norm
	auto get_spatial_norm(int x, int y, int z)				const -> int { return this->spatialNorm[x][y][z]; };		// returns the spatial norm at x,y,z
	auto get_coordinates(int site, direction axis)			const -> int { return this->coord[site][axis]; };			// returns the given coordinate
	auto get_Ns()											const -> uint { return this->Ns; };							// returns the number of sites
	auto get_Dim()											const -> uint { return this->dim; };						// returns dimension of the lattice
	auto get_info()											const -> std::string;

	// ----------------------- CALCULATORS -----------------------
	void calculate_nn();
	void calculate_nnn();
	void calculate_spatial_norm();
	
    // CALCULATORS
    virtual void calculate_nn(bool pbcx, bool pbcy, bool pbcz)		{};
    virtual void calculate_nnn(bool pbcx, bool pbcy, bool pbcz)     {};

	// ------ nn ------
	virtual void calculate_nn_pbc()							{ this->calculate_nn(true, true, true); 	};
	virtual void calculate_nn_obc()							{ this->calculate_nn(false, false, false); 	};
	virtual void calculate_nn_mbc()							{ this->calculate_nn(true, false, false); 	};
	virtual void calculate_nn_sbc()							{ this->calculate_nn(false, true, false); 	};		
	
	// ------ nnn ------ 
	virtual void calculate_nnn_pbc()  						{ this->calculate_nnn(true, true, true); 	};
	virtual void calculate_nnn_obc()						{ this->calculate_nnn(false, false, false); };
	
	// ------ coords ------ 
	virtual void calculate_coordinates() = 0;

	// ------ others ------
	virtual void calculate_dft_matrix(bool phase = true);
	virtual arma::Mat<cpx> calculate_dft_vectors(bool phase = true);

	// ----------------------- SYMMETRY -----------------------
	virtual t_3d<int> getNumElems() = 0;																				// returns the number of elements if the symmetry is possible
	virtual t_3d<int> getSymPosInv(int x, int y, int z) = 0;															// from symmetrised form return coordinates
	virtual t_3d<int> getSymPos(int x, int y, int z) = 0;																// from given coordinates return their symmetrised form
	virtual bool symmetry_checker(int xx, int yy, int zz) = 0;

private:
	virtual void calculate_kVec() = 0;
	virtual void calculate_rVec() = 0;
	// ----------------------- TOPOLOGY -----------------------
public:
	virtual v_1d<uint> get_flux_sites(int X, int Y, int Z = 0) const { return {}; };									// returns the sites where the flux is applied
};

#endif // !LATTICE_H