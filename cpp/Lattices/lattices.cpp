/******************************************************************************
 *
 *  @file cpp/Lattices/lattices.cpp
 *  @brief Implementations for the base Lattice class methods.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/
#include "../../src/Lattices/lattices.h"
#include "../../src/algebra/lin_alg.h"

// ####################################################################################################

std::string Lattice::get_info() const
{
    std::string _inf;
    strSeparatedP(_inf, 
        ',', 
        3,
        this->type, 
        getSTR_BoundaryConditions(this->_BC), 
        VEQV(d, this->dim), 
        VEQ(Ns),
        VEQV(Lx, this->get_Lx()), 
        VEQV(Ly, this->get_Ly()), 
        VEQV(Lz, this->get_Lz()));
    return _inf;
}

// ####################################################################################################

/*
* @brief calculates the nearest neighbors
*/
void Lattice::calculate_nn() {
	switch (this->_BC)
	{
	case BoundaryConditions::PBC:
		this->calculate_nn_pbc();
		break;
	case BoundaryConditions::OBC:
		this->calculate_nn_obc();
		break;
	case BoundaryConditions::MBC:
		this->calculate_nn_mbc();
		break;
	case BoundaryConditions::SBC:
		this->calculate_nn_sbc();
		break;
	default:
		this->calculate_nn_pbc();
		break;
	}
	LOGINFOG("Created NN. Using: " + SSTR(getSTR_BoundaryConditions(this->_BC)), LOG_TYPES::INFO, 2);
}

// #################################################################################################### 

/*
* @brief calculates the next nearest neighbors
*/
void Lattice::calculate_nnn()
{
	switch (this->_BC)
	{
	case 0:
		this->calculate_nnn_pbc();
		break;
	case BoundaryConditions::OBC:
		this->calculate_nnn_obc();
		break;
	default:
		this->calculate_nnn_pbc();
		break;
	}
	LOGINFOG("Created NNN. Using: " + SSTR(getSTR_BoundaryConditions(this->_BC)), LOG_TYPES::INFO, 2);
}

// ####################################################################################################

/*
* @brief calculates the spatial repetition of difference between the lattice sites considering _BC and enumeration
*/
void Lattice::calculate_spatial_norm()
{
	// spatial norm
	auto [x_n, y_n, z_n]	= this->getNumElems();
	this->spatialNorm		= SPACE_VEC(x_n, y_n, z_n, int);

	// go through the lattice sites
	for (uint i = 0; i < this->Ns; i++) 
	{
		for (uint j = 0; j < this->Ns; j++) 
		{
			// calculate the coordinates of two site difference
			const auto [xx, yy, zz]		= this->getSiteDifference(i, j);
			auto [a, b, c]				= this->get_sym_pos(xx, yy, zz);
			spatialNorm[a][b][c]++;
		}
	}
}

// ####################################################################################################

/*
* @brief gets the neighbor from a given lat_site lattice site at corr_len length
*/
int Lattice::get_nei(int lat_site, int corr_len) const
{
	switch (this->_BC) 
	{
	case BoundaryConditions::PBC:
		return modEUC<int>(lat_site + corr_len, this->Ns);
		break;
	case BoundaryConditions::OBC:
		return uint(lat_site + corr_len) > this->Ns ? -1 : (lat_site + corr_len);
		break;
	default:
		return modEUC<int>(lat_site + corr_len, this->Ns);
	}
}

// ####################################################################################################

/**
* @brief Collect all forward bonds of the lattice.
* @details Iterates the forward neighbor table nnF; slot index becomes the
* bond type (square {0:+x, 1:+y, 2:+z}, honeycomb Kitaev {0:z, 1:y, 2:x}).
* Slots holding -1 (no forward bond, open boundary) are skipped.
* @returns vector of (from, to, type) forward bonds
*/
v_1d<LatticeBond> Lattice::get_bonds() const
{
	v_1d<LatticeBond> _bonds;
	_bonds.reserve(this->Ns * 2);

	for (uint i = 0; i < this->Ns; ++i)
	{
		if (i >= this->nnF.size())
			break;
		for (uint _slot = 0; _slot < this->nnF[i].size(); ++_slot)
		{
			const int _nei = this->nnF[i][_slot];
			if (_nei >= 0 && _nei < int(this->Ns))
				_bonds.push_back(LatticeBond{ i, uint(_nei), int(_slot) });
		}
	}
	return _bonds;
}

// ####################################################################################################

/*
* @brief Number of valid nearest neighbors of a site (the coordination number).
* Open-boundary missing neighbors (stored as -1) are not counted.
*/
uint Lattice::coordination_number(int _site) const
{
	if (_site < 0 || _site >= int(this->Ns) || uint(_site) >= this->nn.size())
		return 0;
	uint _count = 0;
	for (const int _nei : this->nn[_site])
		if (_nei >= 0 && _nei < int(this->Ns))
			++_count;
	return _count;
}

// ####################################################################################################

/*
* @brief Symmetric nearest-neighbor adjacency matrix A (Ns x Ns): A(i,j)=1 when
* sites i and j are nearest neighbors, else 0. Built from the forward bonds and
* symmetrized, so each undirected edge is set exactly once per direction.
*/
arma::SpMat<double> Lattice::adjacency_matrix() const
{
	arma::SpMat<double> _A(this->Ns, this->Ns);
	for (const auto& _bond : this->get_bonds())
	{
		_A(_bond.from, _bond.to) = 1.0;
		_A(_bond.to, _bond.from) = 1.0;
	}
	return _A;
}

// ####################################################################################################

/*
* @brief Sublattice index of every site (site = cell * sites_per_cell + sublattice).
* For a two-site unit cell this is the bipartite (A/B) class.
*/
v_1d<uint> Lattice::sublattice_partition() const
{
	v_1d<uint> _partition(this->Ns);
	for (uint _i = 0; _i < this->Ns; ++_i)
		_partition[_i] = this->get_Sublattice(_i);
	return _partition;
}

// ####################################################################################################

/*
* @brief Peierls flux phases for twisted boundary conditions. For each forward
* bond (i -> j, in get_bonds() order) that crosses the periodic boundary along
* an axis, the bond carries exp(i * flux_axis) (with the sign set by the
* wrap direction); bonds that do not cross a boundary carry phase 1. A bond
* crosses the boundary along an axis when the coordinate jump exceeds one cell.
*/
arma::cx_vec Lattice::boundary_flux_phases(double _flux_x, double _flux_y, double _flux_z) const
{
	const auto _bonds = this->get_bonds();
	arma::cx_vec _phases(_bonds.size(), arma::fill::ones);
	const double _flux[3] = { _flux_x, _flux_y, _flux_z };

	for (std::size_t _b = 0; _b < _bonds.size(); ++_b)
	{
		const uint _i = _bonds[_b].from;
		const uint _j = _bonds[_b].to;
		double _accum = 0.0;
		for (int _ax = 0; _ax < 3; ++_ax)
		{
			const int _ci = this->coord[_i][_ax];
			const int _cj = this->coord[_j][_ax];
			// a wrapping bond jumps by more than one cell along the axis; the
			// sign encodes whether it wraps forward (high -> 0) or backward.
			if (_ci - _cj > 1)
				_accum += _flux[_ax];
			else if (_cj - _ci > 1)
				_accum -= _flux[_ax];
		}
		if (_accum != 0.0)
			_phases(_b) = std::exp(cpx(0.0, _accum));
	}
	return _phases;
}

// ####################################################################################################

/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
t_3d<int> Lattice::getSiteDifference(t_3d<int> i, uint j) const
{
	const auto& [x1, y1, z1] = i;
	const int z = z1 - this->get_coordinates(j, Z);
	const int y = y1 - this->get_coordinates(j, Y);
	const int x = x1 - this->get_coordinates(j, X);
	// returns the site difference
	return std::tuple<int, int, int>(x, y, z);
}

// ####################################################################################################

/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
t_3d<int> Lattice::getSiteDifference(uint i, uint j) const
{
	const int z = this->get_coordinates(i, Z) - this->get_coordinates(j, Z);
	const int y = this->get_coordinates(i, Y) - this->get_coordinates(j, Y);
	const int x = this->get_coordinates(i, X) - this->get_coordinates(j, X);
	// returns the site difference
	return std::tuple<int, int, int>(x, y, z);
}

// ####################################################################################################

/*
* @brief Calculates the distance norm between two lattice site vectors.
* @param i First coordinate
* @param j Second coordinate
* @return magnitude of the distance vector
*/
double Lattice::getSiteDistance(uint i, uint j) const
{
	auto [x, y, z] = this->getSiteDifference(i, j);
	auto r			= this->getRealVec(x, y, z);
	return std::sqrt(arma::dot(r, r));
}

// ####################################################################################################

/*
* @brief Calculates the DFT matrix for the lattice
* @param phase If true, the phase is included in the calculation
* @note The DFT matrix is calculated only once
* @note Can be faster with using FFT -> to think about
* @url https://en.wikipedia.org/wiki/DFT_matrix
*/
void Lattice::calculate_dft_matrix(bool phase)
{
	this->dft_	= arma::Mat<cpx>(this->Ns, this->Ns, arma::fill::zeros);
	cpx omega_x	= std::exp(-I * cpx(TWOPI / this->get_Lx()));
	cpx omega_y	= std::exp(-I * cpx(TWOPI / this->get_Ly()));
	cpx omega_z	= std::exp(-I * cpx(TWOPI / this->get_Lz()));

	cpx e_min_pi = std::exp(-I * cpx(PI));
	// do double loop - not perfect solution

	// rvectors
	for (int row = 0; row < this->Ns; ++row)
	{
		const auto x_row		= this->get_coordinates(row, direction::X);
		const auto y_row		= this->get_coordinates(row, direction::Y);
		const auto z_row		= this->get_coordinates(row, direction::Z);
		// kvectors
		for (int col = 0; col < this->Ns; ++col)
		{
			const auto x_col	= this->get_coordinates(col, direction::X);
			const auto y_col	= this->get_coordinates(col, direction::Y);
			const auto z_col	= this->get_coordinates(col, direction::Z);

			// to shift by -PI
			cpx phase_x			= phase ? ((x_col % 2) != 0 ? e_min_pi : 1.0) : 1.0;
			cpx phase_y			= phase ? ((y_col % 2) != 0 ? e_min_pi : 1.0) : 1.0;
			cpx phase_z			= phase ? ((z_col % 2) != 0 ? e_min_pi : 1.0) : 1.0;
			// set the omegas - not optimal powers, but is calculated once
			this->dft_(row, col) = std::pow(omega_x, x_row * x_col) * std::pow(omega_y, y_row * y_col) * std::pow(omega_z, z_row * z_col) * phase_x * phase_y * phase_z;
		}
	}
	this->dft_ = this->dft_.t();
}

// ####################################################################################################

arma::Mat<cpx> Lattice::calculate_dft_vectors(bool phase)
{
	const uint k_num	= this->get_Ns();
	arma::Mat<cpx> _vc(this->Ns, this->Ns, arma::fill::zeros);

	cpx e_min_pi		= std::exp(-I * cpx(PI));

	// calculate the DFT matrix
	for (int k = 0; k < k_num; k++)
	{
		const auto _k = this->get_kVec(k);

		for (int r = 0; r < k_num; r++)
		{
			const auto _r	= this->get_rVec(r);
			// rows are exponents for given k!
			_vc(k, r)		= (phase ? e_min_pi : 1.0) * std::exp(-I * arma::dot(_k, _r));
		}
	}
	return _vc;
}

// ####################################################################################################

bool Lattice::save_bonds(std::shared_ptr<Lattice> _lat, const std::string &_dir, const std::string& _name)
{
	if (_lat && _lat->type_ == LatticeTypes::HON)
	{
		const auto Ns = _lat->get_Ns();

		arma::Mat<double> bonds_ = -arma::Mat<double>(Ns, 3, arma::fill::ones);
		for (int i = 0; i < Ns; ++i)
		{
			uint NUM_OF_NN = (uint)_lat->get_nn_ForwardNum(i);
			for (uint nn = 0; nn < NUM_OF_NN; nn++)
			{
				if (int nei = _lat->get_nnf(i, nn); nei >= 0) 
					bonds_(i, nn) = nei;
			}
		}
		saveAlgebraic(_dir, _name, bonds_, "lattice", false); 				// save the results to HDF5 file
		return true;
	}
	return false;
}

// ####################################################################################################
