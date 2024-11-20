#include "../src/lattices.h"

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
			auto [a, b, c]				= this->getSymPos(xx, yy, zz);
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