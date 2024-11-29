#include "../src/Lattices/honeycomb.h"

// ####################################################################################################

Honeycomb::Honeycomb(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim	= dim;
	this->_BC	= static_cast<BoundaryConditions>(_BC);
	this->type_	= LatticeTypes::HON;
	this->type	= SSTR(getSTR_LatticeTypes(this->type_));

    // fix sites depending on _BC
	switch (this->dim)
	{
	case 1:
		this->Ly            = 1; 
		this->Lz            = 1;
		this->nnForward	    = { 0 };
		this->nnnForward    = { 0 };
		break;
	case 2:
		this->Lz            = 1;
		this->nnForward	    = { 0, 1, 2 };  // this may change as it depends on the lattice site % 2
		this->nnnForward	= { 0, 1, 2 };  // this may change as it depends on the lattice site % 2
		break;
	default:
		break;
	}

	// we take 2 * Ly because of the fact that we have two elements in one elementary cell always
	this->Ns = 2 * this->Lx * this->Ly * this->Lz;

	// neighbors
	Lattice::calculate_nn();
	Lattice::calculate_nnn();

	// coordinates
	this->calculate_coordinates();
	this->calculate_spatial_norm();


	this->a1    = arma::vec({ sqrt(3) * this->a / 2.0, 3 * this->a / 2.0, 0 });
	this->a2    = arma::vec({ -sqrt(3) * this->a / 2.0, 3 * this->a / 2.0, 0 });
	this->a3    = arma::vec({ 0, 0, this->c });

	this->kVec  = arma::mat(this->Lx * this->Ly * this->Lz, 3, arma::fill::zeros);
	this->rVec  = arma::mat(this->Lx * this->Ly * this->Lz, 3, arma::fill::zeros);
	
	//! make vectors
	this->calculate_kVec();
	this->calculate_rVec();
}

// ####################################################################################################

/*
* @brief returns the nn for a given x direction at a given lattice site.
* The neighbors are sorted as follows:
* 0 - x direction
* 1 - y direction
* 2 - z direction
* @param _site - lattice site
* @param d - direction
* @return nn[_site][d]
*/
int Honeycomb::get_nn(int _site, direction d) const
{
    switch (d)
    {
    case direction::X:
        return this->nn[_site][0];
    case direction::Y:
        return this->nn[_site][1];
    case direction::Z:
        return this->nn[_site][2];
    default:
        return 0;
    }
}

// ####################################################################################################

/*
!TODO: Implement the function!
* @brief returns the real space vector for a given multipliers of reciprocal vectors
*/
arma::vec Honeycomb::getRealVec(int x, int y, int z) const
{
    return x * this->a1 + y * this->a2 + z * this->a3;
}

// ####################################################################################################

void Honeycomb::calculate_nn(bool pbcx, bool pbcy, bool pbcz)
{
    auto _bcfun = [](int _i, int _L, bool _pbc) -> int
    {
        if (_pbc)
            return modEUC<int>(_i, _L);
        else
            return (_i >= _L) ? -1 : ((_i < 0) ? -1 : _i);
    };


    switch (this->dim)
    {
    case 1:
        {
            this->nn = v_2d<int>(this->Ns, v_1d<int>(2, 0));
            this->nnF = v_2d<int>(this->Ns, v_1d<int>(1, 0));
            for (int i = 0; i < this->Ns; i++)
            {
                this->nn[i][0] = _bcfun(i + 1, this->Lx, pbcx);
                this->nn[i][1] = _bcfun(i - 1, this->Lx, pbcx);
                // forward
                this->nnF[i][0] = this->nn[i][0];
            }
        }
        break;
    case 2:
        {
            this->nn = v_2d<int>(this->Ns, v_1d<int>(3, 0)); // z, y, x
            this->nnF = v_2d<int>(this->Ns, v_1d<int>(3, 0));
            for (int i = 0; i < this->Ns; ++i)
            {


                // get the site on the SQUARE LATTICE that encapsulates the honeycomb lattice (two nodes in one elementary cell)
                int n   = i / 2;    // n is the site on the square lattice
                int r   = i % 2;    // 0 or 1  - 0 is the first node in the elementary cell, 1 is the second node in the elementary cell
                int X   = n % this->Lx;
                int Y   = n / this->Lx;
                const bool _even = (r == 0);
                
                // z bond
                {
                    int Yprime      = _even ? _bcfun(Y - 1, this->Ly, pbcy) : _bcfun(Y + 1, this->Ly, pbcy);

                    if (Yprime == -1)
                        this->nn[i][0] = -1;
                    else
                        this->nn[i][0]  = (Yprime * this->Lx + X) * 2 + ((r == 0));
                    
                    this->nnF[i][0] = !_even ? this->nn[i][0] : -1;
                }
                // y bond - when i % 2 == 1 - we go right and when i % 2 == 0 we go left
                {
                    int Xprime      = _even ? _bcfun(X - 1, this->Lx, pbcx) : _bcfun(X + 1, this->Lx, pbcx);

                    if (Xprime == -1)
                        this->nn[i][1]  = -1;
                    else 
                        this->nn[i][1]  = (Y * this->Lx + Xprime) * 2 + ((r == 0));

                    this->nnF[i][1] = !_even ? this->nn[i][1] : -1;
                }
                // x bond is always in the same cell
                {
                    this->nn[i][2]  = _even ? i + 1 : i - 1;
                    this->nnF[i][2] = _even ? this->nn[i][2] : -1;
                }
            }
        }   
        break;
    case 3:
        // in addition we have the z direction
        this->nn = v_2d<int>(this->Ns, v_1d<int>(5, 0));

        for (int i = 0; i < this->Ns; ++i)
        {
            // get the site on the SQUARE LATTICE that encapsulates the honeycomb lattice (two nodes in one elementary cell)
            int n   = i / 2;    // n is the site on the square lattice
            int r   = i % 2;    // 0 or 1  - 0 is the first node in the elementary cell, 1 is the second node in the elementary cell
            int X   = n % this->Lx;
            int Y   = n / this->Lx;
            int Z   = n / (this->Lx * this->Ly);
            
            // z bond
            {
                int Yprime      = (r == 0) ? _bcfun(Y - 1, this->Ly, pbcy) : _bcfun(Y + 1, this->Ly, pbcy);
                this->nn[i][0]  = (Yprime * this->Lx + X) * 2 + ((r == 0));
            }
            // y bond - when i % 2 == 1 - we go right and when i % 2 == 0 we go left
            {
                int Xprime      = (r == 0) ? _bcfun(X - 1, this->Lx, pbcx) : _bcfun(X + 1, this->Lx, pbcx);
                this->nn[i][1]  = (Y * this->Lx + Xprime) * 2 + ((r == 0));
            }
            // x bond is always in the same cell
            {
                this->nn[i][2]  = (r == 0) ? i + 1 : i - 1;
            }
            
            // z top 
            {
                int Zprime      = _bcfun(Z + 1, this->Lz, pbcz);
                this->nn[i][3]  = _bcfun(Zprime * this->Lx * this->Ly + Y * this->Lx + X, this->Ns, true);
            }

            // z bottom
            {
                int Zprime      = _bcfun(Z - 1, this->Lz, pbcz);
                this->nn[i][4]  = _bcfun(Zprime * this->Lx * this->Ly + Y * this->Lx + X, this->Ns, true);
            }
        }
        break;
    default:
        break;
    }
    stoutd(this->nn);    
}

// ####################################################################################################

/*
* @brief Returns real space coordinates from a lattice site number
*/
void Honeycomb::calculate_coordinates()
{
	const int LxLy = Lx * Ly;
	this->coord = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	// we must categorize elements by pairs
	for (uint i = 0; i < Ns; i++) {
		this->coord[i][0] = (int(1.0 * i / 2.0)) % Lx;						// x axis coordinate
		this->coord[i][1] = (int(1.0 * i / (2.0 * Lx))) % Ly;				// y axis coordinate
		this->coord[i][2] = (int(1.0 * i / (LxLy))) % Lz;					// z axis coordinate			

		// we calculate the big Y that is enumerated normally accordingly and then calculate the small y which is twice bigger or twice bigger + 1
		if (i % 2 == 0)
			this->coord[i][1] = this->coord[i][1] * 2;
		else
			this->coord[i][1] = this->coord[i][1] * 2 + 1;

		//stout << VEQ(i) << "->(" << this->coordinates[i][0] << "," << this->coordinates[i][1] << "," << this->coordinates[i][2] << ")\n";
	}


}

/*
* @brief calculates the matrix of all k vectors
*/
void Honeycomb::calculate_kVec()
{
	const auto two_pi_over_Lx = TWOPI / Lx / a;
	const auto two_pi_over_Ly = TWOPI / Ly / a;
	const auto two_pi_over_Lz = TWOPI / Lz / c;

	const arma::vec b1 = { 1. / sqrt(3), 1. / 3., 0 };
	const arma::vec b2 = { -1. / sqrt(3), 1. / 3., 0 };
	const arma::vec b3 = { 0, 0, 1 };

	for (int qx = 0; qx < Lx; qx++) {
		double kx = -PI + two_pi_over_Lx * qx;
		for (int qy = 0; qy < Ly; qy++) {
			double ky = -PI + two_pi_over_Ly * qy;
			for (int qz = 0; qz < Lz; qz++) {
				double kz = -PI + two_pi_over_Lz * qz;
				uint iter = qz * (Lx * Ly) + qy * Lx + qx;
				this->kVec.row(iter) = (kx * b1 + ky * b2 + kz * b3).st();
			}
		}
	}

}

void Honeycomb::calculate_rVec()
{
	for (int x = 0; x < this->Lx; x++)
	{
		for (int y = 0; y < this->Ly; y++)
		{
			for (int z = 0; z < this->Lz; z++)
			{
				const auto _iter = z * (this->Lx * this->Ly) + y * this->Lx + x;
				this->rVec.row(_iter) = this->getRealVec(x, y, z).st();
			}
		}
	}
}

// ####################################################################################################'

/**
* @brief Get the flux sites for a given position (X, Y, Z) in the honeycomb lattice.
* 
* This function calculates the flux sites starting from a given position (X, Y, Z) in the lattice.
* It checks if the given position is within the lattice bounds and if the lattice dimension is sufficient.
* It then calculates the starting node and its neighboring nodes in a specific order.
* 
* @param X The x-coordinate of the position.
* @param Y The y-coordinate of the position.
* @param Z The z-coordinate of the position.
* @return v_1d<uint> A vector containing the flux sites starting from the given position. The position is calculated in the following order:
* - The starting node.
* - The neighbor up.
* - The neighbor right (x bond).
* - The neighbor right again (y bond).
* - The neighbor down (z bond).
* - The neighbor left (x bond).
* - The neighbor left again (y bond).
*/
v_1d<uint> Honeycomb::get_flux_sites(int X, int Y, int Z) const
{
	v_1d<uint> _flux_sites;
	if (X > Lx || Y > Ly || Z > Lz)
	{
		LOGINFO("The site is out of the lattice bounds.", LOG_TYPES::ERROR, 3);
		return _flux_sites;
	}

	if (this->dim < 2)
	{
		LOGINFO("The dimension is too low for the hexagonal lattice.", LOG_TYPES::ERROR, 3);
		return _flux_sites;
	}

	// calculate the starting point 
	const auto node_start = 2 * (X + Lx * Y + Lx * Ly * Z) + 1;
	_flux_sites.push_back(node_start);

	// get neighbor up
	const auto node_up 		= this->nn[node_start][0];
	_flux_sites.push_back(node_up);

	// get neighbor right (x bond)
	const auto node_right 	= (node_up > 0) ? this->nn[node_up][2] : -1;
	_flux_sites.push_back(node_right);

	// get neighbor right again (y bond)
	const auto node_right_2 = (node_right > 0) ? this->nn[node_right][1] : -1;
	_flux_sites.push_back(node_right_2);

	// get neighbor down (z bond)
	const auto node_down 	= (node_right_2 > 0) ? this->nn[node_right_2][0] : -1;
	_flux_sites.push_back(node_down);

	// get neighbor left (x bond)
	const auto node_left 	= (node_down > 0) ? this->nn[node_down][2] : -1;
	_flux_sites.push_back(node_left);

	// get neighbor left again (y bond)
	const auto node_left_2 	= (node_left > 0) ? this->nn[node_left][1] : -1;
	_flux_sites.push_back(node_left_2);

	return _flux_sites;
}