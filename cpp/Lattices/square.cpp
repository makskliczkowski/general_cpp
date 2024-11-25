#include "../../src/Lattices/square.h"

// ############################################################################################################################################

/*
* @brief Constructor for the square lattice
*/
SquareLattice::SquareLattice(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim		=		dim;
	this->_BC		=		static_cast<BoundaryConditions>(_BC);
	this->type_		=		LatticeTypes::SQ;
	this->type		=		SSTR(getSTR_LatticeTypes(this->type_));

	// fix sites depending on _BC
	switch (this->dim)
	{
	case 1:
		this->Ly = 1; this->Lz = 1;
		this->nnForward 	= { 0 };
		this->nnnForward	= { 0 };
		break;
	case 2:
		this->Lz = 1;
		this->nnForward		= { 0,1 };
		this->nnnForward	= { 0,1 };
		break;
	case 3:
		this->nnForward		= { 0,1,2 };
		this->nnnForward	= { 0,1,2 };
		break;
	default:
		break;
	}
	this->Ns = this->Lx * this->Ly * this->Lz;

	// neighbors
	Lattice::calculate_nn();
	Lattice::calculate_nnn();

	// coordinates
	this->calculate_coordinates();
	this->calculate_spatial_norm();

	this->a1 = { this->a, 0, 0 };
	this->a2 = { 0, this->b, 0 };
	this->a3 = { 0, 0, this->c };

	// calculate k_space vectors
	this->kVec = arma::mat(this->Ns, 3, arma::fill::zeros);
	this->rVec = arma::mat(this->Ns, 3, arma::fill::zeros);
	this->calculate_kVec();
	this->calculate_rVec();
	LOGINFOG("Created " + this->type + " lattice", LOG_TYPES::INFO, 1);
	LOGINFOG(this->get_info(), LOG_TYPES::TRACE, 2);
}

// ------------------------------------------------------------- Getters -------------------------------------------------------------

/*
* @brief returns the nn for a given x direction at a given lattice site
*/
int SquareLattice::get_nn(int site, Lattice::direction d) const
{
	switch (d) 
	{
	case X:
		return this->nn[site][0];
		break;
	case Y:
		return this->dim >= 2 ? this->nn[site][1] : this->nn[site][0];
		break;
	case Z:
		return this->dim == 3 ? this->nn[site][2] : this->nn[site][0];
		break;
	default:
		return this->nn[site][0];
		break;
	}
}

// ------------------------------------------------------------- nearest neighbors -------------------------------------------------------------

void SquareLattice::calculate_nn(bool pbcx, bool pbcy, bool pbcz)
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
            this->nn 	= v_2d<int>(this->Ns, v_1d<int>(4, 0));
            this->nnF 	= v_2d<int>(this->Ns, v_1d<int>(4, 0));
            for (int i = 0; i < this->Ns; ++i)
            {
				// right 
				this->nn[i][0] = _bcfun(i + 1, this->Lx, pbcx);
				// top
				this->nn[i][1] = _bcfun(i + this->Lx, this->Ns, pbcy);
				// left
				this->nn[i][2] = _bcfun(i - 1, this->Lx, pbcx);
				// bottom
				this->nn[i][3] = _bcfun(i - this->Lx, this->Ns, pbcy);

				// forward
				this->nnF[i][0] = this->nn[i][0];
				this->nnF[i][1] = this->nn[i][1];
				this->nnF[i][2] = -1;
				this->nnF[i][3] = -1;
			}	
        }   
        break;
    case 3:
        // in addition we have the z direction
        this->nn = v_2d<int>(this->Ns, v_1d<int>(6, 0));
		this->nnF = v_2d<int>(this->Ns, v_1d<int>(6, 0));
        for (int i = 0; i < this->Ns; ++i)
        {
			// right
			this->nn[i][0] = _bcfun(i + 1, this->Lx, pbcx);
			// top
			this->nn[i][1] = _bcfun(i + this->Lx, this->Ns, pbcy);
			// up
			this->nn[i][2] = _bcfun(i + this->Lx * this->Ly, this->Ns, pbcz);
			// left
			this->nn[i][3] = _bcfun(i - 1, this->Lx, pbcx);
			// bottom
			this->nn[i][4] = _bcfun(i - this->Lx, this->Ns, pbcy);
			// down
			this->nn[i][5] = _bcfun(i - this->Lx * this->Ly, this->Ns, pbcz);
			// forward
			this->nnF[i][0] = this->nn[i][0];
			this->nnF[i][1] = this->nn[i][1];
			this->nnF[i][2] = this->nn[i][2];
			this->nnF[i][3] = -1;
			this->nnF[i][4] = -1;
			this->nnF[i][5] = -1;
        }
        break;
    default:
        break;
    }
    stoutd(this->nn);    
}

// ------------------------------------------------------------- next nearest neighbors -------------------------------------------------------------


void SquareLattice::calculate_nnn(bool pbcx, bool pbcy, bool pbcz)
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
            this->nnn = v_2d<int>(this->Ns, v_1d<int>(2, 0));
            this->nnF = v_2d<int>(this->Ns, v_1d<int>(2, 0));
            for (int i = 0; i < this->Ns; i++)
            {
				// right
				this->nnn[i][0] = _bcfun(i + 2, this->Lx, pbcx);
				// left
				this->nnn[i][1] = _bcfun(i - 2, this->Lx, pbcx);
				// forward
				this->nnF[i][0] = this->nn[i][0];
				this->nnF[i][1] = -1;
            }
        }
        break;
    case 2:
        {
            this->nnn 	= v_2d<int>(this->Ns, v_1d<int>(4, 0));
            this->nnnF 	= v_2d<int>(this->Ns, v_1d<int>(4, 0));
            for (int i = 0; i < this->Ns; ++i)
            {
				// right 
				this->nnn[i][0] = _bcfun(i + 1, this->Lx, pbcx);
				// top
				this->nnn[i][1] = _bcfun(i + this->Lx, this->Ns, pbcy);
				// left
				this->nnn[i][2] = _bcfun(i - 1, this->Lx, pbcx);
				// bottom
				this->nnn[i][3] = _bcfun(i - this->Lx, this->Ns, pbcy);

				// forward
				this->nnnF[i][0] = this->nn[i][0];
				this->nnnF[i][1] = this->nn[i][1];
				this->nnnF[i][2] = -1;
				this->nnnF[i][3] = -1;
			}	
        }   
        break;
    case 3:
        // in addition we have the z direction
        this->nnn 	= v_2d<int>(this->Ns, v_1d<int>(6, 0));
		this->nnnF 	= v_2d<int>(this->Ns, v_1d<int>(6, 0));
        for (int i = 0; i < this->Ns; ++i)
        {
			// right
			this->nnn[i][0] = _bcfun(i + 1, this->Lx, pbcx);
			// top
			this->nnn[i][1] = _bcfun(i + this->Lx, this->Ns, pbcy);
			// up
			this->nnn[i][2] = _bcfun(i + this->Lx * this->Ly, this->Ns, pbcz);
			// left
			this->nnn[i][3] = _bcfun(i - 1, this->Lx, pbcx);
			// bottom
			this->nnn[i][4] = _bcfun(i - this->Lx, this->Ns, pbcy);
			// down
			this->nnn[i][5] = _bcfun(i - this->Lx * this->Ly, this->Ns, pbcz);
			// forward
			this->nnnF[i][0] = this->nn[i][0];
			this->nnnF[i][1] = this->nn[i][1];
			this->nnnF[i][2] = this->nn[i][2];
			this->nnnF[i][3] = -1;
			this->nnnF[i][4] = -1;
			this->nnnF[i][5] = -1;
        }
        break;
    default:
        break;
    }
    stoutd(this->nn);    
}

// ------------------------------------------------------------- coordinates -------------------------------------------------------------

/*
* @brief Returns real space coordinates from a lattice site number
*/
void SquareLattice::calculate_coordinates()
{
	const int LxLy	= Lx * Ly;
	this->coord		= v_2d<int>(this->Ns, v_1d<int>(3, 0));
	for (int i = 0; i < (int)this->Ns; i++) 
	{
		this->coord[i][0] = i % Lx;												// x axis coordinate
		this->coord[i][1] = (static_cast<int>(1.0 * i / Lx)) % Ly;				// y axis coordinate
		this->coord[i][2] = (static_cast<int>(1.0 * i / LxLy)) % Lz;			// z axis coordinate			
	}
}

// ############################################################################################################################################

/*
* @brief calculates all the kVec for a square lattice
*/
void SquareLattice::calculate_kVec()
{
	//const auto _V 				= a * b * c;
	const auto two_pi_over_Lx	= TWOPI / a / Lx;
	const auto two_pi_over_Ly	= TWOPI / b / Ly;
	const auto two_pi_over_Lz	= TWOPI / c / Lz;

	for (int qx = 0; qx < Lx; qx++) {
		double kx = -PI + two_pi_over_Lx * qx;
		for (int qy = 0; qy < Ly; qy++) {
			double ky = -PI + two_pi_over_Ly * qy;
			for (int qz = 0; qz < Lz; qz++) {
				double kz = -PI + two_pi_over_Lz * qz;
				uint iter = qz * (Lx * Ly) + qy * Lx + qx;
				this->kVec.row(iter) = { kx, ky, kz };
			}
		}
	}
}

/*
* @brief calculates all the rVec for a square lattice
* @note rVec is the real space coordinates of the lattice sites
* @note rVec is a matrix of size Ns x 3
* @note rVec(i, :) is the real space coordinates of the i-th lattice site
* @note rVec(i, 0) is the x coordinate of the i-th lattice site
* @note rVec(i, 1) is the y coordinate of the i-th lattice site
* @note rVec(i, 2) is the z coordinate of the i-th lattice site
* @note rVec is calculated as rVec = x * a + y * b + z * c
*/
void SquareLattice::calculate_rVec()
{
	for (int z = 0; z < this->Lz; ++z)
	{
		for (int y = 0; y < this->Ly; ++y)
		{
			for (int x = 0; x <	this->Lx; ++x)
			{
				auto _iter = this->Ly * this->Lx * z + this->Lx * y + x;
				this->rVec.row(_iter) = this->getRealVec(x, y, z).st();
			}
		}
	}
}
