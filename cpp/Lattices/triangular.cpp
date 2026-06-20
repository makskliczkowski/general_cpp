#include "../../src/lattices/triangular.h"

// ############################################################################################################################################

/*
* @brief Constructor for the triangular lattice (2D Bravais).
*/
TriangularLattice::TriangularLattice(int Lx, int Ly, int _BC)
	: Lx(Lx), Ly(Ly), Lz(1)
{
	this->dim	= 2;
	this->_BC	= static_cast<BoundaryConditions>(_BC);
	this->type_	= LatticeTypes::TRI;
	this->type	= SSTR(getSTR_LatticeTypes(this->type_));

	// six forward-neighbor slots / six next-nearest forward slots
	this->nnForward		= { 0, 1, 2 };
	this->nnnForward	= { 0, 1, 2 };
	this->Ns			= this->Lx * this->Ly * this->Lz;

	Lattice::calculate_nn();
	Lattice::calculate_nnn();

	this->calculate_coordinates();
	this->calculate_spatial_norm();

	this->a1 = { 1.0, 0.0, 0.0 };
	this->a2 = { 0.5, std::sqrt(3.0) / 2.0, 0.0 };
	this->a3 = { 0.0, 0.0, 1.0 };

	this->kVec = arma::mat(this->Ns, 3, arma::fill::zeros);
	this->rVec = arma::mat(this->Ns, 3, arma::fill::zeros);
	this->calculate_kVec();
	this->calculate_rVec();
	LOGINFOG("Created " + this->type + " lattice", LOG_TYPES::INFO, 1);
	LOGINFOG(this->get_info(), LOG_TYPES::TRACE, 2);
}

// ------------------------------------------------------------- Getters -------------------------------------------------------------

int TriangularLattice::get_nn(int site, Lattice::direction d) const
{
	// directions map onto the first three neighbor slots (the rest are their
	// negatives); callers needing a specific slot use get_nn(site, slot).
	switch (d)
	{
	case X: return this->nn[site][0];
	case Y: return this->nn[site][2];
	case Z: return this->nn[site][4];
	default: return this->nn[site][0];
	}
}

// ------------------------------------------------------- nearest neighbors -------------------------------------------------------

void TriangularLattice::calculate_nn(bool pbcx, bool pbcy, bool pbcz)
{
	auto _bc = [](int _i, int _L, bool _pbc) -> int
	{
		if (_pbc)	return modEUC<int>(_i, _L);
		return (_i >= _L || _i < 0) ? -1 : _i;
	};
	auto _site = [&](int _x, int _y) -> int
	{ return (_x < 0 || _y < 0) ? -1 : _y * this->Lx + _x; };

	// six NN cell offsets: +a1, -a1, +a2, -a2, +(a1-a2), -(a1-a2)
	static const int _dx[6] = { +1, -1,  0,  0, +1, -1 };
	static const int _dy[6] = {  0,  0, +1, -1, -1, +1 };

	this->nn	= v_2d<int>(this->Ns, v_1d<int>(6, -1));
	this->nnF	= v_2d<int>(this->Ns, v_1d<int>(6, -1));
	for (int i = 0; i < (int)this->Ns; ++i)
	{
		const int x = i % this->Lx;
		const int y = i / this->Lx;
		for (int s = 0; s < 6; ++s)
		{
			const int nx = _bc(x + _dx[s], this->Lx, pbcx);
			const int ny = _bc(y + _dy[s], this->Ly, pbcy);
			const int j  = (nx < 0 || ny < 0) ? -1 : _site(nx, ny);
			this->nn[i][s] = j;
			// forward bond: only to a strictly larger site index (no double count)
			this->nnF[i][s] = (j > i) ? j : -1;
		}
	}
	(void)pbcz;
}

// ----------------------------------------------------- next nearest neighbors -----------------------------------------------------

void TriangularLattice::calculate_nnn(bool pbcx, bool pbcy, bool pbcz)
{
	auto _bc = [](int _i, int _L, bool _pbc) -> int
	{
		if (_pbc)	return modEUC<int>(_i, _L);
		return (_i >= _L || _i < 0) ? -1 : _i;
	};
	auto _site = [&](int _x, int _y) -> int
	{ return (_x < 0 || _y < 0) ? -1 : _y * this->Lx + _x; };

	// six NNN cell offsets (the second shell of the triangular lattice)
	static const int _dx[6] = { +1, -1, +2, -2, +1, -1 };
	static const int _dy[6] = { +1, -1,  0,  0, -2, +2 };

	this->nnn	= v_2d<int>(this->Ns, v_1d<int>(6, -1));
	this->nnnF	= v_2d<int>(this->Ns, v_1d<int>(6, -1));
	for (int i = 0; i < (int)this->Ns; ++i)
	{
		const int x = i % this->Lx;
		const int y = i / this->Lx;
		for (int s = 0; s < 6; ++s)
		{
			const int nx = _bc(x + _dx[s], this->Lx, pbcx);
			const int ny = _bc(y + _dy[s], this->Ly, pbcy);
			const int j  = (nx < 0 || ny < 0) ? -1 : _site(nx, ny);
			this->nnn[i][s]  = j;
			this->nnnF[i][s] = (j > i) ? j : -1;
		}
	}
	(void)pbcz;
}

// ------------------------------------------------------------- coordinates -------------------------------------------------------------

void TriangularLattice::calculate_coordinates()
{
	this->coord = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	for (int i = 0; i < (int)this->Ns; ++i)
	{
		this->coord[i][0] = i % this->Lx;
		this->coord[i][1] = (i / this->Lx) % this->Ly;
		this->coord[i][2] = 0;
	}
}

// ------------------------------------------------------------- k / r vectors -------------------------------------------------------------

void TriangularLattice::calculate_kVec()
{
	// reciprocal vectors of a1=(1,0), a2=(1/2,sqrt3/2):
	// b1 = 2pi (1, -1/sqrt3), b2 = 2pi (0, 2/sqrt3)
	const double _b1x = TWOPI;
	const double _b1y = -TWOPI / std::sqrt(3.0);
	const double _b2x = 0.0;
	const double _b2y = TWOPI * 2.0 / std::sqrt(3.0);

	for (int qy = 0; qy < this->Ly; ++qy)
	{
		for (int qx = 0; qx < this->Lx; ++qx)
		{
			const double _fx = static_cast<double>(qx) / this->Lx;
			const double _fy = static_cast<double>(qy) / this->Ly;
			const uint _iter = qy * this->Lx + qx;
			this->kVec.row(_iter) = arma::rowvec{
				_fx * _b1x + _fy * _b2x,
				_fx * _b1y + _fy * _b2y,
				0.0 };
		}
	}
}

void TriangularLattice::calculate_rVec()
{
	for (int y = 0; y < this->Ly; ++y)
		for (int x = 0; x < this->Lx; ++x)
		{
			const auto _iter = this->Lx * y + x;
			this->rVec.row(_iter) = this->getRealVec(x, y, 0).st();
		}
}
