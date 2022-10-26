#include "../../src/Lattices/square.h"


/*
* @brief Constructor for the square lattice
*/
SquareLattice::SquareLattice(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim = dim;
	this->_BC = _BC;
	this->type = "square";
	// fix sites depending on _BC
	switch (this->dim)
	{
	case 1:
		this->Ly = 1; this->Lz = 1;
		this->nn_forward = { 0 };
		this->nnn_forward = { 0 };
		break;
	case 2:
		this->Lz = 1;
		this->nn_forward = { 0,1 };
		this->nnn_forward = { 0,1 };
		break;
	case 3:
		this->nn_forward = { 0,1,2 };
		this->nnn_forward = { 0,1,2 };
		break;
	default:
		break;
	}
	this->Ns = this->Lx * this->Ly * this->Lz;

	// neighbors
	this->calculate_nn();
	this->calculate_nnn();

	// coordinates
	this->calculate_coordinates();
	this->calculate_spatial_norm();

	this->a1 = { this->a, 0, 0 };
	this->a2 = { 0, this->b, 0 };
	this->a3 = { 0, 0, this->c };

	// calculate k_space vectors
	this->k_vectors = mat(this->Ns, 3, arma::fill::zeros);
	this->calculate_k_vectors();
}

// ------------------------------------------------------------- Getters -------------------------------------------------------------

/*
* @brief returns the nn for a given x direction at a given lattice site
*/
int SquareLattice::get_x_nn(int lat_site) const
{
	return this->get_nn(lat_site, 0);
}

/*
* @brief returns the nn for a given y direction at a given lattice site
*/
int SquareLattice::get_y_nn(int lat_site) const
{
	return this->dim == 2 ? this->get_nn(lat_site, 1) : this->get_nn(lat_site, 0);
}

/*
* @brief returns the nn for a given z direction at a given lattice site
*/
int SquareLattice::get_z_nn(int lat_site) const
{
	return this->dim == 3 ? this->get_nn(lat_site, 2) : this->get_nn(lat_site, 0);
}

/*
* @brief returns the real space vector for a given multipliers of reciprocal vectors
*/
vec SquareLattice::get_real_space_vec(int x, int y, int z) const
{
	return { a * x, b * y, c * z };
}

// ------------------------------------------------------------- nearest neighbors -------------------------------------------------------------

/*
* @brief Calculate the nearest neighbors with PBC
*/
void SquareLattice::calculate_nn_pbc()
{
	switch (this->dim)
	{
	case 1:
		// One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = myModuloEuclidean(i + 1, Lx);											// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i - 1, Lx);											// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			this->nearest_neighbors[i][0] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i + 1, Lx);		// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i + Lx, Ns);											// top
			this->nearest_neighbors[i][2] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i - 1, Lx);		// left
			this->nearest_neighbors[i][3] = myModuloEuclidean(i - Lx, Ns);											// bottom
		}
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(6, 0));
		for (int i = 0; i < Ns; i++) {
			int x = i % Lx;
			int y = static_cast<int>(1.0 * i / Lx) % Ly;
			int z = static_cast<int>(1.0 * i / Lx / Ly) % Lz;
			this->nearest_neighbors[i][0] = z * Lx * Ly + y * Lx + myModuloEuclidean(i + 1, Lx);					// right - x
			this->nearest_neighbors[i][1] = z * Lx * Ly + myModuloEuclidean(i + Lx, Lx * Ly);						// right - y
			this->nearest_neighbors[i][2] = myModuloEuclidean(i + Lx * Ly, Ns);										// right - z

			this->nearest_neighbors[i][3] = z * Lx * Ly + y * Lx + myModuloEuclidean(i - 1, Lx);					// left - x
			this->nearest_neighbors[i][4] = z * Lx * Ly + myModuloEuclidean(i - Lx, Lx * Ly);						// left - y
			this->nearest_neighbors[i][5] = myModuloEuclidean(i - Lx * Ly, Ns);										// left - z
		}
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with OBC
*/
void SquareLattice::calculate_nn_obc()
{
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = (i + 1) >= Lx ? -1 : i + 1;										// right
			this->nearest_neighbors[i][1] = (i - 1) < 0 ? -1 : i - 1;										// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			auto x = i % Lx;
			auto y = static_cast<int>(1.0 * i / Lx) % Ly;
			this->nearest_neighbors[i][0] = (i + 1) < (y + 1) * Lx ? i + 1 : -1;							// right
			this->nearest_neighbors[i][1] = i + Lx < Ns ? i + Lx : -1;										// top
			this->nearest_neighbors[i][2] = (i - 1) >= y * Lx ? i - 1 : -1;									// left
			this->nearest_neighbors[i][3] = i - Lx >= 0 ? i - Lx : -1;										// bottom
		}
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(6, 0));
		for (int i = 0; i < Ns; i++) {
			int x = i % Lx;
			int y = static_cast<int>(1.0 * i / Lx) % Ly;
			int z = static_cast<int>(1.0 * i / Lx / Ly) % Lz;
			this->nearest_neighbors[i][0] = z * Lx * Ly + y * Lx + (i + 1 < (z * Lx * Ly + (y + 1) * Lx) ? i + 1 : -1);					// right - x
			this->nearest_neighbors[i][1] = z * Lx * Ly + (i + Lx < ((z + 1)* Lx* Ly) ? i + Lx : -1);									// right - y
			this->nearest_neighbors[i][2] = i + Lx * Ly < Ns ? i + Lx * Ly : -1;														// right - z

			this->nearest_neighbors[i][3] = z * Lx * Ly + y * Lx + (i - 1 >= (z * Lx * Ly + y * Lx) ? i - 1 : -1);						// left - x
			this->nearest_neighbors[i][4] = z * Lx * Ly + (i - Lx >= (z * Lx * Ly) ? i - Lx : -1);										// left - y
			this->nearest_neighbors[i][5] = i - Lx * Ly >= 0 ? i - Lx * Ly : -1;														// left - z
		}
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with MBC [PBC->x;OBC->y]
*/
void SquareLattice::calculate_nn_mbc()
{
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = (i + 1) >= Lx ? -1 : i + 1;										// right
			this->nearest_neighbors[i][1] = (i - 1) == 0 ? -1 : i - 1;										// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			auto x = i % Lx;
			auto y = static_cast<int>(1.0 * i / Lx) % Ly;
			this->nearest_neighbors[i][0] = (i + 1) < (y + 1) * Lx ? y * Lx + x + 1 : -1;					// right
			this->nearest_neighbors[i][1] = i + Lx < Ns ? i + Lx : -1;										// top
			this->nearest_neighbors[i][2] = (i - 1) >= y * Lx ? y * Lx + x - 1 : -1;						// left
			this->nearest_neighbors[i][3] = i - Lx >= 0 ? i - Lx : -1;										// bottom
		}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

// ------------------------------------------------------------- next nearest neighbors -------------------------------------------------------------

/*
* @brief Calculate the next nearest neighbors with PBC
*/
void SquareLattice::calculate_nnn_pbc()
{
	switch (this->dim)
	{
	case 1:
		/* One dimension */
		this->next_nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->next_nearest_neighbors[i][0] = myModuloEuclidean(i + 2, Lx);											// right
			this->next_nearest_neighbors[i][1] = myModuloEuclidean(i - 2, Lx);											// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->next_nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			this->next_nearest_neighbors[i][0] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i + 2, Lx);		// right
			this->next_nearest_neighbors[i][1] = myModuloEuclidean(i + 2 * Lx, Ns);											// top
			this->next_nearest_neighbors[i][2] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i - 2, Lx);		// left
			this->next_nearest_neighbors[i][3] = myModuloEuclidean(i - 2 * Lx, Ns);											// bottom
		}
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->next_nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(6, 0));
		for (int i = 0; i < Ns; i++) {
			int x = i % Lx;
			int y = static_cast<int>(1.0 * i / Lx) % Ly;
			int z = static_cast<int>(1.0 * i / Lx / Ly) % Lz;
			this->next_nearest_neighbors[i][0] = z * Lx * Ly + y * Lx + myModuloEuclidean(i + 2, Lx);					// right - x
			this->next_nearest_neighbors[i][1] = z * Lx * Ly + myModuloEuclidean(i + 2 * Lx, Lx * Ly);						// right - y
			this->next_nearest_neighbors[i][2] = myModuloEuclidean(i + 2 * Lx * Ly, Ns);										// right - z

			this->next_nearest_neighbors[i][3] = z * Lx * Ly + y * Lx + myModuloEuclidean(i - 2, Lx);					// left - x
			this->next_nearest_neighbors[i][4] = z * Lx * Ly + myModuloEuclidean(i - 2 * Lx, Lx * Ly);						// left - y
			this->next_nearest_neighbors[i][5] = myModuloEuclidean(i - 2 * Lx * Ly, Ns);										// left - z
		}
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the next nearest neighbors with OBC
*/
void SquareLattice::calculate_nnn_obc()
{
	switch (this->dim)
	{
	case 1:
		/* One dimension */
		this->next_nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->next_nearest_neighbors[i][0] = (i + 2) >= Lx ? -1 : i + 2;										// right
			this->next_nearest_neighbors[i][1] = (i - 2) < 0 ? -1 : i - 2;										// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->next_nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		break;
	case 3:
		// Three dimensions
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->next_nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(6, 0));
		break;
	default:
		break;
	}
}

// ------------------------------------------------------------- coordinates -------------------------------------------------------------

/*
* @brief Returns real space coordinates from a lattice site number
*/
void SquareLattice::calculate_coordinates()
{
	const int LxLy = Lx * Ly;
	this->coordinates = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	for (int i = 0; i < Ns; i++) {
		this->coordinates[i][0] = i % Lx;												// x axis coordinate
		this->coordinates[i][1] = (static_cast<int>(1.0 * i / Lx)) % Ly;				// y axis coordinate
		this->coordinates[i][2] = (static_cast<int>(1.0 * i / LxLy)) % Lz;				// z axis coordinate			
		//std::cout << "(" << this->coordinates[i][0] << "," << this->coordinates[i][1] << "," << this->coordinates[i][2] << ")\n";
	}
}

/*
* @brief calculates all the k_vectors for a square lattice
*/
void SquareLattice::calculate_k_vectors()
{
	const auto two_pi_over_Lx = TWOPI / a / Lx;
	const auto two_pi_over_Ly = TWOPI / b / Ly;
	const auto two_pi_over_Lz = TWOPI / c / Lz;

	for (int qx = 0; qx < Lx; qx++) {
		double kx = -PI + two_pi_over_Lx * qx;
		for (int qy = 0; qy < Ly; qy++) {
			double ky = -PI + two_pi_over_Ly * qy;
			for (int qz = 0; qz < Lz; qz++) {
				double kz = -PI + two_pi_over_Lz * qz;
				uint iter = qz * (Lx * Ly) + qy * Lx + qx;
				this->k_vectors.row(iter) = { kx, ky, kz };
			}
		}
	}
}

// ------------------------------------------------------------- forwards -------------------------------------------------------------

// ------------------------------------------------------------- nn 

/*
* @brief returns forward neighbors number
*/
v_1d<uint> SquareLattice::get_nn_forward_number(int lat_site) const
{
	return this->nn_forward;
}

/*
* @brief returns the integer given neighbor for a given site
*/
uint SquareLattice::get_nn_forward_num(int lat_site, int num) const
{
	return this->nn_forward[num];
}

// ------------------------------------------------------------- nnn

/*
* @brief returns forward next neighbors number
*/
v_1d<uint> SquareLattice::get_nnn_forward_number(int lat_site) const
{
	return this->nnn_forward;
}

/*
* @brief returns the integer given neighbor for a given site
*/
uint SquareLattice::get_nnn_forward_num(int lat_site, int num) const
{
	return this->nnn_forward[num];
}