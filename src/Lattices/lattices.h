/******************************************************************************
 *
 *  @file src/lattices/lattices.h
 *  @brief Pure virtual base class for all lattice geometries.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef LATTICE_H
#define LATTICE_H

#include "../common.h"
#include "../algebra/backend_config.h"
#include "lattice_symmetry.h"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

struct LatticeBond {
	uint from;
	uint to;
	int type;
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
enum LatticeTypes { SQ, HEX, HON, TRI };						//%
														//%
BEGIN_ENUM(LatticeTypes)								//%
{														//%
	DECL_ENUM_ELEMENT(SQ),								//%
	DECL_ENUM_ELEMENT(HEX),								//%
	DECL_ENUM_ELEMENT(HON),								//%
	DECL_ENUM_ELEMENT(TRI)								//%
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

class Lattice {
protected:
	BoundaryConditions _BC	= BoundaryConditions::PBC;
	LatticeTypes type_		= LatticeTypes::SQ;
	std::string type		= "";
	
	unsigned int dim		= 1;
	unsigned int Ns			= 1;
	
	v_2d<int> nn;
	v_2d<int> nnF;
	v_1d<uint> nnForward;
	
	v_2d<int> nnn;
	v_2d<int> nnnF;
	v_1d<uint> nnnForward;
	
	v_2d<int> coord;
	v_3d<int> spatialNorm;

	arma::vec a1, a2, a3;
	arma::vec b1, b2, b3;
	arma::mat kVec;
	arma::mat rVec;
	arma::Mat<cpx> dft_;

public:
	enum direction { X, Y, Z };

	virtual ~Lattice() { LOGINFOG("General lattice is destroyed.", LOG_TYPES::DEBUG, 3); };

	virtual int get_Lx() const = 0;
	virtual int get_Ly() const = 0;
	virtual int get_Lz() const = 0;
	auto getSiteDifference(t_3d<int> i, uint j) const -> t_3d<int>;
	auto getSiteDifference(uint i, uint j) const -> t_3d<int>;
	auto getSiteDistance(uint i, uint j) const -> double;

	virtual arma::vec getRealVec(int x, int y, int z) const = 0;
	virtual int getNorm(int x, int y, int z) const = 0;

	virtual uint get_nn_ForwardNum(int site, int num) const = 0;
	virtual uint get_nnn_ForwardNum(int site, int num) const = 0;
	virtual v_1d<uint> get_nn_ForwardNum(int, v_1d<uint>) const = 0;
	virtual v_1d<uint> get_nnn_ForwardNum(int, v_1d<uint>) const = 0;
	virtual int get_nn(int site, direction d) const = 0;
	virtual int get_nnf(int site, int n) const { return this->nnF[site][n]; };
	
	[[nodiscard]] auto get_nn_ForwardNum(int site) const -> uint { return (uint)this->nnForward.size(); };
	[[nodiscard]] auto get_nnn_ForwardNum(int site) const -> uint { return (uint)this->nnnForward.size(); };
	[[nodiscard]] auto get_nn(int site, int nei_num) const -> int { return this->nn[site][nei_num]; };
	[[nodiscard]] auto get_nnn(int site, int nei_num) const -> int { return this->nnn[site][nei_num]; };
	[[nodiscard]] auto get_nn(int site) const -> uint { return (uint)this->nn[site].size(); };
	[[nodiscard]] auto get_nnn(int site) const -> uint { return (uint)this->nnn[site].size(); };
	[[nodiscard]] auto get_nei(int lat_site, int corr_len) const -> int;

	// ----------------------- LATTICE GRAPH / BONDS
	virtual uint get_Sublattice(uint site) const { return 0; };
	v_1d<LatticeBond> get_bonds() const;
	uint coordination_number(int _site) const;
	arma::SpMat<double> adjacency_matrix() const;
	v_1d<uint> sublattice_partition() const;
	arma::cx_vec boundary_flux_phases(double _flux_x, double _flux_y, double _flux_z) const;

	virtual uint get_SitesPerCell() const = 0;
	uint get_Cells() const { return this->Ns / this->get_SitesPerCell(); }

	bool is_periodic_x() const { return this->_BC == BoundaryConditions::PBC || this->_BC == BoundaryConditions::MBC; }
	bool is_periodic_y() const { return this->_BC == BoundaryConditions::PBC || this->_BC == BoundaryConditions::SBC; }
	bool is_periodic_z() const { return this->_BC == BoundaryConditions::PBC; }

	std::vector<std::vector<uint32_t>> get_translation_perms() const {
		return LatticeSym::translation_perms(
			static_cast<std::size_t>(this->get_Lx()),
			static_cast<std::size_t>(this->get_Ly()),
			static_cast<std::size_t>(this->get_SitesPerCell())
		);
	}

	const arma::Mat<cpx>& get_DFT() const { return this->dft_; };
	BoundaryConditions get_BC() const { return this->_BC; };
	LatticeTypes get_Type() const { return this->type_; };
	std::string get_type() const { return this->type; };

	arma::mat get_kVec() const { return this->kVec; };
	arma::subview_row<double> get_kVec(uint row) { return this->kVec.row(row); };

	arma::mat get_rVec() const { return this->rVec; };
	arma::subview_row<double> get_rVec(uint row) { return this->rVec.row(row); };

	v_3d<int> get_spatial_norm() const { return this->spatialNorm; };
	auto get_spatial_norm(int x, int y, int z) const -> int { return this->spatialNorm[x][y][z]; };
	auto get_coordinates(int site, direction axis) const -> int { return this->coord[site][axis]; };
	auto get_Ns() const -> uint { return this->Ns; };
	[[nodiscard]] bool wrong_nei(std::ptrdiff_t nei) const noexcept {
		return nei < 0 || static_cast<std::size_t>(nei) >= static_cast<std::size_t>(this->Ns);
	}
	auto get_Dim() const -> uint { return this->dim; };
	auto get_info() const -> std::string;

	void calculate_nn();
	void calculate_nnn();
	void calculate_spatial_norm();
	
	virtual void calculate_nn(bool pbcx, bool pbcy, bool pbcz) {};
	virtual void calculate_nnn(bool pbcx, bool pbcy, bool pbcz) {};

	virtual void calculate_nn_pbc() { this->calculate_nn(true, true, true); };
	virtual void calculate_nn_obc() { this->calculate_nn(false, false, false); };
	virtual void calculate_nn_mbc() { this->calculate_nn(true, false, false); };
	virtual void calculate_nn_sbc() { this->calculate_nn(false, true, false); };
	
	virtual void calculate_nnn_pbc() { this->calculate_nnn(true, true, true); };
	virtual void calculate_nnn_obc() { this->calculate_nnn(false, false, false); };
	
	virtual void calculate_coordinates() = 0;

	virtual void calculate_dft_matrix(bool phase = true);
	virtual arma::Mat<cpx> calculate_dft_vectors(bool phase = true);

	virtual t_3d<int> getNumElems() = 0;
	virtual t_3d<int> get_sym_pos_inv(int x, int y, int z) = 0;
	virtual t_3d<int> get_sym_pos(int x, int y, int z) = 0;
	virtual bool symmetry_checker(int xx, int yy, int zz) = 0;

private:
	virtual void calculate_kVec() = 0;
	virtual void calculate_rVec() = 0;

public:
	virtual v_1d<uint> get_flux_sites(int X, int Y, int Z = 0) const { return {}; };

	static bool save_bonds(std::shared_ptr<Lattice> _lat, const std::string& _dir, const std::string& _name = "history.h5");
};

#endif // !LATTICE_H
