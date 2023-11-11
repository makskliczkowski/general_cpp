#pragma once
/*******************************
* Definitions for the linalg and
* and the diagonalizers.
* Lanczos's method etc.
*******************************/

#include "../../lin_alg.h"
#include "../random.h"

template<typename _T>
class LanczosMethod
{
public:
	bool diag(
		arma::Mat<_T> _eigVal,
		arma::Col<_T> _eigVec,
		arma::Mat<_T> _M,
		size_t N_Krylov,
		randomGen* _r);
};

template<typename _T>
inline bool LanczosMethod<_T>::diag(arma::Mat<_T> _eigVal, arma::Col<_T> _eigVec, arma::Mat<_T> _M, size_t N_Krylov, randomGen* _r)
{
	// check the random generator
	randomGen _ran;
	if (!_r)
		_ran = randomGen();
	else
		_ran = *_r;

	// check the number of states constraint
	if (N_Krylov < 2)
		throw std::runtime_error("Cannot create such small Krylov space, it does not make sense...");

	// start with the random vector
	arma::Mat<_T> _psiMat(N_Krylov, N_Krylov, arma::fill::zeros);
	arma::Col<_T> _psi0 = arma::Col<_T>(N_Krylov, arma::fill::randu) - 0.5;
	// normalize state
	_psi0						/= std::sqrt(arma::cdot(_psi0, _psi0));

	// start with first step of Hamiltonian multiplication
	arma::Col<_T> carryVector	= _M * _psi0;
	auto ai						= arma::cdot(_psi0, carryVector);
	auto bi						= 0.0;

	// take away the parallel part
	carryVector					= carryVector - ai * _psi0;
	auto bip1					= arma::cdot(carryVector, carryVector);
	_psiMat(0, 0)				= ai;
	_psiMat(0, 1)				= bip1;

	// loop other states
	for (auto i = 0; i < N_Krylov; ++i)
	{
		// create new i'th vector

	}

}