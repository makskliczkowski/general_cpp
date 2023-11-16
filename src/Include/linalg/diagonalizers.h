#pragma once
/*******************************
* Definitions for the linalg and
* and the diagonalizers.
* Lanczos's method etc.
*******************************/

#include "../../lin_alg.h"
#include "../../flog.h"
#include "../random.h"
#include <complex>

// matrix base class concepts
#include <concepts>
#include <type_traits>

template<typename _T>
concept HasMatrixType = std::is_base_of<arma::Mat<double>, _T>::value					|| 
						std::is_base_of<arma::Mat<std::complex<double>>, _T>::value		||
						std::is_base_of<arma::SpMat<double>, _T>::value					||
						std::is_base_of<arma::SpMat<std::complex<double>>, _T>::value;

// #####################################################################################################################
// #####################################################################################################################
// ################################################### G E N E R A L ###################################################
// #####################################################################################################################
// #####################################################################################################################

template <typename _T>
class Diagonalizer
{
public:
	template <template <class _TM = _T> class _MatType, HasMatrixType _Concept = _MatType<_T>>
	static void diagS(arma::vec& eigVal_, arma::Mat<_T>& eigVec_, const _MatType<_T>& _mat);
	template <template <class _TM = _T> class _MatType, HasMatrixType _Concept = _MatType<_T>>
	static void diagS(arma::vec& eigVal_, const _MatType<_T>& _mat);

protected:
	/*
	* @brief Based on the memory consumption decides on the method for standard diagonalization
	*/
	template <template <class _TM = _T> class _MatType, HasMatrixType _Concept = _MatType<_T>>
	static inline std::tuple<const char*, double> decideMethod(const _MatType<_T>& _mat)
	{
		auto memory				=	_mat.n_rows * _mat.n_cols * sizeof(_T);
		const char* method		=	(memory > 120 * 1e9) ? "std" : "dc";
		std::string memoryStr	=	"DIMENSION= " + STRP(memory * 1e-6, 5) + "mb";
		LOGINFO(memoryStr, LOG_TYPES::TRACE, 3);
		return std::make_tuple(method, memory);
	}
};

// ################################################# S Y M M E T R I C #################################################

/*
* @brief General procedure to diagonalize the matrix using eig_sym from the Armadillo library
*/
template <typename _T>
template <template <class _TM = _T> class _MatType, HasMatrixType _Concept>
inline void Diagonalizer<_T>::diagS(arma::vec& eigVal_, arma::Mat<_T>& eigVec_, const _MatType<_T>& _mat)
{	
	auto [method, memory] = Diagonalizer<_T>::decideMethod(_mat);
	BEGIN_CATCH_HANDLER
		arma::eig_sym(eigVal_, eigVec_, arma::Mat<_T>(_mat), method);
	END_CATCH_HANDLER("Memory exceeded. " + STRP(memory * 1e-6, 6), ;);
}

// ###########################################################

/*
* @brief General procedure to diagonalize the matrix using eig_sym from the Armadillo library. 
* Without the eigenvectors.
*/
template <typename _T>
template <template <class _TM = _T> class _MatType, HasMatrixType _Concept>
inline void Diagonalizer<_T>::diagS(arma::vec& eigVal_, const _MatType<_T>& _mat)
{	
	auto [method, memory] = Diagonalizer<_T>::decideMethod(_mat);
	BEGIN_CATCH_HANDLER
		arma::eig_sym(eigVal_, arma::Mat<_T>(_mat));
	END_CATCH_HANDLER("Memory exceeded. " + STRP(memory * 1e-6, 6), ;);
}

// ################################################### G E N E R A L ###################################################

// #####################################################################################################################
// #####################################################################################################################
// ################################################### L A N C Z O S ###################################################
// #####################################################################################################################
// #####################################################################################################################

/*
* @brief A class that introduces the Krylov basis {x, Ax, ..., A^{j-1}x}. The eigenvectors converge to the director of the
* eigenvector with the greates eigenvalue. Therefore, to ensure orthonormalization, one uses the Gram-Schmidt 
* orthogonalization.
* Suppose that {q_1, ..., q_i} is the orthonormal basis for K^i(x), i <= j. We construct q_{j+1} by orthogonalizing A^j x
* against q_1, ..., q_j -> making the Arnoldi basis (and in case of Hermitan - the Lanczos basis).
* @link https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf
*/
template<typename _T>
class LanczosMethod: public Diagonalizer<_T>
{
public:
	// ________________ W I T H   V E C T O R S ________________
	template <template <class _TM = _T> class _MatType, class _TM, HasMatrixType _Concept = _MatType<_TM>>
	static void diagS(	
						arma::vec&				_eigVal,
						arma::Mat<_T>&			_eigVec,
						const _MatType<_TM>&		_M,
						size_t					N_Krylov,
						arma::Col<_T>&			_psi0,
						arma::Mat<_T>&			_psiMat,
						arma::Mat<_T>&			_krylovVec
					);

	// ______________ W I T H O U T  V E C T O R S ______________
	template <template <class _TM = _T> class _MatType, class _TM, HasMatrixType _Concept = _MatType<_TM>>
	static void diagS(	
						arma::vec&				_eigVal,
						arma::Mat<_T>&			_eigVec,
						const _MatType<_TM>&		_M,
						size_t					N_Krylov,
						arma::Col<_T>&			_psi0,
						arma::Mat<_T>&			_psiMat
					);

	// _________________ R A N D O M   S T A R T ________________
	template <template <class _TM = _T> class _MatType, class _TM, HasMatrixType _Concept = _MatType<_TM>>
	static void diagS(	
						arma::vec&				_eigVal,
						arma::Mat<_T>&			_eigVec,
						const _MatType<_TM>&		_M,
						size_t					N_Krylov,
						randomGen*				_r
					);

	// ______ R A N D O M   S T A R T   W   V E C T O R S _______
	template <template <class _TM = _T> class _MatType, class _TM, HasMatrixType _Concept = _MatType<_TM>>
	static void diagS(	
						arma::vec&				_eigVal,
						arma::Mat<_T>&			_eigVec,
						const _MatType<_TM>&	_M,
						size_t					N_Krylov,
						randomGen*				_r,
						arma::Mat<_T>			_krylovMat
					);
};

// ################################################# S Y M M E T R I C #################################################

/*
* @brief Construct the Krylov space and use Lanczos' method to diagonalize the Hamiltonian
* @param _eigVal - eigenvalues to be saved
* @param _eigVec - eigenvectors to be saved
* @param _M	- matrix to be diagonalized with a Lanczos' method
* @param N_Krylov - number of the Krylov vectors (or vectors in the Lanczos' basis)
* @param _psi0 - starting vector 
* @param _psiMat - matrix of coefficients constructed from the Krylov space
* @param _krylovVec - save the Krylov vectors here
*/
template<class _T>
template <template <class> class _MatType, class _TM, HasMatrixType _Concept>
inline void LanczosMethod<_T>::diagS(
										arma::vec&				_eigVal, 
										arma::Mat<_T>&			_eigVec, 
										const _MatType<_TM>&	_M,
										size_t					N_Krylov, 
										arma::Col<_T>&			_psi0,
										arma::Mat<_T>&			_psiMat,
										arma::Mat<_T>&			_krylovVec
									)
{
	// check the number of states constraint
	if (N_Krylov < 2 || N_Krylov > _M.n_rows)
		throw std::runtime_error("Cannot create such small Krylov space, it does not make sense...");
	if (_M.n_cols != _M.n_rows)
		throw std::runtime_error("The matrix is not square...");

	// set Krylov subspace - Q vectors constructing Lanczos basis.
	_krylovVec.zeros(_M.n_rows, N_Krylov);
	_psiMat.zeros(N_Krylov, N_Krylov);

	// normalize state
	_psi0						= _psi0 / (_T)arma::norm(_psi0);

	// add vector to matrix at zeroth position
	_krylovVec.col(0)			= _psi0;

	// start with first step of Hamiltonian multiplication
	arma::Col<_T> carryVec0		= _M * _psi0;
	_T ai						= arma::cdot(_psi0, carryVec0);
	_T bi						= 0.0;

	// take away the parallel part and update the basis vector
	arma::Col<_T> carryVec1		= carryVec0 - ai * _psi0;
	_T bip1						= (_T)arma::norm(carryVec1);
	_psiMat(0, 0)				= ai;
	_psiMat(0, 1)				= bip1;

	// loop other states
	for (auto i = 1; i < (N_Krylov - 1); ++i)
	{
		// create new i'th vector (q := r/beta_{i-1})
		carryVec0			= carryVec1 / bip1;
		
		// add vector to matrix at i'th position
		_krylovVec.col(i)	= carryVec0;

		// put on the matrix again
		carryVec1			= _M * carryVec0;
		ai					= arma::cdot(carryVec0, carryVec1);
		// bi is bip1 from last step so we don't have to calculate it again
		bi					= bip1;
		// new carry
		carryVec1			= carryVec1 - (ai * carryVec0) - (bi * _psi0);
		bip1				= (_T)arma::norm(carryVec1);
		// set matrix
		_psiMat(i, i - 1)	= bi;
		_psiMat(i, i)		= ai;
		_psiMat(i, i + 1)	= bip1;
		//after writing to matrix we need to change psi_0(which is psi_i-1) to psii
		_psi0				= carryVec0;
	}
	// last conditions
	carryVec0									= carryVec1 / bip1;
	_krylovVec.col(N_Krylov - 1)				= carryVec0;
	carryVec1									= _M * carryVec0;
	ai											= arma::cdot(carryVec0, carryVec1);
	_psiMat(N_Krylov - 1, N_Krylov - 2)			= bip1;
	_psiMat(N_Krylov - 1, N_Krylov - 1)			= ai;

	// diagonalize
	Diagonalizer<_T>::diagS(_eigVal, _eigVec, _psiMat);
}

// ###########################################################

/*
* @brief Construct the Krylov space and use Lanczos' method to diagonalize the Hamiltonian
* @param _eigVal - eigenvalues to be saved
* @param _eigVec - eigenvectors to be saved
* @param _M	- matrix to be diagonalized with a Lanczos' method
* @param N_Krylov - number of the Krylov vectors
* @param _psi0 - starting vector
* @param _psiMat - matrix of coefficients constructed from the Krylov space
*/
template<class _T>
template <template <class> class _MatType, class _TM, HasMatrixType _Concept>
inline void LanczosMethod<_T>::diagS(
										arma::vec&				_eigVal, 
										arma::Mat<_T>&			_eigVec, 
										const _MatType<_TM>&	_M,
										size_t					N_Krylov, 
										arma::Col<_T>&			_psi0,
										arma::Mat<_T>&			_psiMat
									)
{
	// check the number of states constraint
	if (N_Krylov < 2)
		throw std::runtime_error("Cannot create such small Krylov space, it does not make sense...");

	_psiMat.zeros(N_Krylov, N_Krylov);

	// normalize state
	_psi0						= _psi0 / (_T)arma::norm(_psi0);

	// start with first step of Hamiltonian multiplication
	arma::Col<_T> carryVec0		= _M * _psi0;
	_T ai						= arma::cdot(_psi0, carryVec0);
	_T bi						= 0.0;

	// take away the parallel part
	arma::Col<_T> carryVec1		= carryVec0 - ai * _psi0;
	_T bip1						= (_T)arma::norm(carryVec1);
	_psiMat(0, 0)				= ai;
	_psiMat(0, 1)				= bip1;

	// loop other states
	for (auto i = 1; i < (N_Krylov - 1); ++i)
	{
		// create new i'th vector
		carryVec0			= carryVec1 / bip1;
		
		// put on the matrix again
		carryVec1			= _M * carryVec0;
		ai					= arma::cdot(carryVec0, carryVec1);
		// bi is bip1 from last step so we don't have to calculate it again
		bi					= bip1;
		// new carry
		carryVec1			= carryVec1 - (ai * carryVec0) - (bi * _psi0);
		bip1				= (_T)arma::norm(carryVec1);
		// set matrix
		_psiMat(i, i - 1)	= bi;
		_psiMat(i, i)		= ai;
		_psiMat(i, i + 1)	= bip1;
		//after writing to matrix we need to change psi_0(which is psi_i-1) to psii
		_psi0				= carryVec0;
	}
	// last conditions
	carryVec0				= carryVec1 / bip1;
	carryVec1				= _M * carryVec0;
	ai						= arma::cdot(carryVec0, carryVec1);
	_psiMat(N_Krylov - 1, N_Krylov - 2)			= bip1;
	_psiMat(N_Krylov - 1, N_Krylov - 1)			= ai;

	// diagonalize
	Diagonalizer<_T>::diagS(_eigVal, _eigVec, _psiMat);
}

// ###########################################################

/*
* @brief Construct the Krylov space and use Lanczos' method to diagonalize the Hamiltonian
* @param _eigVal - eigenvalues to be saved
* @param _eigVec - eigenvectors to be saved
* @param _M	- matrix to be diagonalized with a Lanczos' method
* @param N_Krylov - number of the Krylov vectors
* @param _r random number generator for creating the first vector
*/
template<typename _T>
template <template <class> class _MatType, class _TM, HasMatrixType _Concept>
inline void LanczosMethod<_T>::diagS(
									arma::vec&				_eigVal,
									arma::Mat<_T>&			_eigVec,
									const _MatType<_TM>&	_M,
									size_t					N_Krylov,
									randomGen*				_r
								)
{
	// set the seed to a random value
	arma::arma_rng::set_seed(_r->seed());  
	// define random vectors
	arma::Mat<_T> _psiMat(N_Krylov, N_Krylov, arma::fill::zeros);
	arma::Col<_T> _psi0 = arma::Col<_T>(_M.n_rows, arma::fill::randu) - 0.5;
	LanczosMethod<_T>::diagS(_eigVal, _eigVec, _M, N_Krylov, _psi0, _psiMat);
}

// ###########################################################

/*
* @brief Construct the Krylov space and use Lanczos' method to diagonalize the Hamiltonian
* @param _eigVal - eigenvalues to be saved
* @param _eigVec - eigenvectors to be saved
* @param _M	- matrix to be diagonalized with a Lanczos' method
* @param N_Krylov - number of the Krylov vectors
* @param _r random number generator for creating the first vector
* @param _krylovVec - save the Krylov vectors here
*/
template<typename _T>
template <template <class> class _MatType, class _TM, HasMatrixType _Concept>
inline void LanczosMethod<_T>::diagS(
									arma::vec&				_eigVal,
									arma::Mat<_T>&			_eigVec,
									const _MatType<_TM>&	_M,
									size_t					N_Krylov,
									randomGen*				_r,
									arma::Mat<_T>			_krylovVec
								)
{
	// check the random generator
	randomGen _ran;
	if (!_r)
		_ran = randomGen();
	else
		_ran = *_r;
	// set the seed to a random value
	arma::arma_rng::set_seed(_ran.seed());  
	// define random vectors
	arma::Mat<_T> _psiMat(N_Krylov, N_Krylov, arma::fill::zeros);
	arma::Col<_T> _psi0 = arma::Col<_T>(N_Krylov, arma::fill::randu) - 0.5;
	LanczosMethod<_T>::diagS(_eigVal, _eigVec, _M, N_Krylov, _psi0, _psiMat, _krylovVec);
}

// ###########################################################
