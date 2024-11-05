#pragma once
/*******************************
* Contains the possible methods
* for linear algebra usage.
* Using the methods from: 
* Conrad Sanderson and Ryan Curtin.
* Armadillo: a template-based 
* C++ library for linear algebra.
* Journal of Open Source Software,
* Vol. 1, No. 2, pp. 26, 2016. 
*******************************/

#ifndef ALG_H
#define ALG_H

using uint = unsigned int;
// ############################################## INCLUDE FROM ARMADILLO #############################################

#ifdef _WIN32
//#define H5_BUILT_AS_DYNAMIC_LIB 
#endif

// #define ARMA_WARN_LEVEL 1
#define ARMA_USE_LAPACK             
#define ARMA_PRINT_EXCEPTIONS
//#define ARMA_BLAS_LONG_LONG                                                                 // using long long inside LAPACK call
//#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
//#define ARMA_DONT_USE_WRAPPER
//#define ARMA_USE_SUPERLU
//#define ARMA_USE_ARPACK 
#define ARMA_USE_MKL_ALLOC
#define ARMA_USE_MKL_TYPES
#define ARMA_DONT_USE_OPENMP
#define ARMA_USE_HDF5
////#define ARMA_USE_OPENMP
#define ARMA_ALLOW_FAKE_GCC
#define ARMA_DONT_PRINT_CXX11_WARNING
#define ARMA_DONT_PRINT_CXX03_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include <armadillo>

#define DH5_USE_110_API
#define D_HDF5USEDLL_ 

using u64 = arma::u64;

template<class T>
using v_mat_1d = std::vector<arma::Mat<T>>;							// 1d matrix vector
template<class T>
using v_mat = v_mat_1d<T>;											// 1d matrix vector

// matrix base class concepts

#ifdef __has_include
#	if __has_include(<concepts>)
// #		include <concepts>
#		include <type_traits>
		template<typename _T>
		concept HasMatrixType = std::is_same_v<_T, arma::Mat<double>>               || 
			std::is_same_v<_T, arma::Mat<std::complex<double>>>						||
			std::is_same_v<_T, arma::SpMat<double>>									||
			std::is_same_v<_T, arma::SpMat<std::complex<double>>>;

		template<typename _T2, typename _T>
		concept HasColType = std::is_same_v<_T, arma::Col<_T2>>						||
			std::is_same_v<_T, arma::subview_col<_T2>>;

		template<typename _T2, typename _T>
		concept HasRowType = std::is_same_v<_T, arma::Row<_T2>>						||
			std::is_same_v<_T, arma::subview_row<_T2>>;

		template<typename _T>
		concept HasArmaVectorType = std::is_same_v<_T, arma::Col<double>>			||
			std::is_same_v<_T, arma::Col<std::complex<double>>>						||
			std::is_same_v<_T, arma::Col<u64>>										||
			std::is_same_v<_T, arma::subview_col<double>>							||
			std::is_same_v<_T, arma::subview_col<std::complex<double>>>				||
			std::is_same_v<_T, arma::subview_col<u64>>								||
			std::is_same_v<_T, arma::Row<double>>									||
			std::is_same_v<_T, arma::Row<std::complex<double>>>						||
			std::is_same_v<_T, arma::Row<u64>>										||
			std::is_same_v<_T, arma::subview_row<double>>							||
			std::is_same_v<_T, arma::subview_row<std::complex<double>>>				||
			std::is_same_v<_T, arma::subview_row<u64>>;

#	endif
#else
#	pragma message ("--> Skipping concepts")
#endif

// #######################################################################################################################

template<class _T, typename = void>
struct inner_type 
{
	using type = _T;
};

template<class T>
struct inner_type<T, std::void_t<typename T::value_type>>
	: inner_type<typename T::value_type> {};

template<class T>
using inner_elem_type_t = typename T::elem_type;

template<class T>
using inner_type_t = typename inner_type<T>::type;


// ############################################# DEFINITIONS FROM ARMADILLO #############################################

#define DIAG(X)										arma::diagmat(X)
#define EYE(X)										arma::eye(X,X)
#define ZEROV(X)									arma::zeros(X)
#define ZEROM(X)									arma::zeros(X,X)
#define SUBV(X, fst, lst)							X.subvec(fst, lst)
#define SUBM(X, fstr, fstc, lstr, lstc)				X.submat(fstr, fstc, lstr, lstc)
#define UPDATEV(L, R, condition)					if (condition) (L += R); else (L -= R);

// types

using CCOL											= arma::Col<std::complex<double>>;
using CMAT											= arma::Mat<std::complex<double>>;
using DCOL											= arma::Col<double>;
using DMAT											= arma::Mat<double>;

// template types

template <typename _T>
using COL											= arma::Col<_T>;
template <typename _T>
using MAT											= arma::Mat<_T>;

// #######################################################################################################################

template<typename _T, typename MatType = arma::Mat<_T>>
class VMAT
{
private:
	std::vector<MatType> mats_;

public:
	// Constructor
	VMAT() = default;

	template <typename _ft>
	VMAT(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols, const arma::fill::fill_class<_ft>& f)	
	{
		this->mats_ = std::vector <arma::Mat<_T>>(in_z, arma::Mat<_T>(in_n_rows, in_n_cols, f));
	};

	template <typename _ft>
	VMAT(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols, const arma::fill::fill_class<_ft>& f, _T _mult)	
	{
		this->mats_ = std::vector<arma::Mat<_T>>(in_z, _mult * arma::Mat<_T>(in_n_rows, in_n_cols, f));
	};

	// Constructor taking a vector of matrices
	VMAT(const std::vector<MatType>& _mats) : mats_(_mats) {}

	// Copy constructor
	VMAT(const VMAT<_T>& other) : mats_(other.mats_) {}

	// Move constructor
	VMAT(VMAT<_T>&& other) noexcept : mats_(std::move(other.mats_)) {}

	// Destructor
	~VMAT() = default;

	// #######################################################################################################################

	// Copy assignment operator
	template<typename _T2, typename _MatType2>
	VMAT<typename std::common_type<_T, _T2>>& operator=(const VMAT<_T2, _MatType2>& other) {
		if (this != &other) {
			this->mats_ = other.mats_;
		}
		return *this;
	}

	// Move assignment operator
	template<typename _T2, typename _MatType2>
	VMAT<typename std::common_type<_T, _T2>::type, arma::Mat<typename std::common_type<_T, _T2>::type>>&
	operator=(VMAT<_T2, _MatType2>&& other) noexcept { // No const here, as we are moving
		if (this != &other) {
			this->mats_ = std::move(other.mats_); // Move the internal mats_ from other
		}
		return *this;
	}

	// #######################################################################################################################

	// ############ GETTERS ############
	
	// Get the matrix at index i
	const MatType& matrix(const arma::uword index) const 
	{
		if (index < 0 || index >= mats_.size()) 
		{
			// Handle index out of range
			throw std::out_of_range("Index out of range");
		}
		return mats_[index];
	}

	// Get the number of matrices in the container
	size_t size() const 
	{
		return mats_.size();
	}

	auto row(const arma::uword in_z, const arma::uword in_n_rows)
	{
		return this->mats_[in_z].row(in_n_rows);
	}

	auto col(const arma::uword in_z, const arma::uword in_n_cols)
	{
		return this->mats_[in_z].col(in_n_cols);
	}

	// ############# GETTERS #############

	_T get(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols)
	{
		return (this->mats_[in_z])(in_n_rows, in_n_cols);
	}

	// ############# SETTERS #############

	// Add a matrix to the container
	void add(const MatType& matrix) 
	{
		mats_.push_back(matrix);
	}

	void set(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols, const _T val)
	{
		(this->mats_[in_z])(in_n_rows, in_n_cols) = val;
	}

	void add(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols, const _T val)
	{
		(this->mats_[in_z])(in_n_rows, in_n_cols) += val;
	}
	
	void divide(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols, const double val)
	{
		(this->mats_[in_z])(in_n_rows, in_n_cols) /= val;
	}

	void multiply(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols, const double val)
	{
		(this->mats_[in_z])(in_n_rows, in_n_cols) *= val;
	}

	// ############ OPERATORS ############

	// Get the matrix at index i
	MatType& operator[](size_t index) {
		return mats_[index];
	}

	//const MatType& operator[](size_t index) const {
	//	return mats_[index];
	//}

	_T& operator()(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols)
	{
		return this->mats_[in_z](in_n_rows, in_n_cols);
	}

	const _T& operator()(const arma::uword in_z, const arma::uword in_n_rows, const arma::uword in_n_cols) const
	{
		return this->mats_[in_z](in_n_rows, in_n_cols);
	}

	// ############ ITERATORS ############

	// Iterator support
	using iterator			= typename std::vector<MatType>::iterator;
	using const_iterator	= typename std::vector<MatType>::const_iterator;

	iterator begin() 
	{
		return mats_.begin();
	}

	iterator end() 
	{
		return mats_.end();
	}

	const_iterator begin() const 
	{
		return mats_.begin();
	}

	const_iterator end() const 
	{
		return mats_.end();
	}

	// ############# PROPERTIES #############

	size_t n_rows(size_t i) const
	{
		return mats_[i].n_rows;
	}

	size_t n_cols(size_t i) const
	{
		return mats_[i].n_cols;
	}

};

// #######################################################################################################################

// ##################################################### A L G E B R A ###################################################

// #######################################################################################################################

#include "./Include/maths.h"
#include "./Include/files.h"

namespace algebra 
{

	// ################################################################## CONJUGATE #####################################################################

	template <typename _T>
	inline auto conjugate(_T x)														-> _T		{ return std::conj(x); };
	template <>
	inline auto conjugate(double x)													-> double	{ return x; };
	template <>
	inline auto conjugate(float x)													-> float	{ return x; };
	template <>
	inline auto conjugate(int x)													-> int		{ return x; };

	template <typename _T>
	inline auto real(_T x)															-> double	{ return std::real(x); };
	template <>
	inline auto real(double x)														-> double	{ return x; };

	template <typename _T>
	inline auto imag(_T x)															-> double	{ return std::imag(x); };
	template <>
	inline auto imag(double x)														-> double	{ return 0.0; };
	
	// ###################################################################### CAST #####################################################################

	template <typename _T, typename _Tin>
	inline auto cast(_Tin x)														-> _T								{ return static_cast<_T>(x); };
	template <typename _T>
	inline auto cast(std::complex<double> x)										-> _T								{ return x; };
	template <>
	inline auto cast<double>(std::complex<double> x)								-> double							{ return std::real(x); };
	
	// Armadillo columns
	template <typename _T>
	inline auto cast(const arma::Col<double>& x)									-> arma::Col<_T>					{ return x; };
	template <>
	inline auto cast<std::complex<double>>(const arma::Col<double>& x)				-> arma::Col<std::complex<double>>	{ return x + std::complex<double>(0, 1) * arma::zeros(x.n_rows); };
	template <typename _T>
	inline auto cast(const arma::Col<std::complex<double>>& x)						-> arma::Col<_T>					{ return x; };
	template <>
	inline auto cast<double>(const arma::Col<std::complex<double>>& x)				-> arma::Col<double>				{ return arma::real(x); };
	
	// Armadillo matrices 
	template <typename _T>
	inline auto cast(const arma::Mat<double>& x)									-> arma::Mat<_T>					{ return x; };
	template <>
	inline auto cast<std::complex<double>>(const arma::Mat<double>& x)				-> arma::Mat<std::complex<double>>	{ return x + std::complex<double>(0, 1) * arma::zeros(x.n_rows, x.n_cols); };
	template <typename _T>
	inline auto cast(const arma::Mat<std::complex<double>>& x)						-> arma::Mat<_T>					{ return x; };
	template <>
	inline auto cast<double>(const arma::Mat<std::complex<double>>& x)				-> arma::Mat<double>				{ return arma::real(x); };
	
	template <typename _T>
	inline auto cast(const arma::SpMat<double>& x)									-> arma::SpMat<_T>					{ return x; };
	template <>
	inline auto cast<std::complex<double>>(const arma::SpMat<double>& x)			-> arma::SpMat<std::complex<double>>{ return x + std::complex<double>(0, 1) * arma::SpMat<double>(); };
	template <typename _T>
	inline auto cast(const arma::SpMat<std::complex<double>>& x)					-> arma::SpMat<_T>					{ return x; };
	template <>
	inline auto cast<double>(const arma::SpMat<std::complex<double>>& x)			-> arma::SpMat<double>				{ return arma::real(x); };

	// #################################################################################################################################################
	
	// ############################################################# MATRIX MULTIPLICATION #############################################################
	
	// #################################################################################################################################################

	/*
	* @brief same as in https://numpy.org/doc/stable/reference/generated/numpy.outer.html
	* @param A first vector
	* @param B second vector
	* @returns the outer product of two vectors
	*/
	template <template <typename _T1i> class _T1, template <typename _T2i> class _T2, typename _T1i, typename _T2i>
	arma::Mat<typename std::common_type<_T1i, _T2i>::type> outer(const _T1<_T1i>& A, const _T2<_T2i>& B)
	{
		using res_typ = typename std::common_type<_T1i, _T2i>::type;
		arma::Mat<res_typ> out(A.n_elem, B.n_elem, arma::fill::zeros);

		for(size_t i = 0; i < A.n_elem; i++)
			for(size_t j = 0; j < B.n_elem; j++)
				out(i, j) = res_typ(A(i) * B(j));

		return out;
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Allows to calculate the matrix consisting of COL vector times ROW vector
	* @param setMat matrix to set the elements onto
	* @param setVec column vector to set the elements from
	*/
	template <typename _type>
	inline void setKetBra(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec) {
		setMat = arma::cdot(setVec, setVec.as_row());
	}
	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Allows to calculate the matrix consisting of COL vector times ROW vector
	* @param setMat matrix to set the elements onto
	* @param setVec column vector to set the elements from
	* @param plus if add or substract
	*/
	template <typename _type>
	inline void setKetBra(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec, bool plus) {
		UPDATEV(setMat, arma::cdot(setVec, setVec.as_row()), plus);
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* Puts the given matrix mSet(smaller) to a specific place in the m2Set (bigger) matrix
	* @param m2Set (bigger) matrix to find the submatrix in and set it's elements
	* @param mSet (smaller) matrix to be put in the m2Set
	* @param row row of the left upper element (row,col) of M2Set
	* @param col col of the left upper element (row,col) of M2Set
	* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
	* @param minus substract?
	*/
	template <typename _type1, typename _type2>
	void setSubMFromM(arma::Mat<_type1>& m2Set, const arma::Mat<_type2>& mSet, uint row, uint col, uint nrow, uint ncol, bool update = true, bool minus = false)
	{
		if (update)
			UPDATEV(SUBM(m2Set, row, col, row + nrow - 1, col + ncol - 1), mSet, !minus)
		else
			SUBM(m2Set, row, col, row + nrow - 1, col + ncol - 1) = mSet;
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Uses the given matrix MSet (bigger) to set the M2Set (smaller) matrix
	* @param M2Set (smaller) matrix to find the submatrix in and set it's elements
	* @param MSet (bigger) matrix to be put in the M2Set
	* @param row row of the left upper element (row,col) of MSet
	* @param col col of the left upper element (row,col) of MSet
	* @param update if we shall add or substract MSet elements from M2Set depending on minus parameter
	* @param minus substract?
	*/
	template <typename _type1, typename _type2>
	void setMFromSubM(arma::Mat<_type1>& m2Set, const arma::Mat<_type2>& mSet, uint row, uint col, uint nrow, uint ncol, bool update = true, bool minus = false)
	{
		if (update)
			UPDATEV(m2Set, SUBM(mSet, row, col, row + nrow - 1, col + ncol - 1), !minus)
		else
			m2Set = SUBM(mSet, row, col, row + nrow - 1, col + ncol - 1);
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief Calculates the multiplication of two matrices. One is a diagonal matrix (right).
	* @param _A left matrix
	* @param _D right diagonal matrix
	*/
	template<typename _MatType, typename _T>
	inline _MatType matTimesDiagMat(const _MatType& _A, const arma::Col<_T>& _D)
	{
		return _A.each_row() % _D.t();
	}

	/*
	* @brief Calculates the diagonal being the result of matrix multiplication
	*/
	template<typename _MatType, typename _MatType2>
	inline arma::Col<typename std::common_type<inner_elem_type_t<_MatType>, inner_elem_type_t<_MatType2>>::type>
		matDiagProduct(const _MatType& _L, const _MatType2& _R)
	{
		using _type = typename std::common_type<inner_elem_type_t<_MatType>, inner_elem_type_t<_MatType2>>::type;
		arma::Col<_type> diagonal(_L.n_rows, arma::fill::zeros);

		// assert(_L.n_cols == _R.n_rows && "Matrix dimensions must be compatible for multiplication");

		// calculate the diagonal
		for (std::size_t i = 0; i < _L.n_rows; ++i)
		{
			for (std::size_t k = 0; k < _L.n_cols; ++k)
			{
				diagonal(i) += algebra::cast<_type>(_L(i, k)) * algebra::cast<_type>(_R(k, i));
			}
		}
		return diagonal;
	}

	// #################################################################################################################################################
	
	// ############################################################### MATRIX PROPERTIES ###############################################################
	
	// #################################################################################################################################################
	

	// *************************************************************************************************************************************************
	namespace Solvers
	{	
		// !TODO:
		// - Implement the MKL Conjugate Gradient solver
		// - Implement the ARMA solver for the matrix-vector multiplication
		// - Add various methods for sparse matrices, non-symmetric matrices, etc.
		// - Add the option to use the solver without explicitly forming the matrix A
		// - Add checkers for the matrix properties (symmetric, positive definite, etc.)
	
		// #################################################################################################################################################

		namespace Preconditioners {

			/*
			* @brief Preconditioner interface for any method that can be used as a preconditioner for the conjugate gradient method.
			*/
			template<typename T, bool _isPositiveSemidefinite = false>
			class Preconditioner {
			public:
				const bool isPositiveSemidefinite_ 	= _isPositiveSemidefinite;	// is the matrix positive semidefinite
				bool isGram_ 						= false;					// is the matrix a Gram matrix
				double sigma_ 						= 0.0;						// regularization parameter
			
				// -----------------------------------------------------------------------------------------------------------------------------------------
			public:
				virtual ~Preconditioner() = default;
				Preconditioner() 
					: isGram_(false)
				{};
				Preconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: isGram_(isGram)
				{
					this->set(A, isGram, _sigma);
				}
				Preconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: isGram_(true), sigma_(_sigma)
				{
					this->set(Sp, S, _sigma);
				}
				// -----------------------------------------------------------------------------------------------------------------------------------------

				// set the preconditioner
				virtual void set(bool _isGram, double _sigma = 0.0) { this->isGram_ = _isGram; this->sigma_ = _sigma; }
				virtual void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) = 0;		// set the preconditioner
				virtual void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) = 0;	// set the preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				// apply the preconditioner
				virtual arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const = 0;			// general matrix preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				// operator overloading
				arma::Col<T> operator()(const arma::Col<T>& r, double sigma = 0.0) const { return this->apply(r, sigma); } 
			};

			// #################################################################################################################################################

			/**
			* @brief Identity preconditioner for the conjugate gradient method.
			* The identity preconditioner does not change the input vector.
			* @tparam T The type of the matrix elements.
			*/
			template <typename T, bool _F = false>
			class IdentityPreconditioner : public Preconditioner<T, _F> {

			public:
				IdentityPreconditioner()
					: Preconditioner<T, _F>()
					{};
				IdentityPreconditioner(const arma::Mat<T>& A, bool _isGram, double _sigma = 0.0)
					: Preconditioner<T, _F>(A, _isGram, _sigma)
				{}

				IdentityPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _F>(Sp, S, _sigma)
				{}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				// set the preconditioner
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _F>::set(isGram, _sigma);
					// do nothing
				}

				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _F>::set(true, _sigma);
					// do nothing
				}

				// apply the preconditioner
				arma::Col<T> apply(const arma::Col<T>& r, double sigma) const override
				{
					return r;
				}
			};	
			
			// #################################################################################################################################################

			/**
			* @brief Jacobi preconditioner for the conjugate gradient method. This preconditioner is used for symmetric positive definite matrices. 
			* The Jacobi preconditioner is a diagonal matrix with the diagonal elements of the original matrix on the diagonal. 
			* The inverse of the diagonal elements is used as the preconditioner.
			* @tparam T The type of the matrix elements.
			*/
			template <typename T, bool _T = true>
			class JacobiPreconditioner : public Preconditioner<T, _T> {
			private:
				arma::Col<T> diaginv_;			// diagonal inverse of the matrix 
				double tolBig_ 		= 1.0e10;	// if the value is bigger than this, then 1/value is small and we create cut-off
				T  bigVal_			= 1e-10;	// treated as zero for 1/value
				double tolSmall_ 	= 1.0e-10; 	// if the value is smaller than this, then 1/value is big and we create cut-off
				T  smallVal_		= 1e10;		// treated as zero for 1/value
			public:

				JacobiPreconditioner() 
					: Preconditioner<T, _T>()
				{};
				// is any matrix A, not necessarily a Gram matrix. Otherwise, use isGram = true and A = S+ * S
				JacobiPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{}

				JacobiPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _T>(Sp, S, _sigma)
				{}

				// set the preconditioner
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(isGram, _sigma);

					if (!isGram)
					{
						arma::Col<T> diag 	= arma::diagvec(A);
						diag 				+= (T)this->sigma_;
						this->diaginv_ 		= 1.0 / diag;
						this->diaginv_ 		= arma::clamp(this->diaginv_, -1.0e10, 1.0e10);
					}
					else
						this->set(A, A, _sigma); // setting A, as Aplus is not needed
				}

				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(true, _sigma);

					this->diaginv_.set_size(S.n_cols);
					for (size_t i = 0; i < diaginv_.n_elem; ++i) 
					{
						const T norm_val 		= arma::norm(S.col(i)) + this->sigma_;
						const double norm_abs 	= std::abs(norm_val);

						this->diaginv_(i) = (norm_abs > tolBig_) 	? bigVal_ 	:
											(norm_abs < tolSmall_) 	? smallVal_ :
											(1.0 / norm_val);
					}
				}

				// apply the preconditioner
				arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const override { return this->diaginv_ % r; }
			};

			// #################################################################################################################################################
			
			
			/*
			* @brief Incomplete Cholesky preconditioner for the conjugate gradient method. This preconditioner is used for symmetric positive definite matrices.
			* @tparam T The type of the matrix elements.
			*/
			template <typename T, bool _T = true>
			class IncompleteCholeskyPreconditioner : public Preconditioner<T, _T> {

			private:
				arma::Mat<T> L_;     // lower triangular incomplete Cholesky factor
				bool success_ 	= false;
			public:
				IncompleteCholeskyPreconditioner()
					: Preconditioner<T, _T>()
				{};			
				/**
				* @brief Constructor to initialize the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				*/
				IncompleteCholeskyPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{}

				IncompleteCholeskyPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _T>(Sp, S, _sigma)
				{}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(isGram, _sigma);

					if (!isGram) {
						// Directly calculate incomplete Cholesky factor L
						this->success_ = arma::chol(L_, A + arma::Mat<T>(A.n_cols, A.n_cols, arma::fill::eye)  * this->sigma_, "lower");
						if (!success_) {
							std::cerr << "Incomplete Cholesky decomposition failed.\n";
							L_.reset(); // Clear L_ if decomposition fails
						}
					} else 
						this->set(A.t(), A, _sigma);
				}

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param Sp The matrix to decompose.
				* @param S The matrix to decompose.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(true, _sigma);

					arma::Mat<T> A 	= Sp * S;
					A.diag() 		+= this->sigma_;
					
					if (this->success_ 	= arma::chol(L_, A, "lower"); !this->success_) {
						std::cerr << "Incomplete Cholesky decomposition failed.\n";
						L_.reset(); // Clear L_ if decomposition fails
					}
				}

				// 

				/**
				* @brief Apply the preconditioner to a given vector.
				* @param r The vector to precondition.
				* @param sigma Regularization parameter (default is 0.0).
				* @return The preconditioned vector.
				*/
				arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const override
				{
					if (this->success_) {
						
						arma::Col<T> y;
						
						// Forward solve L*y = r
						try {
							y = arma::solve(arma::trimatl(L_), r);
						} catch (const std::runtime_error& e) {
							std::cerr << "Forward solve failed: " << e.what() << "\n";
							return r; // If forward solve fails, return r as is
						}

						// Backward solve L^T*z = y
						try {
							return arma::solve(arma::trimatu(L_.t()), y);
						} catch (const std::runtime_error& e) {
							std::cerr << "Backward solve failed: " << e.what() << "\n";
							return r; // If backward solve fails, return r as is
						}
					} else
						return r; // If decomposition failed, return r as is
				}
			};

			// #################################################################################################################################################

			namespace Symmetric {
				enum class PreconditionerType {
					Identity,
					Jacobi,
					IncompleteCholesky
				}; 
			};
			namespace NonSymmetric {
				enum class PreconditionerType {
					Identity,
				}; 
			};

			// -----------------------------------------------------------------------------------------------------------------------------------------
		
			template <typename T, bool _Sym = true>
			inline Preconditioner<T, _Sym>* choose(int i = 0) {
				switch (i) {
				case 0:
					return new IdentityPreconditioner<T, _Sym>;
				case 1:
					return new JacobiPreconditioner<T, _Sym>;
				case 2:
					return new IncompleteCholeskyPreconditioner<T, _Sym>;
				default:
					return new IdentityPreconditioner<T, _Sym>;
				};
			}

			// -----------------------------------------------------------------------------------------------------------------------------------------

			template <typename T>
			inline Preconditioner<T>* choose(Symmetric::PreconditionerType i) {
				switch (i) {
				case Symmetric::PreconditionerType::Identity:
					return new IdentityPreconditioner<T>;
				case Symmetric::PreconditionerType::Jacobi:
					return new JacobiPreconditioner<T>;
				case Symmetric::PreconditionerType::IncompleteCholesky:
					return new IncompleteCholeskyPreconditioner<T>;
				default:
					return new IdentityPreconditioner<T>;
				};
			}
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

			inline std::string name(int i = 0) {
				switch (i) {
				case 0:
					return "Identity";
				case 1:
					return "Jacobi";
				case 2:
					return "Incomplete Cholesky";
				default:
					return "Identity";
				};
			}
			
			// -----------------------------------------------------------------------------------------------------------------------------------------
			
			inline std::string name(Symmetric::PreconditionerType i) {
				switch (i) {
				case Symmetric::PreconditionerType::Identity:
					return "Identity";
				case Symmetric::PreconditionerType::Jacobi:
					return "Jacobi";
				case Symmetric::PreconditionerType::IncompleteCholesky:
					return "Incomplete Cholesky";
				default:
					return "Identity";
				};
			}
		};

		// #################################################################################################################################################

		namespace FisherMatrix 
		{
			
			/*
			* @brief In case we know that the matrix S that shall be inverted is a Fisher matrix, 
			* we may use the knowledge that S_{ij} = <\Delta O^*_i \Delta O_j>, where \Delta O is the
			* derivative of the observable with respect to the parametes. (rows are samples, columns are parameters)
			* and the mean over the samples is taken and then taken out of the matrix afterwards.
			* @note The matrix S is symmetric and positive definite, so we can use the conjugate gradient method.
			* @note The matrix S is not explicitly formed, but the matrix-vector multiplication is used.
			* @note The matrix S is just a covariance matrix of the derivatives of the observable.
			* @note The matrix shall be divided by the number of samples N.
			*/
			template <typename _T>
			inline arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _DeltaO, const arma::Col<_T>& _x, const double _reg = 0.0)
			{
				const size_t _N 			= _DeltaO.n_rows;               	// Number of samples (rows)
				arma::Col<_T> _intermediate = _DeltaO * _x;     				// Calculate \Delta O * x

				// apply regularization on the diagonal
				if (_reg > 0.0)
					return (_DeltaO.t() * _intermediate) / static_cast<_T>(_N)  + _reg * _x;
				
				// no regularization
				return (_DeltaO.t() * _intermediate) / static_cast<_T>(_N);    // Calculate \Delta O^* * (\Delta O * v) / N
			}

			template <typename _T>
			inline arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _DeltaO, const arma::Mat<_T>& _DeltaOConjT, const arma::Col<_T>& x, const double _reg = 0.0)
			{
				const size_t _N 			= _DeltaO.n_rows;               	// Number of samples (rows)
				arma::Col<_T> _intermediate = _DeltaO * x;     					// Calculate \Delta O * x

				// apply regularization on the diagonal
				if (_reg > 0.0)
					return (_DeltaOConjT * _intermediate) / static_cast<_T>(_N)  + _reg * x;
				
				// no regularization
				return (_DeltaOConjT * _intermediate) / static_cast<_T>(_N);    // Calculate \Delta O^* * (\Delta O * v) / N
			}

			// -----------------------------------------------------------------------------------------------------------------------------------------

			// Conjugate gradient solver for the Fisher matrix inversion
			namespace CG 
			{
				
				/**
				* @brief Conjugate gradient solver for the Fisher matrix inversion. This method is used whenever the matrix can be 
				* decomposed into the form S = \Delta O^* \Delta O, where \Delta O is the derivative of the observable with respect to the parameters. 
				* The matrix S is symmetric and positive definite, so the conjugate gradient method can be used.
				* @equation S_{ij} = <\Delta O^*_i \Delta O_j> / N 
				* @param _DeltaO The matrix \Delta O.
				* @param _DeltaOConjT The matrix \Delta O^+.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _eps The convergence criterion.
				* @param _max_iter The maximum number of iterations.
				* @param _converged The flag indicating if the solver converged.
				* @param _reg The regularization parameter. (A + \lambda I) x \approx b
				* @return The solution vector x.
				*/
				template<typename _T1>
				inline arma::Col<_T1> conjugate_gradient(const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0,
										double _eps 				= 1.0e-6,
										size_t _max_iter 			= 1000,
										bool* _converged 			= nullptr, 
										double _reg 				= 0.0
										)
				{
					// set the initial values for the solver
					arma::Col<_T1> x 	= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 	= _F - matrixFreeMultiplication(_DeltaO, _DeltaOConjT, x, _reg);
					_T1 rs_old 			= arma::cdot(r, r);

					// check for convergence already
					if (std::abs(rs_old) < _eps) {
						if (_converged != nullptr)
							*_converged = true;
						return x;
					}

					// create the search direction vector
					arma::Col<_T1> p 	= r;
					arma::Col<_T1> Ap;		// matrix-vector multiplication result

					// iterate until convergence
					for (size_t i = 0; i < _max_iter; ++i)
					{
						Ap 					= matrixFreeMultiplication(_DeltaO, _DeltaOConjT, p, _reg);
						_T1 alpha 			= rs_old / arma::cdot(p, Ap);
						x 					+= alpha * p;
						r 					-= alpha * Ap;
						_T1 rs_new 			= arma::cdot(r, r);

						// Check for convergence
						if (std::abs(rs_new) < _eps) {
							if (_converged != nullptr)
								*_converged = true;
							return x;
						}
						
						// update the search direction
						p 					= r + (rs_new / rs_old) * p;
						rs_old 				= rs_new;
					}

					std::cerr << "\t\t\tConjugate gradient solver did not converge." << std::endl;
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}
				// -----------------------------------------------------------------------------------------------------------------------------------------

				/*
				* @brief Conjugate gradient solver for the Fisher matrix inversion. This method is used whenever the matrix can be
				* decomposed into the form S = \Delta O^* \Delta O, where \Delta O is the derivative of the observable with respect to the parameters.
				* The matrix S is symmetric and positive definite, so the conjugate gradient method can be used.
				* @equation S_{ij} = <\Delta O^*_i \Delta O_j> / N
				* @param _DeltaO The matrix \Delta O.
				* @param _DeltaOConjT The matrix \Delta O^+.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _preconditioner The preconditioner for the conjugate gradient method.
				* @param _eps The convergence criterion.
				* @param _max_iter The maximum number of iterations.
				* @param _converged The flag indicating if the solver converged.
				* @param _reg The regularization parameter. (A + \lambda I) x \approx b
				* @return The solution vector x.
				*/
				template<typename _T1>
				inline arma::Col<_T1> conjugate_gradient(const arma::Mat<_T1>& _DeltaO,
														const arma::Mat<_T1>& _DeltaOConjT,
														const arma::Col<_T1>& _F, 
														arma::Col<_T1>* _x0,
														Preconditioners::Preconditioner<_T1, true>* _preconditioner = nullptr,
														double _eps 			= 1.0e-6,
														size_t _max_iter 		= 1000,
														bool* _converged 		= nullptr,
														double _reg 			= 0.0
														)
				{
					if (_preconditioner == nullptr)
						return conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);

					// set the initial values for the solver
					arma::Col<_T1> x 	= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 	= _F - matrixFreeMultiplication(_DeltaO, _DeltaOConjT, x, _reg);	// calculate the first residual
					arma::Col<_T1> z 	= _preconditioner->apply(r);										// apply the preconditioner to Mz = r
					arma::Col<_T1> p 	= z;																// set the search direction
					arma::Col<_T1> Ap;																		// matrix-vector multiplication result

					_T1 rs_old 			= arma::cdot(r, z);													// the initial norm of the residual
					// _T1 initial_rs		= std::abs(rs_old);  												// For relative tolerance check
					
					// iterate until convergence
					for (size_t i = 0; i < _max_iter; ++i)
					{
						Ap 						= matrixFreeMultiplication(_DeltaO, _DeltaOConjT, p, _reg);
						_T1 alpha 				= rs_old / arma::cdot(p, Ap);
						x 						+= alpha * p;
						r 						-= alpha * Ap;

						// Check for convergence
						if (std::abs(arma::cdot(r, r)) < _eps) {
							if (_converged != nullptr)
								*_converged = true;
							return x;
						}
						z 						= _preconditioner->apply(r); 								// update the preconditioner
						_T1 rs_new 				= arma::cdot(r, z);
						p 						= z + (rs_new / rs_old) * p;
						rs_old 					= rs_new;
					}

					std::cerr << "\t\t\tConjugate gradient solver did not converge." << std::endl;
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}
			};


			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES_QLP 
			{	
				// #################################################################################################################################################

				/**
				* @brief Check the sign of a value
				* @param val value to be checked
				* @return sign of a variable
				*/
				template <typename T1, typename T2 = T1>
				inline T1 sgn(T2 val) {
					return (T2(0) < val) - (val < T2(0));
				}

				// #################################################################################################################################################

				/**
				* @brief  The reflectors from Algorithm 1 in Choi and Saunders (2005) for real a and b, which is a stable form for computing
				r = √a2 + b2 ≥ 0, c = a/r , and s = b/r 
				* @param a first value
				* @param b second value
				* @param c = a/r
				* @param s = b/r
				* @param r = √a2 + b2
				*/
				template <typename _T1>
				inline void sym_ortho(_T1 a, _T1 b, _T1& c, _T1& s, _T1& r)
				{
					if (b == 0)
					{
						if (a == 0)
							c = 1;
						else 
							c = sgn<_T1>(a);
						r = std::abs(a);
					}
					else if (a == 0)
					{
						c = 0;
						s = sgn<_T1>(b);
						r = std::abs(b);
					}
					else if (std::abs(b) >= std::abs(a))
					{
						_T1 tau = a / b;
						s 		= sgn<_T1>(b) / std::sqrt(1 + tau * tau);
						c 		= s * tau;
						r 		= b / s;
					}
					else 
					{
						_T1 tau = b / a;
						c 		= sgn<_T1>(a) / std::sqrt(1 + tau * tau);
						s 		= c * tau;
						r 		= a / c;
					}
				}

				template <>
				inline void sym_ortho<std::complex<double>>(std::complex<double> a, std::complex<double> b, std::complex<double>& c, std::complex<double>& s, std::complex<double>& r)
				{
					double _c2, _s2, _r2;
					sym_ortho<double>(algebra::real(a), algebra::real(b), _c2, _s2, _r2);
					c = std::complex<double>(_c2, 0);
					s = std::complex<double>(_s2, 0);
					r = std::complex<double>(_r2, 0);
				}

				// #################################################################################################################################################

				// ---- MY MINRES_QLP SOLVER (symmetric matrices) and Gram Matrix ----
				template <typename _T1>
				inline arma::Col<_T1> minres_qlp(const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0			= nullptr,
										double _eps 				= 1.0e-6,
										size_t _max_iter 			= 1000,
										bool* _converged 			= nullptr, 
										double _reg 				= 0.0)
				{
					// !TODO: Implement the MINRES_QLP solver
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
				
				// #################################################################################################################################################

				/*
				* @brief MINRES_QLP solver for the Fisher matrix inversion. This method is used whenever the matrix can be
				* decomposed into the form S = \Delta O^* \Delta O, where \Delta O is the derivative of the observable with respect to the parameters.
				* The matrix S is symmetric and positive definite, so the MINRES_QLP method can be used.
				* The method is a preconditioned solver for approximating (A-\sigma I) * x = b, where A = S and b = F. 
				* Preconditioning makes it A = M^{-1/2} (A - \sigma I) M^{-1/2} x = M^{-1/2} b, where M is the preconditioner.
				* @ref S.-C. T. Choi and M. A. Saunders MINRES-QLP for Symmetric and Hermitian Linear Equations and Least-Squares Problems
				* @equation S_{ij} = <\Delta O^*_i \Delta O_j> / N
				* @param _DeltaO The matrix \Delta O.
				* @param _DeltaOConjT The matrix \Delta O^+.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _preconditioner The preconditioner for the MINRES_QLP method.
				* @param _eps The convergence criterion.
				* @param _max_iter The maximum number of iterations.
				* @param _converged The flag indicating if the solver converged.
				* @param _reg The regularization parameter. (A + \lambda I) x \approx b
				* @note Finally, note that the choice of parameter values can have a critical effect on the
						convergence of iterative solvers. While the default parameter values in MINRES-QLP
						work well in most tests, they may need to be fine-tuned by trial and error, and for some
						applications it may be worthwhile to implement full or partial reorthogonalization of
						the Lanczos vectors [Simon 1984].
				* @return The solution vector x.
				*/
				template <typename _T1>
				inline arma::Col<_T1> minres_qlp(const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0			= nullptr,
										Preconditioners::Preconditioner<_T1, true>* _preconditioner = nullptr,
										double _eps 				= 1.0e-6,
										size_t _max_iter 			= 1000,
										bool* _converged 			= nullptr, 
										double _reg 				= 0.0)
				{
					if (_preconditioner == nullptr)
						return minres_qlp<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);

					const size_t _n 	= _F.n_elem;										// number of elements			
					arma::Col<_T1> x_km3= (_x0 == nullptr) ? arma::Col<_T1>(_n, arma::fill::zeros) : *_x0; // initial vector x_{k-3}
					arma::Col<_T1> x_km2= arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector x_{k-2}
					arma::Col<_T1> x_km1= arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector x_{k-1}
					arma::Col<_T1> x_k	= arma::Col<_T1>(_n, arma::fill::zeros);			// final solution vector

					// initial values
					arma::Col<_T1> z_km2= arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector z0 
					arma::Col<_T1> z_km1 = _F; 												// initial vector z1
					arma::Col<_T1> z_k; 													// vector z_{k+1} in the loop - updated at each iteration
					arma::Col<_T1> q_k 	= _preconditioner->apply(z_km2);					// initial vector q1
					arma::Col<_T1> p_k 	= q_k;												// initial vector p1
							
					double _beta_km1 = 0.0, _beta_k = algebra::real(arma::cdot(_F, q_k)); 	// beta_0 = 0, beta_1 = sqrt(_F^T * q1)
		
					// Krylov vectors for Lanczos		
					double _alpha_k = 0.0;													// alpha_k = 0 - at each iteration
					double _rho_k = 0.0;													// rho_k = 0 - at each iteration

					// for the reflection
					double _delta_k = 0.0;													// use them as previous values of delta's
					double _c_km1_1 = -1.0, _c_km1_2 = -1.0, _c_km1_3 = -1.0, _c_k_1, _c_k_2, _c_k_3; // use them as previous values of c's
					double _s_km1_1 = 0.0, _s_km1_2 = 0.0, _s_km1_3 = 0.0, _s_k_1, _s_k_2, _s_k_3;	// use them as previous values of s's
					double _gamm_k = 0.0, _gamm_km1 = 0.0, _gamm_km2 = 0.0;					// use them as previous values of gamma's
					double _gamm_min = 0.0;
					double _eta_k = 0.0, _eta_km1 = 0.0, _eta_km2 = 0.0;						// use them as previous values of eta's
					double _eps_k = 0.0, _eps_km1 = 0.0;									// use them as previous values of eps's
					double _theta_k = 0.0, _theta_km1 = 0.0, _theta_km2;					// use them as previous values of theta's
					double _tau_k = 0.0, _tau_km1 = 0.0, _tau_km2;							// use them as previous values of tau's
					double _phi_k = 0.0, _phi_km1 = _beta_k;									// use them as previous values of phi's
					double _psi_k = 0.0, _psi_km1 = 0.0;										// use them as previous values of psi's
					double _A_k = 0.0, _A_km1 = 0.0;										// use them as previous values of A's
					double _om_k = 0.0, _om_km1 = 0.0;										// use them as previous values of omega's
					double _kappa_k = 0.0, _kappa_km1 = 0.0;								// use them as previous values of kappa's
					double _mu_k = 0.0, _mu_km1 = 0.0, _mu_km2 = 0.0, _mu_km3, _mu_km4 = 0.0;	// use them as previous values of mu'
					double _chi_k = 0.0, _chi_km1 = 0.0, _chi_km2 = 0.0, _chi_km3 = 0.0;		// use them as previous values of chi's
					arma::Col<_T1> _w_k(_n, arma::fill::zeros), _w_km1(_n, arma::fill::zeros), _w_km2(_n, arma::fill::zeros); // use them as previous values of w's

					// iterate until convergence, if the value is k+1 in the algorithm, we update the corresponding variables.
					// Remember, here the index starts from 0 (k = -1 is the iteration before the loop)
					for (int k = 0; k < _max_iter; ++k)
					{
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// preconditioned Lanczos 
						// MINRES and MINRES-QLP use the symmetric Lanczos process [Lanczos 1950] to reduce
						// A to a tridiagonal form Tk. The process is initialized with v0 ≡ 0, β1 = ‖b‖, and β1v1 = b.
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						
						p_k 				= matrixFreeMultiplication(_DeltaO, _DeltaOConjT, q_k, -1.0) - _reg * q_k;	// apply the matrix-vector multiplication 
						_alpha_k 			= algebra::real(arma::cdot(q_k, p_k)) / _beta_k / _beta_k;					// alpha_k = (q_k, p_k) / beta_k^2 - at each iteration
						z_k					= p_k / _beta_k - (_alpha_k / _beta_k) * z_km1 - (_beta_km1 / _beta_k) * z_km2;	// z_{k+1} = p_k / beta_k - (alpha_k / beta_k) * z_k - (beta_{k-1} / beta_k) * z_{k-1} - updated at each iteration 
						q_k 				= _preconditioner->apply(z_k);												// apply the preconditioner - can be updated now as is not used anymore
						//! new value of beta_k - update later
						double _beta_kp1 	= algebra::real(std::sqrt(arma::cdot(q_k, z_k)));							// beta_{k+1} = sqrt(q_k, z_k) - defined here, beta needs to be updated at each iteration
						// rule for ρk = ‖[αk βk+1 ]‖ else ρk = ‖[βk αk βk+1 ]‖
						if (k == 0)
							_rho_k 			= _alpha_k * _alpha_k + _beta_kp1 * _beta_kp1;
						else
							_rho_k 			= _alpha_k * _alpha_k + _beta_kp1 * _beta_kp1 + _beta_k * _beta_k;
						
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// update delta_k ((2) second stage)
						_delta_k 			= _c_km1_1 * _delta_k + _s_km1_1 * _alpha_k; 								// previous left reflection - at each iteration - is a second stage for this variable
						_gamm_k 			= _s_km1_1 * _delta_k - _c_km1_1 * _alpha_k; 								// [on middle two entries of Tk ek...] - at each iteration
						//! new value of eta_k - update later
						double _eps_k_p1		= _s_km1_1 * _beta_kp1; 												// [produces first two entries in Tk+1ek+1] - local variable
						//! new value of delta_k - update later
						double _delta_kp1 		= -_c_km1_1 * _beta_kp1; 												// update delta after the reflection
						
						// use symortho to get the new reflection
						// update _gamma_k ((2) second stage)
						sym_ortho(_gamm_k, _beta_kp1, _c_k_1, _s_k_1, _gamm_k); 										// [current left reflection]
						// update _gamma_km2 and _gamma_km1 ((5) fifth and (6) sixth stage)
						sym_ortho(_gamm_km2, _eps_k, _c_k_2, _s_k_2, _gamm_km2); 										// [First right reflection]		
						// update _theta_km1 ((2) second stage)
						_theta_km1 			= _c_k_2 * _theta_km1 + _s_k_2 * _delta_k; 									// use delta from (2) stage 
						// update delta_k ((3) third stage)
						_delta_k 			= _s_k_2 * _theta_km1 - _c_k_2 * _delta_k;
						_eta_k 				= _s_k_2 * _gamm_k;															// use gamma from (2) stage
						// update _gamma_k ((3) third stage)
						_gamm_k 			= -_c_k_2 * _gamm_k;
						// update _gamma_km1 ((4) fourth stage and (5) fifth stage)
						sym_ortho(_gamm_km1, _delta_k, _c_k_3, _s_k_3, _gamm_km1); 										// [Second right reflection]
						_theta_k 			= _s_k_3 * _gamm_k; 														// use gamma from (3) stage	
						// update _gamma_k ((4) fourth stage)
						_gamm_k 			= -_c_k_3 * _gamm_k;														// [to zero out δ(3)k]
						_tau_k 				= _c_k_1 * _phi_km1; 														// [Last element of tk]
						_phi_k 				= _s_k_1 * _phi_km1; 														// [Update ‖ ¯r k‖, ‖ ˜A¯r k−1‖]
						_psi_km1 			= _phi_km1 * (_gamm_k * _gamm_k + _delta_kp1 * _delta_kp1);

						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k == 0)
							_gamm_min 		= _gamm_k;
						else
							_gamm_min 		= std::min(_gamm_km2, std::min(_gamm_km1, std::min(_gamm_min, std::abs(_gamm_k))));
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!

						_A_k 				= std::max(_A_km1, std::max(_rho_k, std::max(_gamm_km2, std::max(_gamm_km1, std::abs(_gamm_k)))));	// [Update ‖A‖]
						_om_k 				= _om_km1 * _om_km1 + _tau_k * _tau_k;										// [Update ‖ ˜Axk‖, cond( ˜A)]
						_kappa_k 			= _A_k / _gamm_min;															// [Update wk−2, wk−1, wk]
						_w_k 				= -(_c_k_2 / _beta_k) * q_k + _s_k_2 * _w_km2;								// [Update wk], use w_km2 from (3) stage
						_w_km2 				= (_s_k_2 / _beta_k) * q_k + _c_k_2 * _w_km2;								// [Update wk−2], use w_km2 from (3) stage
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 1)
						{
							// update w_k ((2) second stage)
							_w_k = _s_k_3 * _w_km1 - _c_k_3 * _w_k;														// [Update wk], use w_km1 from (2) stage
							_w_km1 = _c_k_3 * _w_km1 + _s_k_3 * _w_k;													// [Update wk−1], use w_km1 from (2) stage
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 1)
						{
							_mu_km2 = (_tau_km2 - _eta_km2 * _mu_km4 - _theta_km2 * _mu_km3) / _gamm_km2;				// [Update µk−2]
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 0)
						{
							_mu_km1 = (_tau_km1 - _eta_km1 * _mu_km3 - _theta_km1 * _mu_km2) / _gamm_km1;				// [Update µk−1]
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (std::abs(_gamm_k) > _eps)
						{
							_mu_k = (_tau_k - _eta_k * _mu_km2 - _theta_k * _mu_km1) / _gamm_k;							// [Update µk]
						} else {
							_mu_k = 0.0;
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						
						// update the solution
						x_km2 		= x_km3 + _mu_km2 * _w_km2;															// [Update xk−2]
						x_k 		= x_km2 + _mu_km1 * _w_km1 + _mu_k * _w_k;											// [Update xk]

						// check for convergence
						_chi_km2 	= _chi_km3 * _chi_km3 + _mu_km2 * _mu_km2;											// [Update ‖xk−2‖]
						_chi_k 		= _chi_km2 * _chi_km2 + _mu_km1 * _mu_km1 + _mu_k * _mu_k;							// [Update ‖xk‖]
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!

						// update the values for the next iteration
						{
							// values of delta
							_delta_k = _delta_kp1;
							// values of beta
							_beta_km1 = _beta_k;
							_beta_k = _beta_kp1;
							// values of c - tridiagonal matrix
							_c_km1_1 = _c_k_1;
							_c_km1_2 = _c_k_2;
							_c_km1_3 = _c_k_3;
							// values of s - tridiagonal matrix
							_s_km1_1 = _s_k_1;
							_s_km1_2 = _s_k_2;
							_s_km1_3 = _s_k_3;
							// values of gamma
							_gamm_km2 = _gamm_km1;
							_gamm_km1 = _gamm_k;
							_gamm_min = _gamm_k;
							// values of delta
							_eta_km2 = _eta_km1;
							_eta_km1 = _eta_k;
							// values of eps
							_eps_km1 = _eps_k;
							_eps_k = _eps_k_p1;
							// values of theta
							_theta_km2 = _theta_km1;
							_theta_km1 = _theta_k;
							// values of tau
							_tau_km1 = _tau_k;
							_tau_km2 = _tau_km1;
							// values of phi
							_phi_km1 = _phi_k;
							// values of psi
							_psi_km1 = _psi_k;
							// values of A
							_A_km1 = _A_k;
							// values of omega
							_om_km1 = _om_k;
							// values of kappa
							_kappa_km1 = _kappa_k;
							// values of mu
							_mu_km4 = _mu_km3;
							_mu_km3 = _mu_km2;
							_mu_km2 = _mu_km1;
							_mu_km1 = _mu_k;
							// values of chi
							_chi_km3 = _chi_km2;
							_chi_km2 = _chi_km1;
							_chi_km1 = _chi_k;
							// vector W
							_w_km2 = _w_km1;
							_w_km1 = _w_k;
						}

						if (std::sqrt(_chi_k) < _eps)
						{
							if (_converged != nullptr)
								*_converged = true;
							return x_k;
						}
					}
					
					std::cerr << "\t\t\tMINRES_QLP solver did not converge." << std::endl;
					if (_converged != nullptr)
						*_converged = false;
					return x_k;
				}
			};

			// #################################################################################################################################################

			enum class Type {
				ARMA,
				ConjugateGradient,
				MINRES_QLP, 
				PseudoInverse,
				Direct
			};
			
			// -----------------------------------------------------------------------------------------------------------------------------------------


			template <typename _T1>
			inline arma::Col<_T1> solve(Type _type,
										const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0										= nullptr,
										Preconditioners::Preconditioner<_T1, true>* _preconditioner = nullptr,
										double _eps 											= 1.0e-6,
										size_t _max_iter 										= 1000,
										bool* _converged 										= nullptr, 
										double _reg 											= 0.0)
			{
				switch (_type) 
				{
				case Type::ARMA:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::solve(_S, _F);
				}
				case Type::ConjugateGradient:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				case Type::MINRES_QLP:
					return MINRES_QLP::minres_qlp<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				case Type::PseudoInverse:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::pinv(_S, _eps) * _DeltaOConjT * _F;
				}
				case Type::Direct:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::inv(_S) * _F;
				}
				default:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}
			}

			template <typename _T1>
			inline arma::Col<_T1> solve(int _type,
										const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0										= nullptr,
										Preconditioners::Preconditioner<_T1, true>* _preconditioner = nullptr,
										double _eps 											= 1.0e-6,
										size_t _max_iter 										= 1000,
										bool* _converged 										= nullptr, 
										double _reg 											= 0.0) 
			{
				return solve<_T1>(static_cast<Type>(_type), _DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
			}

			template <typename _T1>
			inline arma::Col<_T1> solve(Type _type,
										const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0										= nullptr,
										double _eps 											= 1.0e-6,
										size_t _max_iter 										= 1000,
										bool* _converged 										= nullptr, 
										double _reg 											= 0.0)
			{
				switch (_type) 
				{
				case Type::ARMA:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::solve(_S, _F);
				}
				case Type::ConjugateGradient:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				case Type::MINRES_QLP:
					return MINRES_QLP::minres_qlp<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				case Type::PseudoInverse:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::pinv(_S, _eps) * _DeltaOConjT * _F;
				}
				case Type::Direct:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::inv(_S) * _F;
				}
				default:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
			}
			
			template <typename _T1>
			inline arma::Col<_T1> solve(int _type,
										const arma::Mat<_T1>& _DeltaO,
										const arma::Mat<_T1>& _DeltaOConjT,
										const arma::Col<_T1>& _F,
										arma::Col<_T1>* _x0										= nullptr,
										double _eps 											= 1.0e-6,
										size_t _max_iter 										= 1000,
										bool* _converged 										= nullptr, 
										double _reg 											= 0.0)
			{
				return solve<_T1>(static_cast<Type>(_type), _DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
			}

			// #################################################################################################################################################

			inline std::string name(Type _type)
			{
				switch (_type) 
				{
				case Type::ARMA:
					return "ARMA";
				case Type::ConjugateGradient:
					return "Conjugate Gradient";
				case Type::MINRES_QLP:
					return "MINRES_QLP";
				case Type::PseudoInverse:
					return "Pseudo Inverse";
				case Type::Direct:
					return "Direct";
				default:
					return "Conjugate Gradient";
				}
			}

			inline std::string name(int _type)
			{
				return name(static_cast<Type>(_type));
			} 

			// #################################################################################################################################################
		};
		
		// #################################################################################################################################################

		namespace Symmetric
		{	
			// #################################################################################################################################################

			template <typename _T>
			inline arma::Col<_T> matrixMultiplication(const arma::Mat<_T>& A, const arma::Col<_T>& x, const double _reg = 0.0)
			{
				arma::Col<_T> _intermediate = A * x;
				if (_reg > 0.0)
					return A.t() * _intermediate + _reg * x;
				return A.t() * _intermediate;
			}
			
			// #################################################################################################################################################

			template <typename _T>
			inline arma::Col<_T> solve_arma(const arma::Mat<_T>& A, 
											const arma::Col<_T>& b)
			{
				return arma::solve(A, b, arma::solve_opts::likely_sympd);
			}
			
			// #################################################################################################################################################

			namespace CG
			{
				// -----------------------------------------------------------------------------------------------------------------------------------------

				template <typename _T>
				inline arma::Col<_T> conjugate_gradient(const arma::Mat<_T>& _A,
														const arma::Col<_T>& _F,
														arma::Col<_T>* _x0										= nullptr,
														Preconditioners::Preconditioner<_T>* _preconditioner 	= nullptr,
														double _eps 											= 1.0e-6,
														size_t _max_iter 										= 1000,
														bool* _converged 										= nullptr, 
														double _reg 											= 0.0)
				{
					if (_preconditioner == nullptr)
						return conjugate_gradient<_T>(_A, _F, _x0, _eps, _max_iter, _converged, _reg);

					// set the initial values for the solver
					arma::Col<_T> x 	= (_x0 == nullptr) ? arma::Col<_T>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T> r 	= _F - matrixMultiplication(_A, x, _reg);
					arma::Col<_T> z 	= _preconditioner->apply(r);
					arma::Col<_T> p 	= z;
					arma::Col<_T> Ap;		// matrix-vector multiplication result

					_T rs_old 			= arma::cdot(r, z);
					_T initial_rs		= std::abs(rs_old);

					// iterate until convergence
					for (size_t i = 0; i < _max_iter; ++i)
					{
						Ap 					= matrixMultiplication(_A, p, _reg);
						_T alpha 			= rs_old / arma::cdot(p, Ap);
						x 					+= alpha * p;
						r 					-= alpha * Ap;

						// Check for convergence
						if (std::abs(arma::cdot(r, r)) < _eps) {
							if (_converged != nullptr)
								*_converged = true;
							return x;
						}
						z 					= _preconditioner->apply(r);
						_T rs_new 			= arma::cdot(r, z);
						p 					= z + (rs_new / rs_old) * p;
						rs_old 				= rs_new;
					}

					std::cerr << "\t\t\tConjugate gradient solver did not converge." << std::endl;
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				// conjugate gradient solver for symmetric matrices
				template <typename _T>
				inline arma::Col<_T> conjugate_gradient(const arma::Mat<_T>& _A,
														const arma::Col<_T>& _F,
														arma::Col<_T>* _x0			= nullptr,
														double _eps 				= 1.0e-6,
														size_t _max_iter 			= 1000,
														bool* _converged 			= nullptr, 
														double _reg 				= 0.0)
				{
					// set the initial values for the solver
					arma::Col<_T> x 	= (_x0 == nullptr) ? arma::Col<_T>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T> r 	= _F - matrixMultiplication(_A, x, _reg);
					_T rs_old 			= arma::cdot(r, r);

					// check for convergence already
					if (std::abs(rs_old) < _eps) {
						if (_converged != nullptr)
							*_converged = true;
						return x;
					}

					// create the search direction vector
					arma::Col<_T> p 	= r;
					arma::Col<_T> Ap;		// matrix-vector multiplication result

					// iterate until convergence
					for (size_t i = 0; i < _max_iter; ++i)
					{
						Ap 					= matrixMultiplication(_A, p, _reg);
						_T alpha 			= rs_old / arma::cdot(p, Ap);
						x 					+= alpha * p;
						r 					-= alpha * Ap;
						_T rs_new 			= arma::cdot(r, r);

						// Check for convergence
						if (std::abs(rs_new) < _eps) {
							if (_converged != nullptr)
								*_converged = true;
							return x;
						}
						
						// update the search direction
						p 					= r + (rs_new / rs_old) * p;
						rs_old 				= rs_new;
					}

					std::cerr << "\t\t\tConjugate gradient solver did not converge." << std::endl;
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

			};

			// #################################################################################################################################################
		};

		// #################################################################################################################################################

		namespace General 
		{	
			// #################################################################################################################################################

			// ------------ ARMA SOLVER ------------
			template <typename _T>
			arma::Col<_T> solve_arma(	const arma::Mat<_T>& A, 
										const arma::Col<_T>& b, 
										auto _opts = arma::solve_opts::likely_sympd)
			{
				return arma::solve(A, b, _opts);
			}

			// #################################################################################################################################################
		};

		// #################################################################################################################################################

		
	};


	// #################################################################################################################################################
	
	// ################################################################### PFAFFIANS ###################################################################

	// #################################################################################################################################################
	namespace Pfaffian
	{
		enum class PfaffianAlgorithms
		{
			ParlettReid,
			Householder,
			Schur,
			Hessenberg,
			Recursive
		};

		/*
		* @brief Calculate the Pfaffian of a skew square matrix A. Use the recursive definition.
		* @link https://s3.amazonaws.com/researchcompendia_prod/articles/2f85f444b9e340246d9991177acf9732-2013-12-23-02-19-16/a30-wimmer.pdf
		* @param A skew-symmetric matrix
		* @param N size of the matrix
		* @returns the Pfaffian of a skew-symmetric matrix A
		*/
		template <typename _T>
		_T pfaffian_r(const arma::Mat<_T>& A, arma::u64 N)
		{
			if (N == 0)
				return _T(1.0);
			else if (N == 1)
				return _T(0.0);
			else
			{
				_T pfa = 0.0;
				for (arma::u64 i = 2; i <= N; i++)
				{
					arma::Mat<_T> temp = A;
					_T _sign = (i % 2 == 0) ? 1. : -1.;
					temp.shed_col(i - 1);
					temp.shed_row(i - 1);
					temp.shed_row(0);
					if (N > 2)
						temp.shed_col(0);
					pfa += _sign * A(0, i - 1) * pfaffian_r(temp, N - 2);
				}
				return pfa;
			}
		}

		// #################################################################################################################################################

		/*
		* @brief Computing the Pfaffian of a skew-symmetric matrix. Using the fact that for an arbitrary skew-symmetric matrix,
		* the pfaffian Pf(B A B^T ) = det(B)Pf(A). This is done via Hessenberg decomposition.
		* @param A skew-symmetric matrix
		* @param N size of the matrix
		* @returns the Pfaffian of a skew-symmetric matrix A
		*/
		template <typename _T>
		_T pfaffian_hess(const arma::Mat<_T>& A, arma::u64 N)
		{
			// calculate the Upper Hessenberg decomposition. Take the upper diagonal only
			arma::Mat<_T> H, Q;
			arma::hess(Q, H, A);
			return arma::det(Q) * arma::prod(arma::Col<_T>(H.diag(1)).elem(arma::regspace<arma::uvec>(0, N - 1, 2)));
		}

		// #################################################################################################################################################

		/*
		* @brief Computing the Pfaffian of a skew-symmetric matrix. Using the fact that for an arbitrary skew-symmetric matrix,
		* the pfaffian Pf(B A B^T ) = det(B)Pf(A). This is done via Parlett-Reid algorithm.
		* @param A skew-symmetric matrix
		* @param N size of the matrix
		* @returns the Pfaffian of a skew-symmetric matrix A
		*/
		template <typename _T>
		_T pfaffian_p(arma::Mat<_T> A, arma::u64 N)
		{
			if(!(A.n_rows == A.n_cols && A.n_rows == N && N > 0))
				throw std::runtime_error("Error: Matrix size must be even for Pfaffian calculation.");
	#ifdef _DEBUG
			// Check if it's skew-symmetric
			//if(!(((A + A.st()).max()) < 1e-14))
	#endif
			// quick return if possible
			if (N % 2 == 1)
				return 0; 
			// work on a copy of A

			_T pfaffian = 1.0;
			for (arma::u64 k = 0; k < N - 1; k += 2)
			{
				// First, find the largest entry in A[k + 1:, k] and
				// permute it to A[k + 1, k]
				auto kp = k + 1 + arma::abs(A.col(k).subvec(k + 1, N - 1)).index_max();

				// Check if we need to pivot
				if (kp != k + 1)
				{
					// interchange rows k + 1 and kp
					A.swap_rows(k + 1, kp);

					// Then interchange columns k + 1 and kp
					A.swap_cols(k + 1, kp);

					// every interchange corresponds to a "-" in det(P)
					pfaffian *= -1;
				}

				// Now form the Gauss vector
				if (A(k + 1, k) != 0.0)
				{
					pfaffian *=	A(k, k + 1);
					if (k + 2 < N)
					{
						arma::Row<_T> tau	=	A.row(k).subvec(k + 2, N - 1) / A(k, k + 1);
						// Update the matrix block A(k + 2:, k + 2)
						const auto col				=	A.col(k + 1).subvec(k + 2, N - 1);
						auto subMat 				=	A.submat(k + 2, k + 2, N - 1, N - 1);	
						const auto col_times_row	=	outer(col, tau);
						const auto row_times_col	=	outer(tau, col);
						//col_times_row.print("COL * TAU");
						//row_times_col.print("TAU * COL");
						subMat				+=	row_times_col;
						subMat				-=	col_times_row;
					}
				}
				// if we encounter a zero on the super/subdiagonal, the Pfaffian is 0
				else
					return 0.0;
			}
			return pfaffian;
		}

		// #################################################################################################################################################
	
		/*
		* @brief Computing the Pfaffian of a skew-symmetric matrix. Using the fact that for an arbitrary skew-symmetric matrix,
		* the pfaffian Pf(B A B^T ) = det(B)Pf(A). This is done via Schur decomposition.
		* @param A skew-symmetric matrix
		* @param N size of the matrix
		* @returns the Pfaffian of a skew-symmetric matrix A
		*/
		template <typename _T>
		_T pfaffian_s(arma::Mat<_T> A, arma::u64 N)
		{
			arma::Mat<_T> U, S;
			arma::schur(U, S, A);
			return arma::det(U) * arma::prod(arma::Col<_T>(S.diag(1)).elem(arma::regspace<arma::uvec>(0, N - 1, 2)));
		}
	
		// #################################################################################################################################################

		template <typename _T>
		_T pfaffian(const arma::Mat<_T>& A, arma::u64 N, PfaffianAlgorithms _alg = PfaffianAlgorithms::ParlettReid)
		{
	//#ifdef _DEBUG
	//		A.save(arma::hdf5_name("A.h5"));
	//#endif
			switch (_alg)
			{
			case PfaffianAlgorithms::ParlettReid:
				return pfaffian_p<_T>(A, N);
			case PfaffianAlgorithms::Householder:
				//LOGINFO("Householder Pfaffian algorithm not implemented yet.", LOG_TYPES::ERROR, 2);
				return 0;
			case PfaffianAlgorithms::Schur:
				return pfaffian_s<_T>(A, N);
			case PfaffianAlgorithms::Hessenberg:
				return pfaffian_hess<_T>(A, N);
			case PfaffianAlgorithms::Recursive:
				return pfaffian_r(A, N);
			default:
				return pfaffian_p<_T>(A, N);
			}
		}
		
		// #################################################################################################################################################
		
		/*
		* @brief Update the Pfaffian of a skew-symmetric matrix after a row and column update.
		* @param _pffA current Pfaffian
		* @param _Ainv inverse of the original matrix
		* @returns the Pfaffian of an updated skew-symmetric matrix
		*/
		template <typename _Tp, typename _T>
		_Tp pfaffian_upd_row_n_col(_Tp _pffA, const _T& _Ainv_row, const _T& _updRow)
		{
			return -_pffA * algebra::cast<_Tp>(arma::dot(_Ainv_row, _updRow));
		}


	};

	// #################################################################################################################################################

	/*
	* @brief Calculate the inverse of a skew-symmetric matrix using the Scheher-Morrison formula.
	* @param _Ainv inverse of the matrix before the update
	* @param _updIdx index of the updated row
	* @param _updRow updated row
	* @returns the inverse of the updated skew-symmetric matrix
	*/
	template<typename _T, typename _T2 = arma::subview_col<_T>>
	arma::Mat<_T> scherman_morrison_skew(const arma::Mat<_T>& _Ainv, uint _updIdx, const _T2& _updRow)
	{
		auto _out					= _Ainv;
		// precalculate all the dotproducts
		const arma::Col<_T> _dots	= _Ainv * _updRow.as_col();

		// precalculate the dot product inverse for updated row
		const auto _dotProductInv	= 1.0 / _dots(_updIdx);

		// go through the update
		for(int i = 0; i < _Ainv .n_rows; i++)
		{
			auto _d_i_alpha = (i == _updIdx) ? 1.0 : 0.0;
			for(int j = 0; j < _Ainv.n_cols; j++)
			{
				auto _d_j_alpha = (j == _updIdx) ? 1.0 : 0.0;
				_out(i, j) += _dotProductInv * ((_d_i_alpha - _dots(i)) * _Ainv(_updIdx, j) + (_dots(j) - _d_j_alpha) * _Ainv(_updIdx, i));
				// why????!!!!
				if(_d_i_alpha || _d_j_alpha)
					_out(i, j) *= -1;
			}
		}
		return _out;
	}

	// #################################################################################################################################################
	
	// ############################################################# MATRIX DECOMPOSITIONS #############################################################
	
	// #################################################################################################################################################	

	template<typename _T>
	class UDT
	{
	public:
		arma::Mat<_T> U;
		arma::Col<_T> D;			// here we will put D vector - diagonal part of R
		arma::Col<_T> Di;			// here we will put D vector inverse
		arma::Mat<_T> T;
		arma::Col<_T> Db;			// Dmax - for spectral values scaling
		arma::Col<_T> Ds;			// Dmin - for spectral values scaling
	public:
		virtual ~UDT() {
			//LOGINFO("Deleting base UDT class", LOG_TYPES::INFO, 2);
		}
		UDT()
		{
			//LOGINFO("Building base UDT class", LOG_TYPES::INFO, 2);
		}
		UDT(const arma::Mat<_T>& M) : UDT<_T>()
		{
			this->U.zeros(M.n_rows, M.n_cols);
			this->T.zeros(M.n_rows, M.n_cols);
			this->D		= ZEROV(M.n_rows);
			this->Di	= ZEROV(M.n_rows);
		}
		UDT(const UDT<_T>& o) : U(o.U), D(o.D), Di(o.Di), Db(o.Db), Ds(o.Ds), T(o.T) 
		{
			//LOGINFO("Building base UDT class", LOG_TYPES::INFO, 2);
		};
		UDT(UDT<_T>&& o) : U(std::move(o.U)), D(std::move(o.D)), Di(std::move(o.Di)), Db(std::move(o.Db)), Ds(std::move(o.Ds)), T(std::move(o.T)) {};
		UDT(const arma::Mat<_T>& u, const arma::Col<_T>& d, const arma::Mat<_T>& t) : U(u), D(d), Di(1.0/d), Db(d), Ds(d), T(t) {};
		UDT(arma::Mat<_T>&& u, arma::Col<_T>&& d, arma::Mat<_T>&& t) : U(u), D(d), Di(1.0/d), Ds(d), Db(d), T(t) {};

		virtual void decompose()									= 0;
		virtual void decompose(const arma::Mat<_T>& M)				= 0;
		virtual arma::Mat<_T> eval()								{ return U * DIAG(D) * T;  };
	
		// spectral decomposition using method by Loh et. al. 
		virtual void loh()											= 0;
		virtual void loh_inv()										= 0;
		virtual void loh_inplace()									= 0;

		/*
		* @brief copy assignment
		*/
		UDT<_T>& operator=(const UDT<_T>& o) {
			U	= o.U;
			D	= o.D;
			Di	= o.Di;
			Ds	= o.Ds;
			Db	= o.Db;
			T	= o.T;
			return *this;
		};

		/*
		* @brief move assignment
		*/
		UDT<_T>& operator=(UDT<_T>&& o) {
			U	= std::move(o.U);
			D	= std::move(o.D);
			Di	= std::move(o.Di);
			Ds	= std::move(o.Ds);
			Db	= std::move(o.Db);
			T	= std::move(o.T);
			return *this;
		};
	
		// ###############################################################################################################
		// ############################################## MATRIX OPERATIONS ##############################################
		// ###############################################################################################################
 
		// ----------------------------------- INVERSE -----------------------------------

		/*
		* @brief Calculates the inverse of the UDT decomposition of a matrix. With return.
		* @returns the inverse of a matrix set by current UDT decomposition
		*/
		arma::Mat<_T> inv()											{ return arma::solve(T, Di) * U.t(); };

		/*
		* @brief Calculates the inverse of the UDT decomposition of a matrix. 
		* @param M matrix to set the inverse onto.
		*/
		void inv(arma::Mat<_T>& M)									{ M = arma::solve(T, Di) * U.t(); };

		// ------------------------------- MULTIPLICATION --------------------------------

		/*
		* @brief Stabilized multiplication of two `UDT` decompositions.
		* @return UDT factorization object
		*/
		//static UDT<_T> factMult(const UDT<_T>& A, const UDT<_T>& B){
		//	UDT<_T> ret = A;
		//	ret.factMult(B);
		//	return ret;
		//}

		// -------------------------------------------------------------------------------

		/*
		* @brief Stabilized multiplication of two `UDT` decompositions.
		* @param B second UDT decomposition
		*/
		void factMult(const UDT<_T>& B){
			arma::Mat<_T> mat	= T * B.U; 		// Tl * Ur
			mat					= D * mat;		// Rl * (*up)
			mat					= mat * B.D;
			decompose(mat);
		} 
	
		virtual void factMult(const arma::Mat<_T>& Ml)				= 0;

		// -------------------------------------------------------------------------------

		// --------------------------------- (1+A)^(-1) ----------------------------------
		virtual arma::Mat<_T> inv1P()								= 0;
		virtual void inv1P(arma::Mat<_T>& setMat)					= 0;

		// --------------------------------- (A+B)^(-1) ----------------------------------
		virtual arma::Mat<_T> invSum(UDT<_T>* right)				= 0;
		virtual void invSum(UDT<_T>* right, arma::Mat<_T>& setMat)	= 0;
	};

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	/*
	* @brief UDT decomposition using the QR decomposition
	* @cite doi:10.1016/j.laa.2010.06.023
	*/
	template<typename _T>
	class UDT_QR : public UDT<_T>
	{
	protected:
		// arma::Mat<_T> U = Q;		// in this case the U matrix serves as Q	
		arma::Mat<_T> R;			// right triangular matrix
		arma::umat P;				// permutation matrix 
	public:
		~UDT_QR()
		{
			//LOGINFO("Deleting UDT QR class", LOG_TYPES::INFO, 2);
		}
		UDT_QR()
		{
			//LOGINFO("Building QR UDT class", LOG_TYPES::INFO, 2);
		}
		UDT_QR(const arma::Mat<_T>& M)
			: UDT<_T>(M)
		{
			//LOGINFO("Building QR UDT class", LOG_TYPES::INFO, 3);
			this->R.zeros(M.n_rows, M.n_cols);
			this->P.zeros(M.n_rows, M.n_cols);
			this->Db	= ZEROV(M.col(0).n_rows);
			this->Ds	= ZEROV(M.col(0).n_rows);
		};
		UDT_QR(const arma::Mat<_T>& q, const arma::Mat<_T>& r, const arma::umat& p)
			: R(r), P(p), UDT<_T>(q, arma::ones(q.n_rows), ZEROM(q.n_rows))
		{
			decompose();
			this->Db	= ZEROV(q.col(0).n_rows);
			this->Ds	= ZEROV(q.col(0).n_rows);
		};
		UDT_QR(const UDT_QR<_T>& o)
			: R(o.R), P(o.P), UDT<_T>(o) {};
		UDT_QR(UDT_QR<_T>&& o) noexcept 
			: R(std::move(o.R)), P(std::move(o.P)), UDT<_T>(std::move(o)) {};

		/*
		* @brief copy assignment operator
		*/
		UDT_QR<_T>& operator=(const UDT_QR<_T>& o) { 
			UDT<_T>::operator=(o);
			R			= o.R;
			P			= o.P;
			this->Db	= o.Db;
			this->Ds	= o.Ds;
			return *this;
		}

		/*
		* @brief move assignment operator
		*/
		UDT_QR<_T>& operator=(UDT_QR<_T>&& o) { 
			UDT<_T>::operator=(std::move(o));
			R			= std::move(o.R);
			P			= std::move(o.P);
			this->Db	= std::move(o.Db);
			this->Ds	= std::move(o.Ds);
			return *this;
		}

		// ###############################################################################################################
		// ############################################# MATRIX DECOMPOSING ##############################################
		// ###############################################################################################################
	
		/*
		* @brief Create a decomposition using preset matrices
		*/
		void decompose() override {
			// inverse the vector D during setting
			this->D		=	R.diag();
			this->Di	=	1.0 / this->D;
			this->T		=	((DIAG(this->Di)) * this->R) * this->P.t();
		}
	
		/*
		* @brief Create a decomposition
		* @param M Matrix to decompose
		*/
		void decompose(const arma::Mat<_T>& M) override {
			if (!arma::qr(this->U, this->R, this->P, M)) 
				throw "Decomposition failed\n";
			decompose();
		}

		// ###############################################################################################################
		// ############################################# MATRIX STABILIZING ##############################################
		// ###############################################################################################################

		/*
		* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
		*/
		void loh() override{	
			for (auto i = 0; i < this->R.n_rows; i++)
			{
				if (abs(this->R(i, i)) > 1.0) {
					this->Db(i) = this->R(i, i);	// max (R(i,i), 1)
					this->Ds(i) = 1.0;				// min (R(i,i), 1)
				}
				else {
					this->Db(i) = 1.0;
					this->Ds(i) = this->R(i, i);
				}
			}
		}
	
		/*
		* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
		* @warning Saves the inverse to Db = max[R(i,i),1]!
		*/
		void loh_inv() override{
			for (auto i = 0; i < R.n_rows; i++)
			{
				if (abs(this->D(i)) > 1.0) {
					this->Db(i) = this->Di(i);		// max (R(i,i), 1) - but save the inverse
					this->Ds(i) = 1.0;				// min (R(i,i), 1)
				}
				else {
					this->Db(i) = 1.0;				// 
					this->Ds(i) = this->D(i);
				}
			}
		}

		/*
		* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
		* @attention ->(1, R) -> we save that in R -> therefore change is only made to set R(i,i) to 1
		* @attention ->max(1, R) -> we set that as an inverse onto D already
		* @warning Changes D and R!
		*/
		void loh_inplace() override {
			for (int i = 0; i < R.n_rows; i++){
				if (abs(R(i, i)) > 1)
					R(i, i) = 1;				// min(1,R(i,i))
				else 							// (abs(R(i, i)) <= 1)
					this->Di(i) = 1;			// inv of max(1,R(i,i)) because Di is already an inverse
			}
		}

		// ###############################################################################################################
		// ############################################ MATRIX MULTIPLICATION ############################################
		// ###############################################################################################################	

		/*
		* @brief Multiply the UDT decomposition by a matrix from the left
		* @param Ml left matrix
		* @link https://github.com/carstenbauer/StableDQMC.jl/blob/master/src/qr_udt.jl
		*/
		void factMult(const arma::Mat<_T>& Ml) override {
			if (!arma::qr(this->U, this->R, this->P, (Ml * this->U) * DIAG(this->R))) 
				throw "decomposition failed\n";
			// inverse during setting
			this->D		=	R.diag();
			this->Di	=	1.0 / this->D;
			// premultiply old T by new T from left
			this->T		=	((DIAG(this->Di) * this->R) * this->P.t()) * this->T;
		}
	
		/*
		* @brief (UDT + 1)^(-1) with QR decomposition.
		*/
		arma::Mat<_T> inv1P() override {
			// decompose first
			loh_inv();
			return arma::solve(DIAG(this->Db) * this->U.st() + DIAG(this->Ds) * this->T, DIAG(this->Db) * this->U.st());
			//this->loh();
			//return arma::solve(arma::inv(DIAG(this->Db)) * this->U.st() + (DIAG(this->Ds) * this->T), arma::inv(DIAG(Db)) * this->U.st());
			// without the decomposition
			//return arma::inv(DIAG(this->Di) * this->U.st() + DIAG(this->D) * this->T) * (DIAG(this->Di) * this->U.st());
		}
	
		void inv1P(arma::Mat<_T>& setMat) override { setMat = inv1P(); };

		// ################################################## (A+B)^(-1) #################################################
	
		/*
		* @brief Stabilized calculation of [UaDaTa + UbDbTb]^{-1}
		*/
		arma::Mat<_T> invSum(UDT<_T>* right) override {
			// calculate loh decomposition
			loh();
			right->loh();

			// dimension
			const auto d = this->Ds.n_elem;

			// matL = D_min_a * Ta * Tb^{-1} / D_max_b
			arma::Mat<_T> matL = this->T * arma::inv(arma::trimatu(right->T));
			for(int i = 0; i < d; i++)
				for(int j = 0; j < d; j++)
					matL(i,j) *= this->Ds(i) / right->Db(j);

			// matR = 1/(D_max_a) * Ua^\\dag * Ub * D_min_b
			arma::Mat<_T> matR = this->U.t() * right->U;
			for(int i = 0; i < d; i++)
				for(int j = 0; j < d; j++)
					matL(i,j) *= right->Ds(i) / this->Db(j);

			// add two matrices
			matL += matR;

			// create inner decomposition arma::solve(T, Di)
			UDT_QR<_T> inner(matL);
			matR = DIAG(inner.D) * inner.T;//arma::solve(inner.T, DIAG(inner.Di)); //arma::inv(arma::trimatu() * DIAG(inner.Di);
			matR = arma::solve(matR, this->U.t());
			for(int i = 0; i < d; i++)
				for(int j = 0; j < d; j++)
					matR(i,j) /= right->Db(i) * this->Db(j);
		
			// decompose again
			inner.decompose(matR);
			inner.U = arma::solve(arma::trimatu(right->T), inner.U);
			inner.T = inner.T * this->U.t();

			// return evaluated one
			return inner.eval();
		}
		void invSum(UDT<_T>* right, arma::Mat<_T>& setMat) override
		{
			setMat = invSum(right);
		}

	};
};

// dynamic bitset
#include "Include/str.h"
#include "Include/directories.h"

// ###################################################### S A V E R ######################################################

/*
* @brief Save the algebraic matrix to a file with a specific path. The file can be in binary, text or HDF5 format.
* @param _path path to the file
* @param _file name of the file
* @param _toSave matrix to save
* @param _db name of the database in HDF5 file
* @param _app append to the file?
* @returns true if the file was saved
*/
template <HasMatrixType _T>
inline bool saveAlgebraic(const std::string& _path, const std::string& _file, const _T& _toSave, const std::string& _db = "weights", bool _app = false)
{
#ifdef _DEBUG
	//LOGINFO(_path + _file, LOG_TYPES::INFO, 3);
#endif
	createDir(_path);
	bool _isSaved	= false;
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
	{
		if(!_app)
			_isSaved	= _toSave.save(arma::hdf5_name(_path + _file, _db));
		else
			_isSaved	= _toSave.save(arma::hdf5_name(_path + _file, _db, arma::hdf5_opts::append));
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	else if (endsWith(_file, ".bin"))
#endif
		_isSaved	= _toSave.save(_path + _file);
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	else if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		if(!_app)
			_isSaved	= _toSave.save(_path + _file, arma::arma_ascii);
		else
		{
			std::ofstream _out;
			try
			{
				_out.open(_path + _file, std::ios::app);
				_isSaved = _out.is_open();
			}
			catch(std::exception& e)
			{
				//LOGINFO(e.what(), LOG_TYPES::ERROR, 2);
			}
			_out << _toSave;
			_out.close();
		}
	}
	return _isSaved;
}

template<typename _T>
	requires HasArmaVectorType<_T>
inline bool saveAlgebraic(const std::string& _path, const std::string& _file, const _T& _toSaver, const std::string& _db = "weights", bool _app = false)
{
#ifdef _DEBUG
	//LOGINFO(_path + _file, LOG_TYPES::INFO, 3);
#endif
	createDir(_path);
	using _Tp		= typename _T::elem_type;
	bool _isSaved	= false;
	auto _toSave	= (std::is_same_v<_T, arma::subview_row<_Tp>> || std::is_same_v<_T, arma::subview_col<_Tp>>) ? arma::conv_to<arma::Col<_Tp>>::from(_toSaver) : _toSaver;
	
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
	{
		if(!_app)
			_isSaved	= _toSave.save(arma::hdf5_name(_path + _file, _db));
		else
			_isSaved	= _toSave.save(arma::hdf5_name(_path + _file, _db, arma::hdf5_opts::append));
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	else if (endsWith(_file, ".bin"))
#endif
		_isSaved	= _toSave.save(_path + _file);
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	else if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		if(!_app)
			_isSaved	= _toSave.save(_path + _file, arma::arma_ascii);
		else
		{
			std::ofstream _out;
			try
			{
				_out.open(_path + _file, std::ios::app);
				_isSaved = _out.is_open();
			}
			catch(std::exception& e)
			{
				//LOGINFO(e.what(), LOG_TYPES::ERROR, 2);
			}
			_out << _toSave;
			_out.close();
		}
	}
	return _isSaved;
}

template <HasMatrixType _T>
inline bool loadAlgebraic(const std::string& _path, const std::string& _file, _T& _toSet, const std::string& _db = "weights")
{
#ifdef _DEBUG
	//LOGINFO(LOG_TYPES::INFO, _path + _file, 3);
#endif
	createDir(_path);
	bool _isSaved = false;
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
	{
		_toSet.load(arma::hdf5_name(_path + _file, _db));
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	else if (endsWith(_file, ".bin"))
#endif
	{
		_toSet.load(_path + _file);
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	else if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		_toSet.load(_path + _file, arma::arma_ascii);
		return true;
	}
	return _isSaved;
}

template<typename _T>
	requires HasArmaVectorType<_T>
inline bool loadAlgebraic(const std::string& _path, const std::string& _file, _T& _toSet, const std::string& _db = "weights")
{
#ifdef _DEBUG
	//LOGINFO(_path + _file, LOG_TYPES::INFO, 3, '#');
#endif
	createDir(_path);
	bool _isSaved = false;
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
	{
		_toSet.load(arma::hdf5_name(_path + _file, _db));
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	else if (endsWith(_file, ".bin"))
#endif
	{
		_toSet.load(_path + _file);
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	else if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		_toSet.load(_path + _file, arma::arma_ascii);
		return true;
	}
	return _isSaved;
}



#endif