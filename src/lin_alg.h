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
#pragma once

#ifndef ALG_H
#define ALG_H

using uint = unsigned int;
// ################################################ INCLUDE FROM ARMADILLO ###############################################

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
// #######################################################################################################################
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

	namespace MatMul {
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
	};

	using namespace MatMul;
	
	// *************************************************************************************************************************************************

	// ########################################################### MATRIX EQUATIONS SOLVERS ############################################################

	// #################################################################################################################################################

	namespace Solvers
	{	
		constexpr double TINY = 1.0e-16;		// a small number

		// #################################################################################################################################################

		template <typename _T1>
		void sym_ortho(_T1 a, _T1 b, _T1& c, _T1& s, _T1& r);

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
				int type_ 							= 0;						// type of the preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
			public:
				virtual ~Preconditioner() = default;
				Preconditioner() 
					: isGram_(false)
				{};
				Preconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: isGram_(isGram)
				{
					// this->set(A, isGram, _sigma);
				}
				Preconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: isGram_(true), sigma_(_sigma)
				{
					// this->set(Sp, S, _sigma);
				}
				// -----------------------------------------------------------------------------------------------------------------------------------------

				// set the preconditioner
				void set(bool _isGram, double _sigma = 0.0) { this->isGram_ = _isGram; this->sigma_ = _sigma; }
				virtual void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) = 0;		// set the preconditioner
				virtual void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) = 0;	// set the preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				int type() const { return this->type_; }													// get the type of the preconditioner

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
				int type_ 			= 1;		// type of the preconditioner
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
						if (_sigma > 0.0)
							diag += (T)this->sigma_;
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
				int type_ 		= 2;	// type of the preconditioner
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

			/*
			* @brief Binormalization preconditioner for the conjugate gradient method. This preconditioner is used for symmetric positive definite matrices.
			* Scale the matrix with a series of k diagonal matrices D1, D2, ..., Dk -> DAD = D_k ... D_2 D_1 A D_1 D_2 ... D_k
			*/
			template <typename T, bool _T = true>
			class BinormalizationPreconditioner : public Preconditioner<T, _T> {
			private:
				bool success_ 	= false;
				int type_ 		= 3;	// type of the preconditioner

			public:
				BinormalizationPreconditioner()
					: Preconditioner<T, _T>()
				{};			
				/**
				* @brief Constructor to initialize the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				*/
				BinormalizationPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{}

				BinormalizationPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
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
					// !TODO
				}

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param Sp The matrix to decompose.
				* @param S The matrix to decompose.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the
				* diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					// !TODO
				}
			};

			// #################################################################################################################################################

			template <typename T, bool _T = true>
			class IncompleteLUPreconditioner : public Preconditioner<T, _T> {
			private:
				arma::Mat<T> L_;     // lower triangular incomplete Cholesky factor
				arma::Mat<T> U_;     // upper triangular incomplete Cholesky factor
				arma::Col<arma::uword> P_; // permutation vector
				bool success_ 	= false;
				int type_ 		= 3;	// type of the preconditioner
			public:
				IncompleteLUPreconditioner()
					: Preconditioner<T, _T>()
				{};
				/**
				* @brief Constructor to initialize the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				*/
				IncompleteLUPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{}

				IncompleteLUPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
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

					// !TODO
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

					// !TODO
				}

				/**
				* @brief Apply the preconditioner to a given vector.
				* @param r The vector to precondition.
				* @param sigma Regularization parameter (default is 0.0).
				* @return The preconditioned vector.
				*/
				arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const override
				{
					// !TODO
					return r;
				}
			};
			

			// #################################################################################################################################################
			namespace Symmetric {
				enum class PreconditionerType {
					Identity,
					Jacobi,
					IncompleteCholesky,
					IncompleteLU
				}; 
			};
			namespace NonSymmetric {
				enum class PreconditionerType {
					Identity
				}; 
			};

			// -----------------------------------------------------------------------------------------------------------------------------------------
		

			// -----------------------------------------------------------------------------------------------------------------------------------------

			template <typename T, bool _symmetric = true>
			inline Preconditioner<T, _symmetric>* choose(Symmetric::PreconditionerType i) {
				switch (i) {
				case Symmetric::PreconditionerType::Identity:
					return new IdentityPreconditioner<T, _symmetric>;
				case Symmetric::PreconditionerType::Jacobi:
					return new JacobiPreconditioner<T, _symmetric>;
				case Symmetric::PreconditionerType::IncompleteCholesky:
					return new IncompleteCholeskyPreconditioner<T, _symmetric>;
				case Symmetric::PreconditionerType::IncompleteLU:
					return new IncompleteLUPreconditioner<T, _symmetric>;
				default:
					return new IdentityPreconditioner<T, _symmetric>;
				};
			}

			template <typename T, bool _Sym = true>
			inline Preconditioner<T, _Sym>* choose(int i = 0) { return choose<T, _Sym>(static_cast<Symmetric::PreconditionerType>(i)); };
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

			inline std::string name(Symmetric::PreconditionerType i) {
				switch (i) {
				case Symmetric::PreconditionerType::Identity:
					return "Identity";
				case Symmetric::PreconditionerType::Jacobi:
					return "Jacobi";
				case Symmetric::PreconditionerType::IncompleteCholesky:
					return "Incomplete Cholesky";
				case Symmetric::PreconditionerType::IncompleteLU:
					return "Incomplete LU";
				default:
					return "Identity";
				};
			}

			inline std::string name(int i = 0) { return name(static_cast<Symmetric::PreconditionerType>(i)); };
			
			// -----------------------------------------------------------------------------------------------------------------------------------------
			
		};

		// #################################################################################################################################################
		template<typename _T>
		using _AX_fun = std::function<arma::Col<_T>(const arma::Col<_T>&, double)>;	// matrix-vector multiplication function
		template<typename _T, bool _T1 = true>
		using Precond = Preconditioners::Preconditioner<_T, _T1>;					// preconditioner type
		// #################################################################################################################################################
		#define _MATFREE_MULT(_T) _AX_fun(_T) _matFreeMul							// matrix-free multiplication function
		#define SOLVE_GENERAL_ARG_TYPES(_T1) 		const arma::Col<_T1>& _F,									\
													arma::Col<_T1>* _x0,										\
													double _eps,												\
													size_t _max_iter,											\
													bool* _converged, 											\
													double _reg												
		#define SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)const arma::Col<_T1>& _F,						\
																arma::Col<_T1>* _x0,							\
																Solvers::Precond<_T1, _T2>* _preconditioner,	\
																double _eps,									\
																size_t _max_iter,								\
																bool* _converged, 								\
																double _reg			
		// with default values			
		#define SOLVE_GENERAL_ARG_TYPESD(_T1) 		const arma::Col<_T1>& _F,									\
													arma::Col<_T1>* _x0 	= nullptr,							\
													double _eps				= 1e-10,							\
													size_t _max_iter		= 100,								\
													bool* _converged		= nullptr, 							\
													double _reg				= -1.0				
		#define SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2) const arma::Col<_T1>& _F,						\
																arma::Col<_T1>* _x0,							\
																Solvers::Precond<_T1, _T2>* _preconditioner = nullptr, \
																double _eps							= 1e-10,	\
																size_t _max_iter					= 100,		\
																bool* _converged					= nullptr,	\
																double _reg							= -1.0
		// with matrix multiplication function
		#define SOLVE_MATMUL_ARG_TYPES(_T1) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPES(_T1)
		#define SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, _T2) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)
		#define SOLVE_MATMUL_ARG_TYPESD(_T1) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPESD(_T1)
		#define SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, _T2) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2)

		// #################################################################################################################################################
		namespace General {
			// #################################################################################################################################################
			enum class Type {
				ARMA				= 0,				// Armadillo solver
				PseudoInverse		= 4,				// Pseudo Inverse - minimum norm solution
				Direct				= 5,				// Direct solver - may not be s
				// SYMMETRIC
				ConjugateGradient	= 1,				// Conjugate Gradient Method
				MINRES				= 2,				// Minimum Residual Method
				MINRES_QLP			= 3					// Minimum Residual Method with QLP
			};
			// #############################################################################################################################################
						
			template <typename _T, bool _symmetric = true>
			class Solver 
			{
			protected:
				Type type_ 				= Type::Direct;	// type of the solver
				bool isSymmetric_ 		= _symmetric;	// is the matrix symmetric
				bool converged_ 		= false;		// has the method converged
				bool isGram_			= false;		// is the matrix a Gram matrix
				size_t N_				= 1;			// size of the matrix
				size_t iter_ 			= 0; 			// current iteration - [[maybe_unused]]
				size_t max_iter_ 		= 1000;			// maximum number of iterations
				double eps_ 			= 1e-10;		// convergence criterion
				double reg_ 			= -1.0;			// regularization parameter (if needed)
				Precond<_T, _symmetric>* precond_;		// preconditioner (if exists) - this is used to solve the system M^{-1}Ax = M^{-1}b
				bool isPreconditioned_ 	= false;		// is the matrix preconditioned (reffers to the preconditioner_ field)
				_AX_fun<_T> matVecFun_;					// matrix-vector multiplication function such that Ax = b (if exists)
				arma::Col<_T> x_;						// solution vector

			public:
				// -----------------------------------------------------------------------------------------------------------------------------------------
				virtual ~Solver()		= default;
				Solver() 				= default;
				Solver(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T, _symmetric>* _preconditioner = nullptr);

				virtual void init(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(_AX_fun<_T> _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr) = 0;

				// -----------------------------------------------------------------------------------------------------------------------------------------
				// getters
				inline const arma::Col<_T>& solution() 					const { return this->x_; };			
				inline _T solution(size_t _i) 							const { return this->x_(_i); }
				inline size_t getN() 									const { return this->N_; }
				inline size_t getIter() 								const { return this->iter_; }
				inline size_t getMaxIter() 								const { return this->max_iter_; }
				inline double getEps() 									const { return this->eps_; }
				inline double getReg() 									const { return this->reg_; }
				inline bool isConverged() 								const { return this->converged_; }
				inline bool isPreconditioned() 							const { return this->isPreconditioned_; }
				inline Precond<_T, _symmetric>* getPreconditioner() 	const { return this->precond_; }
				// -----------------------------------------------------------------------------------------------------------------------------------------
				// setters
				inline void setMaxIter(size_t _max_iter) 				{ this->max_iter_ = _max_iter; }
				inline void setEps(double _eps) 						{ this->eps_ = _eps; }
				inline void setReg(double _reg) 						{ this->reg_ = _reg; }
				inline void setPreconditioner(Precond<_T, _symmetric>* _precond) { this->precond_ = _precond; isPreconditioned_ = (_precond != nullptr); }
				// -----------------------------------------------------------------------------------------------------------------------------------------

				virtual void solve(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);							// if we want to use a dense matrix
				virtual void solve(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);							// if we want to use a sparse matrix
				virtual void solve(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);	// if we want to use a Fisher matrix
				virtual void solve(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);	// if we want to use a Fisher matrix
				virtual void solve(_AX_fun<_T> _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);										// if we want to use a matrix-vector multiplication function
				virtual void solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr) = 0;													// if the matrix multiplication function is set
			};	
			// #############################################################################################################################################
		};
		// #################################################################################################################################################
		namespace General { 
			// #################################################################################################################################################
			// with a signle matrix
			#define SOLVE_MAT_ARG_TYPES(_T1) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES(_T1)
			#define SOLVE_MAT_ARG_TYPES_PRECONDITIONER(_T1, _T2) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)
			#define SOLVE_MAT_ARG_TYPESD(_T1) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD(_T1)
			#define SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, _T2) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2)
			// with sparse matrix
			#define SOLVE_SPMAT_ARG_TYPES(_T1) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES(_T1)
			#define SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(_T1, _T2) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)
			#define SOLVE_SPMAT_ARG_TYPESD(_T1) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD(_T1)
			#define SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, _T2) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2)
			// #################################################################################################################################################	
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _A, const arma::Col<_T>& _x, const double _reg = 0.0);
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::SpMat<_T>& _A, const arma::Col<_T>& _x, const double _reg = 0.0);
			// #################################################################################################################################################
			#define MAKE_MATRIX_FREE_MULT(_T) auto _f = [&](const arma::Col<_T>& _x, double _reg) -> arma::Col<_T> { return matrixFreeMultiplication<_T>(_A, _x, _reg); };
			// #################################################################################################################################################
			namespace CG {
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MATMUL_ARG_TYPESD(_T1));
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, true));
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_SPMAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg); } 
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
			
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class ConjugateGradient_s : virtual public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Col<_T1> r, p, Ap;
					// for preconditioned only
					arma::Col<_T1> z;
					_T1 rs_old;
				public:
					ConjugateGradient_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::ConjugateGradient;
						if (!_symmetric) 
							throw std::invalid_argument("Conjugate Gradient method is only for symmetric matrices.");
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES {
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPESD(_T1));
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, true));
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_SPMAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				// -------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class MINRES_s : virtual public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Col<_T1> r, pkm1, pk, pkp1, Ap_km1, Ap_k, Ap_kp1;

					_T1 beta0_;

				public:
					MINRES_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::MINRES;
						if(!_symmetric) 
							throw std::invalid_argument("MINRES method is only for symmetric matrices.");
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
			
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			
			/**
			* % MINRES-QLP: Minimum Residual QLP Method - minimal leng solution to symmetric (possibly singular) Ax = b or min ||Ax - b||
			* ---------
			* !TODO 
			*	- Implement the MINRES-QLP method for general symmetric matrices A (possibly singular)
			*	- Implement the MINRES-QLP method for symmetric positive definite matrices A (not singular)
			*  - Implement convergence criterion return rather than finished iterations
			* !CURRENTLY
			* 	- The method is implemented for Fisher matrices, which are symmetric positive definite matrices constructed as S = \Delta O^* \Delta O
			* 	X = minres_qlp(deltaO, deltaO^+, F, x0, eps, max_iter, converged, reg) solves the system Sx = F or the minimization problem ||Sx - F||_2
			*  	The N_samples x N_params matrix deltaO is the derivative of the observable with respect to the parameters (rows are samples, columns are parameters)
			*  	The N_params x N_params matrix deltaO^+ is the conjugate transpose of deltaO (rows are parameters, columns are samples)
			*      The method allows for specification of the initial guess x0, the convergence criterion eps, the maximum number of iterations max_iter, the regularization parameter reg 
			*      such that the system to solve is (S + reg*I)x = F or the minimization problem ||(S + reg*I)x - F||_2
			* @ see MINRES_QLP::minres_qlp in upper part of this namespace - inside other namespaces.
			* 		Additionally, in the method MAXXNORM and ACONDLIM parameters are specified on Norm of X and Condition number of A, respectively.
			* @note The method shall be possible to solve the complex and real systems.
			* @note in minres_qlp one can also specify the preconditioner for the system to solve such that the system to solve is M^{-1}Sx = M^{-1}F or the minimization problem ||M^{-1}Sx - M^{-1}F||_2
			* !CONVERGENCE CRITERION:
			* 		- -1 	(beta_k = 0) 		F and X are eigenvectors of (A - sigma*I) 
			* 		- 0 	(beta_km1  = 0) 	F = 0, X = 0
			* 		- 1     X solves the system to the required tolerance RELRES = RNORM / (ANORM * XNORM + BNORM) <= RTOL, where R = B - (A - sigma*I)X and RNORM = ||R||_2
			* 		- 2     X solves the system to the required tolerance RELRES = ARNORM / (ANORM * XNORM) <= RTOL,  where AR = (A - sigma*I)R and ARNORM = NORM(AR).
			*      	- 3 	same as 1, but with RTOL = EPS
			*      	- 4 	same as 2, but with RTOL = EPS
			*      	- 5 	X converged to eigenvector of (A - sigma*I) 
			*      	- 6     XNORM exceeded MAXXNORM
			*      	- 7     ACOND exceeded ACONDLIM
			*      	- 8 	MAXITER reached
			* 		- 9 	The sytem appears to be singular or badly scaled
			* @ref Sou-Cheng T. Choi and Michael A. Saunders, ALGORITHM: MINRES-QLP for Singular Symmetric and Hermitian Linear Equations and Least-Squares Problems, to appear in ACM Transactions on Mathematical Software.
			* @credit The code was based on the published algorithm and the MATLAB implementation by Sou-Cheng: https://www.mathworks.com/matlabcentral/fileexchange/42419-minres-qlp and translated to C++ 
			* with some modifications and related changes.
			// ---------
			*/
			namespace MINRES_QLP {
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, true));
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MATMUL_ARG_TYPESD(_T1));
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_SPMAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class MINRES_QLP_s : public Solver<_T1, _symmetric> 
				{
				protected:
					bool rnormvec_ = false;
					std::vector<_T1> resvec_, Aresvec_;			// residuals
					// Lanczos vectors and scalars
					arma::Col<_T1> z_km2, z_km1, z_k;
					_T1 _beta1, _beta_km1, _beta_k, _phi_k;
					arma::Col<_T1> v;
					// !!!!!!!!! Previous left reflection 
					_T1 _delta_k;
					_T1 _c_k_1, _c_k_2, _c_k_3; 							// is cs in the algorithm, cr2, cr1 - cosines
					_T1 _s_k_1, _s_k_2, _s_k_3; 							// is sn in the algorithm
					_T1 _gamma_k, _gamma_km1, _gamma_km2, _gamma_km3;
					_T1 _gamma_min, _gamma_min_km1, _gamma_min_km2; 		// is gamma, gammal, gammal2, gammal3 
					_T1 _tau_k, _tau_km1, _tau_km2;							// use them as previous values of tau's - is tau, taul, taul2 in the algorithm
					_T1 _eps_k, _eps_k_p1;								
					_T1 _Ax_norm_k;												
					// !!!!!!!!!! Previous right reflection
					_T1 _theta_k, _theta_km1, _theta_km2;					// use them as previous values of theta's, is theta, thetal, thetal2 in the algorithm
					_T1 _eta_k, _eta_km1, _eta_km2;	
					// !!!!!!!!!! 
					_T1 _xnorm_k;											// is xi in the algorithm - norm of the solution vector, is also xnorm, xnorml
					_T1 _xl2norm_k;											// is xil in the algorithm : xl2norm
					_T1 _mu_k, _mu_km1, _mu_km2, _mu_km3, _mu_km4;			// use them as previous values of mu'
					_T1 _relres_km1, _relAres_km1;							// use them as previous values of relative residuals
					_T1 _rnorm, _rnorm_km1, _rnorm_km2;						// use them as previous values of rnorm's
					_T1 _relres, _relAres;									// relative residual with a safety margin for beta_k = 0
					// !!!!!!!!! Regarding the wektor w and the solution vector x
					arma::Col<_T1> _w_k, _w_km1, _w_km2;
					arma::Col<_T1> x_km1;
					_T1 _Anorm, _Anorm_km1; 
					_T1 _Acond, _Acond_km1;									// use them as previous values of A's norm and condition number
					// !!!!!!!!! QLP part
					_T1 _gammaqlp_k, _gammaqlp_km1;
					_T1 _thetaqlp_k;
					_T1 _muqlp_k, _muqlp_km1;
					_T1 _root_km1;
					int _QLP_iter;											// number of QLP iterations
					// !!!!!!!!!
					int flag_ = -2;											// flag for convergence

				public:
					MINRES_QLP_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::MINRES_QLP;
						if(!_symmetric) 
							throw std::invalid_argument("MINRES_QLP_s: The MINRES_QLP method is only for symmetric matrices.");
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace PseudoInverse {
				template<typename _T1, bool _symmetric = true>
				class PseudoInverse_s : public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Mat<_T1> Amat_;
				public:
					PseudoInverse_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::PseudoInverse;
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace Direct {
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class Direct_s : public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Mat<_T1> Amat_;
				public:
					Direct_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::Direct;
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace ARMA {
				template<typename _T1>
				arma::Col<_T1> arma_solve(SOLVE_MAT_ARG_TYPESD(_T1)) { return arma::solve(_A, _F); }
				template<typename _T1>
				arma::Col<_T1> arma_solve(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { return arma::solve(_A, _F); }
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class ARMA_s : public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Mat<_T1> Amat_;
				public:
					ARMA_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::ARMA;
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			
			// with matrix multiplication function and preconditioner
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));

			// with Matrix A and preconditioner
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			
			// with sparse matrix and preconditioner
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));

			// with matrix multiplication function
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MATMUL_ARG_TYPESD(_T1));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MATMUL_ARG_TYPESD(_T1));

			// with Matrix A
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPESD(_T1));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MAT_ARG_TYPESD(_T1));

			// with sparse matrix
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPESD(_T1));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_SPMAT_ARG_TYPESD(_T1));
			
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			template <typename _T1, bool _symmetric = true>
			Solver<_T1, _symmetric>* choose(Solvers::General::Type _type, size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr);
			template <typename _T1, bool _symmetric = true>
			Solver<_T1, _symmetric>* choose(int _type, size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr);
			// -----------------------------------------------------------------------------------------------------------------------------------------
			std::string name(Solvers::General::Type _type);
			std::string name(int _type);

			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace Tests
			{
				template <typename _T1, bool _symmetric = true>
				std::pair<arma::Mat<_T1>, arma::Col<_T1>> solve_test_mat_vec(bool _makeRandom = true);
				
				template <typename _T1, bool _symmetric = true>
				void solve_test(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _b, Solvers::General::Type _type, double _eps, int _max_iter, double _reg, int _preconditionertype = -1);
				template <typename _T1, bool _symmetric = true>
				void solve_test(Solvers::General::Type _type, double _eps, int _max_iter, double _reg, int _preconditionertype = -1, bool _makeRandom = false);
				
				template <typename _T1, bool _symmetric = true>
				void solve_test_multiple(const arma::Mat<_T1>& A_true, const arma::Col<_T1>& b_true, double _eps, int _max_iter, double _reg, int _preconditionertype = -1);
				template <typename _T1, bool _symmetric = true>
				void solve_test_multiple(double _eps, int _max_iter, double _reg, int _preconditionertype = -1, bool _makeRandom = false);
			};
		};
		// #################################################################################################################################################
		
		#define SOLVE_FISHER_MATRIX(_T1) 						const arma::Mat<_T1>& _DeltaO
		#define SOLVE_FISHER_MATRICES(_T1) 						const arma::Mat<_T1>& _DeltaO, const arma::Mat<_T1>& _DeltaOConjT
		#define SOLVE_FISHER_ARG_TYPES(_T1) 					SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPES(_T1)
		#define SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(_T1) 		SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, true)
		// with default values
		#define SOLVE_FISHER_ARG_TYPESD(_T1) 					SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPESD(_T1)
		#define SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1) 	SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, true)
		#define MAKE_MATRIX_FREE_MULT_FISHER(_T) auto _f = [&](const arma::Col<_T>& _x, double _reg) -> arma::Col<_T> { return FisherMatrix::matrixFreeMultiplication<_T>(_DeltaO, _DeltaOConjT, _x, _reg); };
		namespace FisherMatrix 
		{	
			/*
			* This methods are used whenever the matrix can be 
			* decomposed into the form S = \Delta O^* \Delta O, where \Delta O is 
			* the derivative of the observable with respect to the parameters. 
			* The matrix S is symmetric and positive definite, so the conjugate gradient method can be used.
			* @equation S_{ij} = <\Delta O^*_i \Delta O_j> / N 
			*/


			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _DeltaO, const arma::Col<_T>& _x, const double _reg = 0.0);

			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _DeltaO, const arma::Mat<_T>& _DeltaOConjT, const arma::Col<_T>& x, const double _reg = 0.0);
			
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::SpMat<_T>& _DeltaO, const arma::SpMat<_T>& _DeltaOConjT, const arma::Col<_T>& x, const double _reg = 0.0);
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

			// Conjugate gradient solver for the Fisher matrix inversion
			namespace CG 
			{
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_FISHER_ARG_TYPESD(_T1)) 
				{ 
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::CG::conjugate_gradient<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg); 
				}
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1)) 
				{ 
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::CG::conjugate_gradient<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); 
				}
			};
			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES
			{
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_FISHER_ARG_TYPESD(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES::minres<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES::minres<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}
			}
			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES_QLP 
			{	
				template <typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_FISHER_ARG_TYPESD(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES_QLP::minres_qlp<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
				template <typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES_QLP::minres_qlp<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}
			};

			// #################################################################################################################################################
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
			}

			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
			}

			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_FISHER_ARG_TYPESD(_T1))
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _eps, _max_iter, _converged, _reg);
			}
			
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_FISHER_ARG_TYPESD(_T1)) 
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _eps, _max_iter, _converged, _reg);
			}

			// #################################################################################################################################################
		};
		
		// ################################################################### ARNOLDI #####################################################################

		/**
		* @brief Arnoldi method for solving eigenvalue problems. Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.  If
        * the Krylov subspace becomes A-invariant then V and H are truncated such
        * that :math:`AV_n = V_n H_n`.

		*/
		template <typename _T, bool _symmetric = true, bool _reorthogonalize = false>
		class Arnoldi : public General::Solver<_T, _symmetric>
		{
		protected:
			bool reorthogonalize_ 		= _reorthogonalize;		// reorthogonalize the vectors
			bool isGram_ 				= false;				// is the matrix a Gram matrix 
			bool invariant_ 			= false;				// is the Krylov subspace A-invariant
			size_t krylovDim_ 			= 0;					// dimension of the Krylov subspace
																// as V_n = M * P_n, where M is the preconditioner and P_n is the original basis
			arma::Mat<_T> V_;									// basis (reorthogonalized or not)
			arma::Mat<_T> P_; 									// basis (preconditioned)
			arma::Mat<_T> H_;									// Hessenberg matrix - or Lanczos matrix (if symmetric)
			arma::Col<_T> p_;									// preconditioned vector - maybe unnecessary
			arma::Col<_T> v_;									// original vector
			arma::Col<_T> Av_; 									// A * v
			arma::Col<_T> MAv_; 								// M * A * v
			double vnorm_ 				= 0.0;					// norm of the original vector

		public:
			void init(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr) override final;
			// -----------------------------------------------------------------------------------------------------------------------------------------
			~Arnoldi() 					{};
			Arnoldi() 					= default;
			Arnoldi(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T, _symmetric>* _preconditioner = nullptr);

			// -----------------------------------------------------------------------------------------------------------------------------------------
		
			// single Arnoldi iteration
			void advance();
			// full Arnoldi iteration
			void iterate();

			// -----------------------------------------------------------------------------------------------------------------------------------------
			// getters
			inline const arma::Mat<_T>& getV() 					const { return this->V_; }
			inline const arma::Mat<_T>& getP() 					const { return this->P_; }
			inline const arma::Mat<_T>& getH() 					const { return this->H_; }
			inline const arma::Col<_T>& getAv() 				const { return this->Av_; }
			inline const arma::Col<_T>& getMAv() 				const { return this->MAv_; }
			inline const arma::subview_col<_T> getV(size_t _i) 	const { return this->V_.col(_i); }
			inline const arma::subview_col<_T> getP(size_t _i) 	const { return this->P_.col(_i); }
			inline const arma::subview_col<_T> getH(size_t _i) 	const { return this->H_.col(_i); }
			
			// -----------------------------------------------------------------------------------------------------------------------------------------
			void solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr) override final;
			// -----------------------------------------------------------------------------------------------------------------------------------------
		};

		// #################################################################################################################################################
	
	};


	// #################################################################################################################################################
	
	// ################################################################### PFAFFIANS ###################################################################

	// #################################################################################################################################################
	namespace Pfaffian
	{
		enum class PfaffianAlgorithms {
			ParlettReid,
			Householder,
			Schur,
			Hessenberg,
			Recursive
		};
		// #################################################################################################################################################
		template <typename _T>
		_T pfaffian_r(const arma::Mat<_T>& A, arma::u64 N);
		template <typename _T>
		_T pfaffian_hess(const arma::Mat<_T>& A, arma::u64 N);
		template <typename _T>
		_T pfaffian_p(arma::Mat<_T> A, arma::u64 N);
		template <typename _T>
		_T pfaffian_s(arma::Mat<_T> A, arma::u64 N);
		template <typename _T>
		_T pfaffian(const arma::Mat<_T>& A, arma::u64 N, PfaffianAlgorithms _alg = PfaffianAlgorithms::ParlettReid);
		
		/*
		* @brief Update the Pfaffian of a skew-symmetric matrix after a row and column update.
		* @param _pffA current Pfaffian
		* @param _Ainv inverse of the original matrix
		* @returns the Pfaffian of an updated skew-symmetric matrix
		*/
		template <typename _Tp, typename _T>
		inline _Tp pfaffian_upd_row_n_col(_Tp _pffA, const _T& _Ainv_row, const _T& _updRow)
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