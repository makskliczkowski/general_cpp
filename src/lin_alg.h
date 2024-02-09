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

#define ARMA_WARN_LEVEL 1
#define ARMA_USE_LAPACK             
#define ARMA_PRINT_EXCEPTIONS
//#define ARMA_BLAS_LONG_LONG                                                                 // using long long inside LAPACK call
//#define ARMA_DONT_USE_FORTRAN_HIDDEN_ARGS
//#define ARMA_DONT_USE_WRAPPER
//#define ARMA_USE_SUPERLU
//#define ARMA_USE_ARPACK 
#define ARMA_USE_MKL_ALLOC
#define ARMA_USE_MKL_TYPES
//#define ARMA_DONT_USE_OPENMP
#define ARMA_USE_HDF5
////#define ARMA_USE_OPENMP
#define ARMA_ALLOW_FAKE_GCC
#define ARMA_DONT_PRINT_CXX11_WARNING
#define ARMA_DONT_PRINT_CXX03_WARNING
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include <armadillo>

#define DH5_USE_110_API
#define D_HDF5USEDLL_ 

// matrix base class concepts

#ifdef __has_include
#	if __has_include(<concepts>)
#		include <concepts>
#		include <type_traits>
		template<typename _T>
		concept HasMatrixType = std::is_base_of<arma::Mat<double>, _T>::value						|| 
								std::is_base_of<arma::Mat<std::complex<double>>, _T>::value			||
								std::is_base_of<arma::SpMat<double>, _T>::value						||
								std::is_base_of<arma::SpMat<std::complex<double>>, _T>::value;
#	endif
#else
#	pragma message ("--> Skipping concepts")
#endif


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
// #######################################################################################################################
// ##################################################### A L G E B R A ###################################################
// #######################################################################################################################
// #######################################################################################################################

namespace algebra 
{

	// ##################################################################################################################################################
	// ##################################################################################################################################################
	// ################################################################# G E N E R A L ##################################################################
	// ##################################################################################################################################################
	// ##################################################################################################################################################

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

	template <typename _T>
	inline auto cast(std::complex<double> x)										-> _T								{ return x; };
	template <>
	inline auto cast<double>(std::complex<double> x)								-> double							{ return std::real(x); };
	
	// Armadillo columns
	template <typename _T>
	inline auto cast(const arma::Col<double>& x)									-> arma::Col<_T>					{ return x; };
	template <>
	inline auto cast<std::complex<double>>(const arma::Col<double>& x)				-> arma::Col<std::complex<double>>	{ return x + std::complex<double>(0, 1) * arma::ones(x.n_rows); };
	template <typename _T>
	inline auto cast(const arma::Col<std::complex<double>>& x)						-> arma::Col<_T>					{ return x; };
	template <>
	inline auto cast<double>(const arma::Col<std::complex<double>>& x)				-> arma::Col<double>				{ return arma::real(x); };
	
	// Armadillo matrices 
	template <typename _T>
	inline auto cast(const arma::Mat<double>& x)									-> arma::Mat<_T>					{ return x; };
	template <>
	inline auto cast<std::complex<double>>(const arma::Mat<double>& x)				-> arma::Mat<std::complex<double>>	{ return x + std::complex<double>(0, 1) * arma::ones(x.n_rows, x.n_cols); };
	template <typename _T>
	inline auto cast(const arma::Mat<std::complex<double>>& x)						-> arma::Mat<_T>					{ return x; };
	template <>
	inline auto cast<double>(const arma::Mat<std::complex<double>>& x)				-> arma::Mat<double>				{ return arma::real(x); };

	// #################################################################################################################################################
	// #################################################################################################################################################
	// ############################################################# MATRIX MULTIPLICATION #############################################################
	// #################################################################################################################################################
	// #################################################################################################################################################

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

	// #################################################################################################################################################
	// #################################################################################################################################################
	// ############################################################### MATRIX PROPERTIES ###############################################################
	// #################################################################################################################################################
	// #################################################################################################################################################

	/*
	* @brief Calculate the Pfaffian of a skew square matrix A
	* !TODO
	*/
	template <typename _T>
	_T pfaffian(const arma::Mat<_T>& A, arma::u64 N)
	{
		if (N == 0)
		{
			return _T(1.0);
		}
		else if (N == 1)
		{
			return _T(0.0);
		}
		else
		{
			_T pfa = 0.0;
			for (int i = 1; i < N; ++i)
			{
				arma::Mat<_T> Atmp = A;
				// kill rows and columns
				Atmp.shed_col(i);
				Atmp.shed_row(i);
				Atmp.shed_row(0);
				// additional (from definition)
				if (N > 2)
					Atmp.shed_col(0);
				// recursively calculate 
				pfa += (((i+1) % 2) == 0 ? 1. : -1.) * A(0, i) * pfaffian(Atmp, N - 2);
			}
			return pfa;
		}
	}

	// #################################################################################################################################################
	// #################################################################################################################################################
	// ############################################################# MATRIX DECOMPOSITIONS #############################################################
	// #################################################################################################################################################
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

template <typename _T>
inline bool saveAlgebraic(const std::string& _path, const std::string& _file, const arma::Mat<_T>& _toSave, const std::string& _db = "weights")
{
#ifdef _DEBUG
	LOGINFO(_path + _file, LOG_TYPES::INFO, 3);
#endif
	createDir(_path);
	bool _isSaved	= false;
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
		_isSaved	= _toSave.save(arma::hdf5_name(_path + _file, _db));
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
		_isSaved	= _toSave.save(_path + _file, arma::arma_ascii);
	return _isSaved;
}

template <typename _T>
inline bool saveAlgebraic(const std::string& _path, const std::string& _file, const arma::Col<_T>& _toSave, const std::string& _db = "weights")
{
#ifdef _DEBUG
	LOGINFO(_path + _file, LOG_TYPES::INFO, 3);
#endif
	createDir(_path);
	bool _isSaved	= false;
#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
		_isSaved	= _toSave.save(arma::hdf5_name(_path + _file, _db));
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
		_isSaved	= _toSave.save(_path + _file, arma::arma_ascii);
	return _isSaved;
}

template <typename _T>
inline bool loadAlgebraic(const std::string& _path, const std::string& _file, arma::Mat<_T>& _toSet, const std::string& _db = "weights")
{
#ifdef _DEBUG
	LOGINFO(LOG_TYPES::INFO, _path + _file, 3);
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

template <typename _T>
inline bool loadAlgebraic(const std::string& _path, const std::string& _file, arma::Col<_T>& _toSet, const std::string& _db = "weights")
{
#ifdef _DEBUG
	LOGINFO(_path + _file, LOG_TYPES::INFO, 3, '#');
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