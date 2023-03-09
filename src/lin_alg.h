#pragma once
#ifndef ALG_H
	#define ALG_H
#endif

#define DH5_USE_110_API
#define D_HDF5USEDLL_ 
#include "../Include/statistical.h"

// #############################################################				   DEFINITIONS FROM ARMADILLO				   #############################################################

#define DIAG arma::diagmat
#define EYE(X) arma::eye(X,X)
#define ZEROV(X) arma::zeros(X)
#define ZEROS_LIKE(X) = arma::zeros(size(X));
#define ZEROM(X) arma::zeros(X,X)
#define SUBV(X, fst, lst) X.subvec(fst, lst)
#define SUBM(X, fstr, fstc, lstr, lstc) X.submat(fstr, fstc, lstr, lstc)
#define UPDATEV(L, R, condition) (condition) ? (L += R) : (L -= R)

// #############################################################				   MATRIX MULTIPLICATION				   #############################################################
/*
* @brief Allows to calculate the matrix consisting of COL vector times ROW vector
* @param setMat matrix to set the elements onto
* @param setVec column vector to set the elements from
*/
template <typename _type>
inline void setKetBra(arma::Mat<_type>& setMat, const arma::Col<_type>& setVec) {
	setMat = arma::cdot(setVec, setVec.as_row());
}

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
		UPDATEV(SUBM(m2Set, row, col, row + nrow, col + ncol), mSet, !minus);
	else
		SUBM(m2Set, row, col, row + nrow, col + ncol) = mSet;
}

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
		UPDATEV(mSet, SUBM(m2Set, row, col, row + nrow, col + ncol), !minus);
	else
		mSet = SUBM(m2Set, row, col, row + nrow, col + ncol);
}

/*
* @brief Is used to calculate the equation of the form (U_l * D_l * T_l + U_r * D_r * T_r).
* @details UDT we get from QR decomposition with column pivoting
* @param Ql
* @param Rl
* @param Pl
* @param Tl
* @param Dl
* @param Qr
* @param Rr
* @param Pr
* @param Tr
* @param Dr
* @param Dtmp
* @warning Uses the UDT decomposition from QR with column pivoting
*/
arma::mat inv_left_plus_right_qr(arma::mat& Ql, arma::mat& Rl, arma::umat& Pl, arma::mat& Tl, arma::vec& Dl, arma::mat& Qr, arma::mat& Rr, arma::umat& Pr, arma::mat& Tr, arma::vec& Dr, arma::vec& Dtmp);

// #############################################################				   MATRIX DECOMPOSITIONS				   #############################################################

template<typename _T>
class UDT{
public:
	arma::Mat<_T> U;
	arma::Col<_T> D;			// here we will put D vector - diagonal part of R
	arma::Col<_T> Di;			// here we will put D vector inverse
	arma::Mat<_T> T;

	~UDT() = default;
	UDT() = default;
	UDT(const UDT<_T>& o) : U(o.U), D(o.D), Di(o.Di), T(o.T) {};
	UDT(UDT<_T>&& o) : U(std::move(o.U)), D(std::move(o.D)), Di(std::move(o.Di), T(std::move(o.T)) {};
	UDT(const arma::Mat<_T>& u, const arma::Col<_T>& d, const arma::Mat<_T>& t) : U(u), D(d), Di(1.0/d), T(t){}
	{};
	UDT(arma::Mat<_T>&& u, arma::Col<_T>&& d, arma::Mat<_T>&& t) : U(u), D(d), Di(1.0/d), T(t)
	{};

	virtual void decompose() = 0;
	virtual void decompose(const arma::Mat<_T>& M) = 0
	
	/*
	*@brief copy assignment
	*/
	UDT<_T>& operator=(const UDT<_T>& o){
		U = o.U;
		D = o.D;
		Di = o.Di;
		T = o.T;
		return *this;
	}

	/*
	*@brief move assignment
	*/
	UDT<_T>& operator=(UDT<_T>&& o) { 
		U = std::move(o.U);
		D = std::move(o.D);
		Di = std::move(o.Di);
		T = std::move(o.T);
		return *this; 
	}
	
	// ----------------------------------- OPERATIONS

	// --- INVERSE ---

	/*
	* @brief Calculates the inverse of the UDT decomposition of a matrix. With return.
	*/
	arma::Mat<_T> inv(){
		return arma::solve(T, Di) * U.t();
	}

	/*
	* @brief Calculates the inverse of the UDT decomposition of a matrix. 
	* @param M matrix to set the inverse onto
	*/
	void inv(arma::Mat<_T>& M){
		M = arma::solve(T, Di) * U.t();
	}

	// --- MULTIPLICATION ---

	/*
	* @brief Stabilized multiplication of two `UDT` decompositions.
	*/
	void factMult(const UDT<_T>& B){
		arma::Mat<_T> mat = T * B.U; 		// Tl * Ur
		mat = D * mat;						// Rl * (*up)
		mat = mat * B.D;
		decompose(mat);
	} 

	/*
	* @brief Stabilized multiplication of two `UDT` decompositions.
	* @return UDT factorization object
	*/
	static UDT<_T> factMult(const UDT<_T>& A, const UDT<_T>& B){
		UDT<_T> ret = A;
		ret.factMult(B);
		return ret;
	}

	virtual void factMult(const arma::Mat<_T> Ml) = 0;

	// (1+A)^(-1)
	virtual arma::Mat<_T> inv1P() = 0;

	// (A+B)^(-1)
	virtual arma::Mat<_T> invSum(const UDT_QR& right) = 0;
	virtual arma::Mat<_T> invSum(const UDT_QR& right, const arma::Mat<_T> setMat) = 0;
};

/*
* @brief UDT decomposition using QR decomposition
* @cite doi:10.1016/j.laa.2010.06.023
*/
template<typename _T>
class UDT_QR : public UDT<_T>{
	// arma::Mat<_T> U = Q;		// in this case the U matrix serves as Q	
	arma::Mat<_T> R;			// right triangular matrix
	arma::umat P;				// permutation matrix
	arma::Col<_T> Db;			// Dmax
	arma::Col<_T> Ds;			// Dmin

	~UDT_QR() = default;
	UDT_QR() = default;
	UDT_QR(const arma::Mat<_T>& M){
		decompose(M);
		Db = ZEROS_LIKE(M.col(0));
		Ds = ZEROS_LIKE(M.col(0));
	};
	UDT_QR(const arma::Mat<_T>& q, const arma::Mat<_T>& r, const arma::umat& p)
		: R(r), P(p) 
	{
		U = q;
		decompose();
		Db = ZEROS_LIKE(q.col(0));
		Ds = ZEROS_LIKE(q.col(0));
	}
	UDT_QR(const UDT_QR<_T>& o): R(o.R), P(o.P), Db(o.Db), Ds(o.Ds), UDT(o) {};
	UDT_QR(UDT_QR<_T>&& o) noexcept : R(std::move(o.R)), P(std::move(o.P)), Db(std::move(o.Db)), Ds(std::move(Ds)), UDT(std::move(o)){};

	/*
	* @brief copy assignment operator
	*/
	UDT_QR<_T>& operator=(const UDT_QR<_T>& o) { 
		UDT<_T>::operator=(o);
		R = o.R;
		P = o.P;
		Db = o.Db;
		Ds = o.Ds;
		return *this;
	}

	/*
	* @brief move assignment operator
	*/
	UDT_QR<_T>& operator=(UDT_QR<_T>&& o) { 
		UDT<_T>::operator=(std::move(o));
		R = std::move(o.R);
		P = std::move(o.P);
		Db = std::move(o.Db);
		Ds = std::move(o.Ds);
		return *this;
	}

	// ----------------------------------- DECOMPOSITIONS
	
	/*
	* @brief Create a decomposition using preset matrices
	*/
	void decompose() override {
		// inverse the vector D during setting
		D = R.diag();
		Di = 1.0 / R.diag();
		T = (DIAG(Di) * R) * P.t();
	}
	
	/*
	* @brief Create a decomposition
	* @param M Matrix to decompose
	*/
	void decompose(const arma::Mat<_T>& M) override {
		if (!arma::qr(U, R, P, M)) throw "Decomposition failed\n";
		decompose();
	}

	/*
	* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
	*/
	void loh(){	
		for (auto i = 0; i < R.n_rows; i++)
		{
			if (abs(D(i)) > 1.0) {
				Db(i) = D(i);	// max (R(i,i), 1)
				Ds(i) = 1.0;	// min (R(i,i), 1)
			}
			else {
				Ds(i) = D(i);
				Db(i) = 1.0;
			}
		}
	}
	
	/*
	* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
	* @warning Saves the inverse to Db and Ds!
	*/
	void loh_inv(){
		for (auto i = 0; i < R.n_rows; i++)
		{
			if (abs(D(i)) > 1.0) {
				Db(i) = Di(i);	// max (R(i,i), 1)
				Ds(i) = 1.0;	// min (R(i,i), 1)
			}
			else {
				Db(i) = 1.0;
				Ds(i) = D(i);
			}
		}
	}

	/*
	* @brief Loh's decomposition to two scales in UDT QR decomposition. One is lower than 0 and second higher.
	* @attention ->(1, R) -> we save that in R -> therefore change is only made to set R(i,i) to 1
	* @attention ->max(1, R) -> we set that as an inverse onto D already
	* @warning Changes D and R!
	*/
	void loh_inplace(){
		for (int i = 0; i < R.n_rows; i++){
			if (abs(R(i, i)) > 1)
				R(i, i) = 1;				// min(1,R(i,i))
			else 							// (abs(R(i, i)) <= 1)
				Di(i) = 1;					// inv of max(1,R(i,i)) because Di is already an inverse
		}
	}

	// ----------------------------------- MULTIPLICATION
	// https://github.com/carstenbauer/StableDQMC.jl/blob/master/src/qr_udt.jl
	/*
	* @brief Multiply the UDT decomposition by a matrix from the left
	* @param Ml left matrix
	*/
	void factMult(const arma::Mat<_T> Ml) override {
		if (!arma::qr(U, R, P, (Ml * Q) * DIAG(D))) throw "decomposition failed\n";
		// inverse during setting
		D = R.diag();
		Di = 1.0 / D;
		// premultiply old T by new T from left
		T = ((DIAG(Di) * R) * P.t()) * T;
	}
	
	/*
	* @brief (UDT + 1)^(-1) with QR decomposition.
	*/
	arma::Mat<_T> inv1P() override {
		// decompose first
		loh_inv();
		return arma::solve(DIAG(Db) * U.t() + DIAG(Ds) * T, DIAG(Db) * U.t());
		// loh();
		// return arma::solve(arma::inv(DIAG(Db)) * U.t() + DIAG(Ds) * T, arma::inv(DIAG(Db)) * U.t());
		// without the decomposition
		// return = arma::inv(DIAG(Di) * U.t() + D * T) * DIAG(D_up) * Q_up.t();
	}
	// ########################################## (A+B)^(-1) ##########################################
	
	/*
	* @brief Stabilized calculation of [UaDaTa + UbDbTb]^{-1}
	*/
	arma::Mat<_T> invSum(UDT<_T>* right) override {
		// calculate loh decomposition
		loh();
		right.loh();
		// dimension
		auto d = Ds.size();

		// matL = D_min_a * Ta * Tb^{-1} / D_max_b
		arma::Mat<_T> matL = T * arma::inv(arma::trimatu(right->T));
		for(int i = 0; i < d; i++)
			for(int j = 0; j < d; j++)
				matL(i,j) *= Ds(i) / right->Db(j);

		// matR = 1/(D_max_a) * Ua^\dag * Ub * D_min_b
		arma::Mat<_T> matR = U.t() * right->U;
		for(int i = 0; i < d; i++)
			for(int j = 0; j < d; j++)
				matL(i,j) *= right->Ds(i) / Db(j);

		// add two matrices
		mat1 += mat2;

		// create inner decomposition
		UDT_QR<_T> inner(mat1);
		matR = arma::inv(arma::trimatu(inner.T) * DIAGMAT(inner.Di);
		matR = matR * U.t();
		for(int i = 0; i < d; i++)
			for(int j = 0; j < d; j++)
				matR(i,j) /= right->Db(i) * Db(j);
		
		// decompose again
		inner.decompose(matR);
		inner.U = arma::inv(arma::trimatu(right->T)) * inner.U;
		inner.T = inner.T * U.t();

	}

	arma::Mat<_T> invSum(const UDT_QR& right, const arma::Mat<_T> setMat){


	}

};

// #############################################################				   MATRIX MULTIPLICATION				   #############################################################


/*
* @brief Using ASvQRD - Accurate Solution via QRD with column pivoting to multiply the QR on the right and multiply new matrix mat_to_multiply on the left side.
* @cite doi:10.1016/j.laa.2010.06.023
* @param mat_to_multiply (left) matrix to multiply by the QR decomposed stuff (on the right)
* @param Q unitary Q matrix
* @param R right triangular matrix
* @param P permutation matrix
* @param T upper triangular matrix
* @param D inverse of the new R diagonal
*/
void inline multiplyMatricesQrFromRight(const arma::mat& mat_to_multiply, arma::mat& Q, arma::mat& R, arma::umat& P, arma::mat& T, arma::vec& D) {

}

void inline multiplyMatricesSVDFromRight(const arma::mat& mat_to_multiply, arma::mat& U, arma::vec& s, arma::mat& V, arma::mat& tmpV) {
	svd(U, s, tmpV, mat_to_multiply * U * DIAG(s));
	V = V * tmpV;
}
