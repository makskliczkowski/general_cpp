#pragma once
/***************************************
* Defines the Hamiltonian Matrix override
* for sparse and dense matrices.
* APRIL 2024. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

//#ifndef SYMMETRIES_H
//#	include "algebra/operators.h"
//#endif // !SYMMETRIES_H

//#ifndef SYSTEM_PROPERTIES_H
//#	include "quantities/statistics.h"
//#endif // !SYSTEM_PROPERTIES_H

#include "../../lin_alg.h"

// ############################################################################################################

/*
* @brief This class will allow to create Hamiltonian matrices for different models within the sparse and dense representation.
* @tparam _T: Type of the matrix elements.
*/
template <typename _T>
class GeneralizedMatrix
{
public:
	size_t n_rows	= 0;
	size_t n_cols	= 0;

	// sparsity flag
	bool isSparse_	= true;
protected:
	// number of elements
	u64 Nh_			= 1;

	// matrices placeholders
	arma::SpMat<_T> H_sparse_;
	arma::Mat<_T> H_dense_;

public:

	// ##################################

	// Constructors

	// Destructor
	~GeneralizedMatrix()
	{
		DESTRUCTOR_CALL;
	}

	// Default constructor
	GeneralizedMatrix() = default;

	// Constructor for distinguishing between sparse and dense matrices
	GeneralizedMatrix(u64 _Nh, bool _isSparse = true)
		: n_rows(_Nh), n_cols(_Nh), isSparse_(_isSparse), Nh_(_Nh)
	{
		if (isSparse_)
			H_sparse_ = arma::SpMat<_T>(Nh_, Nh_);
		else
			H_dense_ = arma::Mat<_T>(Nh_, Nh_, arma::fill::zeros);
	}

	// Constructor for distinguishing between sparse and dense matrices - with different number of rows and columns
	GeneralizedMatrix(u64 _nrows, u64 _ncols, bool _isSparse = true)
		: n_rows(_nrows), n_cols(_ncols), isSparse_(_isSparse), Nh_(_nrows)
	{
		if (isSparse_)
			H_sparse_ = arma::SpMat<_T>(_nrows, _ncols);
		else
			H_dense_ = arma::Mat<_T>(_nrows, _ncols, arma::fill::zeros);
	}

	// Constructor for dense matrices
	GeneralizedMatrix(const arma::Mat<_T>& _H)
		: isSparse_(false), Nh_(_H.n_rows), H_dense_(_H)
	{
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		CONSTRUCTOR_CALL;
	}

	// Constructor for sparse matrices
	GeneralizedMatrix(const arma::SpMat<_T>& _H)
		: isSparse_(true), Nh_(_H.n_rows), H_sparse_(_H)
	{
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		CONSTRUCTOR_CALL;
	}

	// copy constructor
	GeneralizedMatrix(const GeneralizedMatrix& _H)
	{
		this->isSparse_ = _H.isSparse_;
		this->Nh_		= _H.Nh_;
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;

		// copy the matrix
		if (this->isSparse_)
		{
			this->H_dense_.clear();
			this->H_sparse_ = _H.H_sparse_;
		}
		else
		{
			this->H_sparse_.clear();
			this->H_dense_	= _H.H_dense_;
		}
	}

	// move constructor
	GeneralizedMatrix(GeneralizedMatrix&& _H) noexcept
	{
		this->isSparse_ = _H.isSparse_;
		this->Nh_		= _H.Nh_;
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;

		// move the matrix
		if (this->isSparse_)
		{
			this->H_dense_.clear();
			this->H_sparse_ = std::move(_H.H_sparse_);
		}
		else
		{
			this->H_sparse_.clear();
			this->H_dense_ = std::move(_H.H_dense_);
		}
	}

	// ##################################

	// Access operator
	_T operator()(u64 row, u64 col) 
	{
		if(this->isSparse_)
			return this->H_sparse_(row, col);
		return this->H_dense_(row, col);
	}

	const _T& operator()(u64 row, u64 col) const
	{
		if (this->isSparse_)
			return this->H_sparse_(row, col);
		return this->H_dense_(row, col);
	}

	_T get(size_t _row, size_t _col) const 
	{
		if (this->isSparse_)
			return this->H_sparse_(_row, _col);
		return this->H_dense_(_row, _col);
	}

	// ##################################

	// Getters
	auto t()				const -> GeneralizedMatrix<_T>		{ return this->isSparse_ ? GeneralizedMatrix<_T>(this->H_sparse_.t()) : GeneralizedMatrix<_T>(this->H_dense_.t()); }
	auto st()				const -> GeneralizedMatrix<_T>		{ return this->isSparse_ ? GeneralizedMatrix<_T>(this->H_sparse_.st()) : GeneralizedMatrix<_T>(this->H_dense_.st()); }
	auto meanLevelSpacing()	const -> double;
	auto diag()				const -> arma::Col<_T>;
	auto diag(size_t _k)	const -> arma::Col<_T>;
	auto diagD()			-> arma::diagview<_T>				{ return this->H_dense_.diag();				}
	auto diagD(size_t k)	-> arma::diagview<_T>				{ return this->H_dense_.diag(k);			}
	auto diagSp()			-> arma::spdiagview<_T>				{ return this->H_sparse_.diag();			}
	auto diagSp(size_t k)	-> arma::spdiagview<_T>				{ return this->H_sparse_.diag(k);			}
	auto getNh()			const -> u64						{ return this->Nh_;							}
	auto isSparse()			const -> bool						{ return this->isSparse_;					}
	auto getSparse()		-> arma::SpMat<_T>&					{ return this->H_sparse_;					}
	auto getSparse()		const -> const arma::SpMat<_T>&		{ return this->H_sparse_;					}
	auto getDense()			-> arma::Mat<_T>&					{ return this->H_dense_;					}
	auto getDense()			const -> const arma::Mat<_T>&		{ return this->H_dense_;					}
	auto size()				const -> u64						{ return this->Nh_;							}
	// Method to convert to dense matrix
	auto toDense()			const -> arma::Mat<_T>				{ return arma::Mat<_T>(H_sparse_);			}

	// Method to convert to sparse matrix
	auto toSparse()			const -> arma::SpMat<_T>			{ return arma::SpMat<_T>(H_dense_);			}
	auto symmetrize()		-> void;
	auto standarize()		-> void;

	// Setters
	void setSparse(const arma::SpMat<_T>& _H)					{ this->isSparse_ = true; this->H_dense_.clear(); this->H_sparse_ = _H;		}
	void setDense(const arma::Mat<_T>& _H)						{ this->isSparse_ = false; this->H_sparse_.clear(); this->H_dense_ = _H;	}

	// ##################################
	
	template<typename _T2>
	void set(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) = _val;
		else
			this->H_dense_(_row, _col) = _val;
	}
	template<typename _T2>
	void add(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) += _val;
		else
			this->H_dense_(_row, _col) += _val;
	}
	template<typename _T2>
	void sub(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) -= _val;
		else
			this->H_dense_(_row, _col) -= _val;
	}
	template<typename _T2>
	void mul(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) *= _val;
		else
			this->H_dense_(_row, _col) *= _val;
	}
	template<typename _T2>
	void div(u64 _row, u64 _col, _T2 _val)
	{
		if (this->isSparse_)
			this->H_sparse_(_row, _col) /= _val;
		else
			this->H_dense_(_row, _col) /= _val;
	}

	// ##################################

	// Overloaded operators
	
	// Copy assignment operator
	GeneralizedMatrix<_T>& operator=(const GeneralizedMatrix<_T>& other) 
	{
		// Check for self-assignment
		if (this != &other) 
		{  
			this->isSparse_		= other.isSparse_;
			this->Nh_			= other.Nh_;
			if (isSparse_)
			{
				this->H_dense_.clear();
				this->H_sparse_	= other.H_sparse_;
			}
			else
			{
				this->H_sparse_.clear();
				this->H_dense_	= other.H_dense_;
			}
		}
		return *this;
	}

	// Move assignment operator
	GeneralizedMatrix<_T>& operator=(GeneralizedMatrix<_T>&& other) noexcept 
	{
		// Check for self-assignment
		if (this != &other) 
		{  
			this->isSparse_		= other.isSparse_;
			this->Nh_			= other.Nh_;
			this->n_cols		= other.n_cols;
			this->n_rows		= other.n_rows;
			if (isSparse_)
			{
				this->H_dense_.clear();
				this->H_sparse_	= std::move(other.H_sparse_);
			}
			else
			{
				this->H_sparse_.clear();
				this->H_dense_	= std::move(other.H_dense_);
			}
		}
		return *this;
	}

	// Assignment operator for dense matrices
	GeneralizedMatrix<_T>& operator=(const arma::Mat<_T>& _H)
	{
		this->isSparse_		= false;
		this->H_sparse_.clear();
		this->H_dense_		= _H;
		this->n_cols		= _H.n_cols;
		this->n_rows		= _H.n_rows;
		return *this;
	}

	// Assignment operator for sparse matrices
	GeneralizedMatrix<_T>& operator=(const arma::SpMat<_T>& _H)
	{
		this->isSparse_ = true;
		this->H_dense_.clear();
		this->H_sparse_ = _H;
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		return *this;
	}

	// Move assignment operator for dense matrices
	GeneralizedMatrix<_T>& operator=(const arma::Mat<_T>&& _H)
	{
		this->isSparse_ = false;
		this->H_sparse_.clear();
		this->H_dense_	= std::move(_H);
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		return *this;
	}

	// Move assignment operator for sparse matrices
	GeneralizedMatrix<_T>& operator=(const arma::SpMat<_T>&& _H)
	{
		this->isSparse_ = true;
		this->H_dense_.clear();
		this->H_sparse_ = std::move(_H);
		this->n_cols	= _H.n_cols;
		this->n_rows	= _H.n_rows;
		return *this;
	}
	// ##################################

	// ---- MULTIPLICATION OPERATORS ----
	
	// ##################################

	template<typename _T2>
	typename std::enable_if<std::is_arithmetic<_T2>::value, GeneralizedMatrix<_T>>::type
		operator*(_T2 _val) const
	{
		GeneralizedMatrix<_T> _result = *this;
		if (this->isSparse_)
			_result.H_sparse_ *= _val;
		else
			_result.H_dense_ *= _val;
		return _result;
	}

	// vectors and columns
	template<typename _OtherType> const
	typename std::enable_if<
			std::is_same_v<_OtherType, arma::Col<typename _OtherType::elem_type>>			|| 
			std::is_same_v<_OtherType, arma::subview_col<typename _OtherType::elem_type>>	|| 
			std::is_same_v<_OtherType, arma::subview<typename _OtherType::elem_type>>,
	arma::Col<typename std::common_type<_T, typename _OtherType::elem_type>::type>>::type
		operator*(const _OtherType& other) const
	{
		using ResultType = typename std::common_type<_T, typename _OtherType::elem_type>::type;
		if (this->isSparse_)
			return arma::Col<ResultType>(algebra::cast<ResultType>(this->H_sparse_) * algebra::cast<ResultType>(other));
		return arma::Col<ResultType>(algebra::cast<ResultType>(this->H_dense_) * algebra::cast<ResultType>(other));
	}

	template<typename _OtherType>
	typename std::enable_if<
			std::is_same_v<_OtherType, arma::Mat<typename _OtherType::elem_type>> || 
			std::is_same_v<_OtherType, arma::SpMat<typename _OtherType::elem_type>>,
	GeneralizedMatrix<typename std::common_type<_T, typename _OtherType::elem_type>::type>>::type
		operator*(const _OtherType& other) const 
	{
		using ResultType	= typename std::common_type<_T, typename _OtherType::elem_type>::type;
		GeneralizedMatrix<ResultType> _result(this->Nh_, this->isSparse_);

		// check the same type of matrix - matrix multiplication shall return GeneralMatrix type
		if constexpr (std::is_same_v<_OtherType, arma::Mat<_T>> || std::is_same_v<_OtherType, arma::SpMat<_T>>) 
		{
			if (other.n_rows != this->Nh_) 
				throw std::invalid_argument("Matrix rows must match vector/matrix rows.");
			if (this->isSparse_)
				_result.H_sparse_	= algebra::cast<ResultType>(this->H_sparse_) * algebra::cast<ResultType>(other);
			else
				_result.H_dense_	= algebra::cast<ResultType>(this->H_dense_) * algebra::cast<ResultType>(other);
		} 
		else 
			static_assert(arma::is_arma_type<_OtherType>::value, "Invalid type for matrix multiplication.");
		return _result;
	}

	// ##################################

	// Override addition operator for dense and sparse matrices
	template<typename _T2>
	GeneralizedMatrix<typename std::common_type<_T, _T2>> operator+(const GeneralizedMatrix<_T2>& other) const 
	{
		using _common = typename std::common_type<_T, _T2>;

		GeneralizedMatrix<_common> result(this->Nh_, this->isSparse_);

		if (this->isSparse_ && other.isSparse_)
			result.H_sparse_ = algebra::cast<_common>(this->H_sparse_) + algebra::cast<_common>(other.H_sparse_);
		else if (!this->isSparse_ && !other.isSparse_)
			result.H_dense_ = algebra::cast<_common>(this->H_dense_) + algebra::cast<_common>(other.H_dense_);
		else 
		{
			// Convert sparse to dense or dense to sparse for addition
			result.reset();
			result.H_dense_ = arma::Mat<_common>(algebra::cast<_common>(H_sparse_)) + algebra::cast<_common>(other.H_dense_);
		}
		return result;
	}

	// Addition operator overload to add HamiltonianMatrix with arma::Mat
	template<typename _T2>
	friend GeneralizedMatrix<typename std::common_type<_T, _T2>> operator+(const GeneralizedMatrix<_T2>& lhs, const arma::Mat<_T2>& rhs)
	{
		using _common = typename std::common_type<_T, _T2>;
		GeneralizedMatrix<_common> result(lhs);
		if (lhs.isSparse_)
			result.H_sparse_ += algebra::cast<_common>(rhs);
		else
			result.H_dense_ += algebra::cast<_common>(rhs);
		return result;
	}

	GeneralizedMatrix<_T>& operator+=(const arma::Mat<_T>& rhs)
	{
		if (this->isSparse_)
			this->H_sparse_ += algebra::cast<_T>(rhs);
		else
			this->H_dense_ += algebra::cast<_T>(rhs);
		return *this;
	}

	// Addition operator overload to add HamiltonianMatrix with arma::SpMat
	template<typename _T2>
	friend GeneralizedMatrix<typename std::common_type<_T, _T2>> operator+(const GeneralizedMatrix<_T2>& lhs, const arma::SpMat<_T2>& rhs)
	{
		using _common = typename std::common_type<_T, _T2>;
		GeneralizedMatrix<_common> result(lhs);
		if (lhs.isSparse_)
			result.H_sparse_ += algebra::cast<_common>(rhs);
		else
			result.H_dense_ += algebra::cast<_common>(rhs);
		return result;
	}

	GeneralizedMatrix<_T>& operator+=(const arma::SpMat<_T>& rhs)
	{
		if (this->isSparse_)
			this->H_sparse_ += algebra::cast<_T>(rhs);
		else
			this->H_dense_ += algebra::cast<_T>(rhs);
		return *this;
	}

	// ##################################

	void print() const 
	{
		if (isSparse_) 
			this->H_sparse_.print("Sparse Matrix:");
		else 
			this->H_dense_.print("Dense Matrix:");
	}

	void reset()
	{
		this->H_sparse_.clear();
		this->H_dense_.clear();
	}

	// ##################################

	// Override cast operator to convert to arma::Mat<_T>
	operator arma::Mat<_T>() const 
	{
		if(this->isSparse_)
			return arma::Mat<_T>(H_sparse_);
		return H_dense_;
	}

	// Override cast operator to convert to arma::SpMat<_T>
	operator arma::SpMat<_T>() const
	{
		if (this->isSparse_)
			return H_sparse_;
		return arma::SpMat<_T>(H_dense_);
	}

	// ##################################

	_T trace() const
	{
		if (this->isSparse_)
			return arma::trace(this->H_sparse_);
		return arma::trace(this->H_dense_);
	}

	_T norm(const char* method = "fro") const
	{
		if (this->isSparse_)
			return arma::norm(this->H_sparse_, method);
		return arma::norm(this->H_dense_, method);
	}

	// Maximum element of the matrix
	_T max() const {
		if (this->isSparse_) {
			return arma::max(arma::nonzeros(this->H_sparse_));
		}
		return arma::max(arma::vectorise(this->H_dense_));
	}

	// Minimum element of the matrix
	_T min() const {
		if (this->isSparse_) {
			return arma::min(arma::nonzeros(this->H_sparse_));
		}
		return arma::min(arma::vectorise(this->H_dense_));
	}

	// Mean of the matrix elements
	_T mean() const {
		if (this->isSparse_) {
			return arma::mean(arma::nonzeros(H_sparse_));
		}
		return arma::mean(arma::vectorise(this->H_dense_));
	}


};

// ############################################################################################################

template<typename _T>
inline double GeneralizedMatrix<_T>::meanLevelSpacing() const
{
	if (this->isSparse_)
	{
		const auto _Nh		= this->n_rows;
		const auto _trace	= algebra::cast<_T>(arma::trace(this->H_sparse_)) / (double)_Nh;
		const auto _trace2	= algebra::cast<_T>(arma::trace(arma::square(this->H_sparse_))) / (double)_Nh;
		return algebra::cast<double>(_trace2 - _trace * _trace);
	}
	const auto _Nh		= this->n_rows;
	const auto _trace	= algebra::cast<_T>(arma::trace(this->H_dense_)) / (double)_Nh;
	const auto _trace2	= algebra::cast<_T>(arma::trace(arma::square(this->H_dense_))) / (double)_Nh;
	return algebra::cast<double>(_trace2 - _trace * _trace);
}

// ############################################################################################################

template<typename _T>
inline arma::Col<_T> GeneralizedMatrix<_T>::diag() const
{
	if (this->isSparse_)
		return static_cast<arma::Col<_T>>(this->H_sparse_.diag());
	return this->H_dense_.diag();
}

// ############################################################################################################

template<typename _T>
inline arma::Col<_T> GeneralizedMatrix<_T>::diag(size_t _k) const
{
	if (this->isSparse_)
		return static_cast<arma::Col<_T>>(this->H_sparse_.diag(_k));
	return this->H_dense_.diag(_k);
}

// ############################################################################################################

/*
* @brief Method to symmetrize the Hamiltonian matrix.
* A = 0.5*(A + A^T)
* @returns void
*/
template<typename _T>
inline void GeneralizedMatrix<_T>::symmetrize()
{
	if (this->isSparse_)
		this->H_sparse_ = 0.5 * (this->H_sparse_ + this->H_sparse_.t());
	else
		this->H_dense_ = 0.5 * (this->H_dense_ + this->H_dense_.t());
}

// ############################################################################################################

/*
* @brief Method to standarize the Hamiltonian matrix.
* A = A - tr(A)/n_rows
* A = A / sqrt(tr(A^2)/n_rows)
* The operator becomes traceless and normalized - with a Hilbert-Schmidt norm of 1.
* @returns void
*/
template<typename _T>
inline void GeneralizedMatrix<_T>::standarize()
{
	if(this->isSparse_)
	{
		this->H_sparse_.diag() -= arma::trace(this->H_sparse_) / (double)this->n_rows;
		auto _Hs				= arma::trace(arma::square(this->H_sparse_)) / (double)this->n_rows;
		this->H_sparse_			= this->H_sparse_ / std::sqrt(_Hs);
	}
	else
	{
		this->H_dense_.diag() -= arma::trace(this->H_dense_) / (double)this->n_rows;
		auto _Hd				= arma::trace(arma::square(this->H_dense_)) / (double)this->n_rows;
		this->H_dense_			= this->H_dense_ / std::sqrt(_Hd);
	}
}

// ############################################################################################################

template<typename _T>
using GeneralizedMatrixFunction = std::function<GeneralizedMatrix<_T>(size_t _Ns)>;