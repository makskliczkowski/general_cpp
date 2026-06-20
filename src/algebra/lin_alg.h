/******************************************************************************
 *
 *  @file src/lin_alg.h
 *  @brief Linear algebra aliases and helper functions using Armadillo.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/
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

// All backend (Armadillo / BLAS / HDF5 / MKL) configuration is centralized in
// backend_config.h and driven by the build system, rather than hardcoded here.
#include "backend_config.h"
// #######################################################################################################################

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
			std::is_same_v<_T, arma::SpMat<std::complex<double>>>					||
			std::is_same_v<_T, arma::subview<double>>								||
			std::is_same_v<_T, arma::subview<std::complex<double>>>					||
			std::is_same_v<_T, arma::subview_cols<double>>							||
			std::is_same_v<_T, arma::subview_cols<std::complex<double>>>			||
			std::is_same_v<_T, arma::subview_cube<double>>							||
			std::is_same_v<_T, arma::subview_cube<std::complex<double>>>;

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

#include "../maths/maths.h"
#include "../common/files.h"

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

	// namespace Physics
	// {
	// 	template <typename _CT, _MT>
	// 	inline _CT quench_state()
	// };

	/**
	* @brief Transforms a state vector to a new basis using a unitary matrix.
	* 
	* This function takes a unitary matrix and a state vector, and transforms the state vector to a new basis defined by the unitary matrix.
	* 
	* @tparam _T The type of the elements in the matrix and vector.
	* @tparam _MatType The type of the unitary matrix. Must be a valid Armadillo matrix type.
	* @tparam _VecType The type of the state vector. Must be a valid Armadillo vector type or std::vector.
	* 
	* @param unitaryMatrix The unitary matrix used for the basis transformation.
	* @param stateVector The state vector to be transformed.
	* 
	* @return The transformed state vector in the new basis.
	* 
	* @note The function uses static assertions to ensure that the matrix and vector types are valid.
	*       If the element types of the matrix and vector are the same, the transformation is performed directly.
	*       If the vector is an Armadillo vector type, it is converted to the appropriate type before transformation.
	*       If the vector is a std::vector, it is converted to an Armadillo column vector before transformation and then converted back to std::vector.
	*/
	template <typename _MatType, typename _VecType>
	inline auto change_basis(const _MatType& unitaryMatrix, const _VecType& stateVector)
	{
		static_assert(HasMatrixType<_MatType>, "The matrix type must be a valid Armadillo matrix type.");
		static_assert(HasArmaVectorType<_VecType> || std::is_same_v<_VecType, std::vector<typename _VecType::value_type>>, "The vector type must be a valid Armadillo vector type or std::vector.");

		using VecElemType 	= typename _VecType::value_type;
		using MatElemType 	= typename _MatType::elem_type;
		using CommonType 	= typename std::common_type<VecElemType, MatElemType>::type;

		if constexpr (std::is_same_v<VecElemType, MatElemType>)
		{
			return unitaryMatrix * stateVector;
		}
		else if constexpr (HasArmaVectorType<_VecType>)
		{
			if constexpr (std::is_same_v<_VecType, arma::Col<CommonType>> || std::is_same_v<_VecType, arma::Row<CommonType>>) 
			{
				return unitaryMatrix * stateVector;
			} 
			else {
				arma::Col<CommonType> transformedVec = unitaryMatrix * arma::conv_to<arma::Col<CommonType>>::from(stateVector);
				return arma::conv_to<_VecType>::from(transformedVec);
			}
		}
		else if constexpr (std::is_same_v<_VecType, std::vector<VecElemType>>)
		{
			arma::Col<CommonType> armaVec(stateVector.data(), stateVector.size(), false, true);
			arma::Col<CommonType> transformedVec = unitaryMatrix * armaVec;
			return std::vector<CommonType>(transformedVec.begin(), transformedVec.end());
		}
	}

	/**
	* @brief Changes the basis of a given matrix using a unitary matrix.
	*
	* This function transforms the given matrix to a new basis defined by the unitary matrix.
	* The transformation is performed as follows: U^T * A * U, where U is the unitary matrix
	* and A is the matrix to be transformed.
	*
	* @tparam _T The type of the elements in the matrices.
	* @tparam _MatType1 The type of the unitary matrix.
	* @tparam _MatType2 The type of the matrix to be transformed.
	* @param unitaryMatrix The unitary matrix used for the basis change.
	* @param matrix The matrix to be transformed.
	* @param back If false, the transformation is performed as U * A * U^T, instead of U^T * A * U.
	* @return The matrix transformed to the new basis.
	*
	* @note Both matrix types must be valid Armadillo matrix types.
	* @note If the element types of the two matrices are different, the function will convert
	*       the matrices to a common type before performing the transformation.
	*/
	template <typename _MatType1, typename _MatType2 = _MatType1>
	inline arma::Mat<typename std::common_type<typename _MatType1::elem_type, typename _MatType2::elem_type>::type> 
	change_basis_matrix(const _MatType1& unitaryMatrix, const _MatType2& matrix, bool back = false)
		requires HasMatrixType<_MatType1> && HasMatrixType<_MatType2>
	{
		using MatElemType1 	= typename _MatType1::elem_type;
		using MatElemType2 	= typename _MatType2::elem_type;
		using CommonType 	= typename std::common_type<MatElemType1, MatElemType2>::type;

		// if are the same inner types
		if constexpr (std::is_same<MatElemType1, MatElemType2>::value)
		{
			if (!back)
				return (unitaryMatrix * matrix) * unitaryMatrix.t();
			else
				return (unitaryMatrix.t() * matrix) * unitaryMatrix;
			return (unitaryMatrix.t() * matrix) * unitaryMatrix;
		}
		else
		{
			if (!back)
				return (unitaryMatrix * arma::conv_to<arma::Mat<CommonType>>::from(matrix)) * unitaryMatrix.t();
			return (unitaryMatrix.t() * arma::conv_to<arma::Mat<CommonType>>::from(matrix)) * unitaryMatrix;
		}
	}

	// #################################################################################################################################################
	

#include "lin_alg/matmul.h"
#include "lin_alg/solvers.h"
#include "lin_alg/ode.h"
#include "lin_alg/pfaffian.h"
#include "lin_alg/udt.h"
#include "lin_alg/many_body.h"


// dynamic bitset
#include "../common/str.h"
#include "../common/directories.h"

// ###################################################### S A V E R ######################################################

/**
* @brief Save the algebraic matrix (or subview) to a file with a specific path. The file can be in binary, text, or HDF5 format.
* @param _path Path to the directory where the file will be saved.
* @param _file Name of the file to save the matrix to.
* @param _toSave Matrix or subview to save.
* @param _db Name of the dataset in the HDF5 file (default is "weights").
* @param _app Append to the file if true, otherwise overwrite (default is false).
* @returns True if the file was saved successfully, false otherwise.
*/
template <typename _T>
requires HasMatrixType<_T>
inline bool saveAlgebraic(const std::string& _path, const std::string& _file, const _T& _toSave, const std::string& _db = "weights", bool _app = false)
{
	// Copy the subview to a new matrix if it is a subview, otherwise return the original matrix as-is (no copy)
	auto savable = [&]() -> decltype(auto) {
		if constexpr (arma::is_subview<_T>::value || arma::is_subview_cols<_T>::value) {
			return arma::Mat<typename _T::elem_type>(_toSave); // Copy subview to a new matrix
		} else {
			return _toSave; // Return the original matrix as-is (no copy)
		}
	}();

#ifdef _DEBUG
	// LOGINFO(_path + _file, LOG_TYPES::INFO, 3);
#endif
	createDir(_path);
	bool _isSaved = false;

#ifdef HAS_CXX20
	if (_file.ends_with(".h5"))
#else
	if (endsWith(_file, ".h5"))
#endif
	{
		if (!_app)
			_isSaved = savable.save(arma::hdf5_name(_path + _file, _db));
		else
			_isSaved = savable.save(arma::hdf5_name(_path + _file, _db, arma::hdf5_opts::append));
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".bin"))
#else
	if (endsWith(_file, ".bin"))
#endif
		_isSaved = savable.save(_path + _file);
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		if (!_app)
			_isSaved = savable.save(_path + _file, arma::arma_ascii);
		else
		{
			std::ofstream _out;
			try
			{
				_out.open(_path + _file, std::ios::app);
				_isSaved = _out.is_open();
			}
			catch (std::exception& e)
			{
				// LOGINFO(e.what(), LOG_TYPES::ERROR, 2);
			}
			_out << savable;
			_out.close();
		}
	}
	std::cout << "\t\t\t\tSaved: " << _path + _file << " with db: " << _db << std::endl;
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
	if (endsWith(_file, ".bin"))
#endif
		_isSaved	= _toSave.save(_path + _file);
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
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
	std::cout << "\t\t\t\tSaved: " << _path + _file << " with db: " << _db << std::endl;
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
	if (endsWith(_file, ".bin"))
#endif
	{
		_toSet.load(_path + _file);
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
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
	if (endsWith(_file, ".bin"))
#endif
	{
		_toSet.load(_path + _file);
		return true;
	}
#ifdef HAS_CXX20
	else if (_file.ends_with(".txt") || _file.ends_with(".dat"))
#else
	if (endsWith(_file, ".txt") || endsWith(_file, ".dat"))
#endif
	{
		_toSet.load(_path + _file, arma::arma_ascii);
		return true;
	}
	return _isSaved;
}



#endif