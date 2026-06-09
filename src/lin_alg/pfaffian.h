#pragma once

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
		// ############################################################################################################################################
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
		// ############################################################################################################################################

		/**
		* @brief Update the Pfaffian of a skew-symmetric matrix after a row and column update.
		* Using the Cayley identity, the Pfaffian of a skew-symmetric matrix can be updated
		* after a row and column update. The updated Pfaffian is calculated as:
		* P'(A) = -P(A) * dot(Ainv_row, updRow)
		* @param _pffA current Pfaffian
		* @param _Ainv inverse of the original matrix
		* @returns the Pfaffian of an updated skew-symmetric matrix
		* @note This function assumes that the input matrix is skew-symmetric. 
		* @note the logaritmic space is used to avoid numerical issues with the Pfaffian
		*/
		template <typename _Tp, typename _Trow, typename _Trow2 = _Trow>
		inline _Tp cayleys(_Tp _pffA, const _Trow& _Ainv_row, const _Trow2& _updRow)
		{
			// do this in the logaritmic space
			auto _logP 		= std::log(_pffA);
			auto _logDot 	= std::log(arma::dot(_Ainv_row, _updRow));
			return -std::exp(_logP + _logDot);
		}

		// ############################################################################################################################################
	};

	// #################################################################################################################################################

	/**
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

	/**
	* @brief Efficiently calculate the inverse of a skew-symmetric matrix using the Sherman-Morrison formula.
	* 
	* @tparam Matrix A generic matrix type supporting element access via `operator()` and `.size()`.
	* @tparam Vector A generic vector type supporting element access via `operator[]` and `.size()`.
	* @param _Ainv The inverse of the original skew-symmetric matrix.
	* @param _updIdx The index of the updated row.
	* @param _updRow The updated row, provided as a vector.
	* @return The inverse of the updated skew-symmetric matrix.
	*/
	template <typename Matrix, typename Vector>
	Matrix scherman_morrison_skew(const Matrix& _Ainv, size_t _updIdx, const Vector& _updRow) 
	{
		const size_t n 	= _Ainv.size(); 	// Assuming a square matrix with `size()` as rows or cols		
		auto _out 		= _Ainv;			// Copy the current inverse to the output matrix

		// Precompute dot products of the inverse matrix with the updated row
		Vector _dots(n, 0.0);
		if constexpr (std::is_same_v<decltype(_Ainv), arma::Mat<typename decltype(_Ainv)::elem_type>> 	&&
			(std::is_same_v<Vector, arma::Col<typename decltype(_Ainv)::elem_type>>  					|| 
			std::is_same_v<Vector, arma::Row<typename decltype(_Ainv)::elem_type>> 						|| 
			std::is_same_v<Vector, arma::subview_col<typename decltype(_Ainv)::elem_type>> 				|| 
			std::is_same_v<Vector, arma::subview_row<typename decltype(_Ainv)::elem_type>>))
		{
			// Precompute dot products of the inverse matrix with the updated row
			_dots = _Ainv * _updRow.as_col();
		} 
		else 
		{
			// Precompute dot products of the inverse matrix with the updated row
			Vector _dots(n, 0.0);
			for (size_t i = 0; i < n; ++i) 
			{
				for (size_t j = 0; j < n; ++j) 
				{
					_dots[i] += _Ainv(i, j) * _updRow[j];
				}
			}
		}

		// Precompute the inverse of the critical dot product (with safety for division by zero)
		const auto _dotProductInv = (_dots[_updIdx] != 0.0) ? (1.0 / _dots[_updIdx]) : 0.0;

		// Update the matrix using the Sherman-Morrison formula
		for(auto i = 0; i < n; ++i)
		{
			const auto _d_i_alpha = (i == _updIdx) ? 1.0 : 0.0;

			for(int j = 0; j < _Ainv.n_cols; j++)
			{
				const auto _d_j_alpha = (j == _updIdx) ? 1.0 : 0.0;

				_out(i, j) += _dotProductInv * ((_d_i_alpha - _dots(i)) * _Ainv(_updIdx, j) + (_dots(j) - _d_j_alpha) * _Ainv(_updIdx, i));

				// Enforce skew-symmetric property
				if (_d_i_alpha > 0 || _d_j_alpha > 0)
					_out(i, j) *= -1;
			}
		}

		return _out;
	}

	// #################################################################################################################################################
