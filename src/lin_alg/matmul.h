#pragma once

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
	
