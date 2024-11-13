#include "../../src/lin_alg.h"
#include "../../src/flog.h"

namespace algebra {
    namespace Pfaffian {
        // #################################################################################################################################################

        /**
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

        /**
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
        
        /**
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
	
		/**
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
		_T pfaffian(const arma::Mat<_T>& A, arma::u64 N, PfaffianAlgorithms _alg)
		{
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

        // template specializations
        template double pfaffian_r(const arma::Mat<double>& A, arma::u64 N);
        template double pfaffian_hess(const arma::Mat<double>& A, arma::u64 N);
        template double pfaffian_p(arma::Mat<double> A, arma::u64 N);
        template double pfaffian_s(arma::Mat<double> A, arma::u64 N);
        template double pfaffian(const arma::Mat<double>& A, arma::u64 N, PfaffianAlgorithms _alg);
        template std::complex<double> pfaffian_r(const arma::Mat<std::complex<double>>& A, arma::u64 N);
        template std::complex<double> pfaffian_hess(const arma::Mat<std::complex<double>>& A, arma::u64 N);
        template std::complex<double> pfaffian_p(arma::Mat<std::complex<double>> A, arma::u64 N);
        template std::complex<double> pfaffian_s(arma::Mat<std::complex<double>> A, arma::u64 N);
        template std::complex<double> pfaffian(const arma::Mat<std::complex<double>>& A, arma::u64 N, PfaffianAlgorithms _alg);

    };
};