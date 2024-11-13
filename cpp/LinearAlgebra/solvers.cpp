#include "../../src/lin_alg.h"
#include "../../src/Include/maths.h"
#include "../../src/Include/str.h"
#include "../../src/flog.h"
#include "armadillo"
#include <cassert>
#include <complex>
#include <stdexcept>
#include <stdlib.h>


// #################################################################################################################################################

// OTHER SOLVER METHODS

// #################################################################################################################################################

namespace algebra
{
    namespace Solvers
    {
        // #################################################################################################################################################

        /**
		* @brief  Stable Symmetric Householder reflection. Usage: sym_ortho(a, b, c, s, r) - modifies c, s, and r
        * The reflectors from Algorithm 1 in Choi and Saunders (2005) for real a and b, which is a stable form for computing
		* r = √a2 + b2 ≥ 0, c = a/r , and s = b/r 
		* @note Is a Givens rotation matrix
		* @param a first value - first element of a two-vector  [a; b]
		* @param b second value - second element of a two-vector [a; b]
		* @param c = a/r - cosine(theta), where theta is the angle of rotation (counter-clockwise) in a plane-rotation matrix.
		* @param s = b/r - sine(theta)
		* @param r = √a2 + b2 - the norm of the vector [a; b] (||[a; b]||)
        * @note Stable symmetric Householder reflection that gives c and s such that 
        *          [ c  s ][a] = [d],
        *          [ s -c ][b]   [0]  
        *       where d = two-norm of vector [a, b],
        *          c = a / sqrt(a^2 + b^2) = a / d, 
        *          s = b / sqrt(a^2 + b^2) = b / d.
        *       The implementation guards against overlow in computing sqrt(a^2 + b^2).
        * @ref Algorithm 4.9, stable *unsymmetric* Givens rotations in Golub and van Loan's book Matrix Computations, 3rd edition. 
        * @ref https://www.mathworks.com/matlabcentral/fileexchange/42419-minres-qlp
		*/
        template <typename _T1>
		void sym_ortho(_T1 a, _T1 b, _T1& c, _T1& s, _T1& r)
        {
			if (b == _T1(0))
			{
				if (a == _T1(0))
					c = 1;
				else 
					c = sgn<_T1>(a);
				s = 0;
				r = std::abs(a);
			}
			else if (a == _T1(0))
			{
				c = 0;
				s = sgn<_T1>(b);
				r = std::abs(b);
			}
			else if (abs(b) > abs(a))
			{
				_T1 tau = a / b;
				s 		= sgn<_T1>(b) / std::sqrt(_T1(1.0) + (tau * tau));
				c 		= s * tau;
				r 		= b / s; // computationally better than d = a / c since | c | <= | s |
			}
			else 
			{
				_T1 tau = b / a;
				c 		= sgn<_T1>(a) / std::sqrt(_T1(1.0) + (tau * tau));
				s 		= c * tau;
				r 		= a / c; // computationally better than d = b / s since | s | <= | c |
			}
		}

        /**
        * @brief  Stable Symmetric Householder reflection. Usage: sym_ortho(a, b, c, s, r) - modifies c, s, and r - for complex numbers
        * Refere to the real version for more details.
        */
        template <>
        void sym_ortho(std::complex<double> a, std::complex<double> b, std::complex<double>& c, std::complex<double>& s, std::complex<double>& r)
        {
            const auto _absa = std::abs(a);
            const auto _absb = std::abs(b);
            const auto _sgna = sgn<std::complex<double>>(a);
            const auto _sgnb = sgn<std::complex<double>>(b);

            // special case when a or b is zero
            if (b == std::complex<double>(0.0))
            {
                c = 1.0;
                s = 0.0;
                r = a;
            }
            else if (a == std::complex<double>(0.0))
            {
                c = 0.0;
                s = 1.0;
                r = b;
            }
            else if (_absb > _absa)
            {
                std::complex<double> tau = _absa / _absb;
                c       = 1.0 / std::sqrt(1.0 + tau * tau); // temporary 
                s 		= c * algebra::conjugate(_sgnb / _sgna);
                c       = c * tau;
                r       = b / algebra::conjugate(s);
            }
            else 
            {
                std::complex<double> tau = _absb / _absa;
                c       = 1.0 / std::sqrt(1.0 + tau * tau);
                s 		= c * algebra::conjugate(_sgna / _sgnb) * tau;
                r       = a / c;
            }
        }

        // the template specializations
        template void sym_ortho<double>(double a, double b, double& c, double& s, double& r);
        // #################################################################################################################################################
    
		namespace Preconditioners
		{
			// template specializations of the preconditioner class
			template class Preconditioner<double, true>;
			template class Preconditioner<std::complex<double>, true>;
			template class Preconditioner<double, false>;
			template class Preconditioner<std::complex<double>, false>;
		};
	};
};

// #################################################################################################################################################

// FOR GENERAL TYPES 

// #################################################################################################################################################

namespace algebra
{
	namespace Solvers
	{
		namespace General
		{
			/**
			* @brief Matrix-vector multiplication for the general case.
			* @param _A The matrix A.
			* @param _x The vector x.
			* @param _reg The regularization parameter.
			*/
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _A, const arma::Col<_T>& _x, const double _reg)
			{
				arma::Col <_T> _out = _A * _x;
				if (_reg > 0.0)
					return _out + _reg * _x;
				else
					return _out;
			}

			/**
			* @brief Matrix-vector multiplication with a sparse matrix for the general case.
			* @param _A The sparse matrix A.
			* @param _x The vector x.
			* @param _reg The regularization parameter.
			*/
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::SpMat<_T>& _A, const arma::Col<_T>& _x, const double _reg)
			{
				arma::Col <_T> _out = _A * _x;
				if (_reg > 0.0)
					return _out + _reg * _x;
				else
					return _out;
			}

			// template specializations
			template arma::Col<double> matrixFreeMultiplication(const arma::Mat<double>& _A, const arma::Col<double>& _x, const double _reg);
			template arma::Col<std::complex<double>> matrixFreeMultiplication(const arma::Mat<std::complex<double>>& _A, const arma::Col<std::complex<double>>& _x, const double _reg);
			// sparse matrix
			template arma::Col<double> matrixFreeMultiplication(const arma::SpMat<double>& _A, const arma::Col<double>& _x, const double _reg);
			template arma::Col<std::complex<double>> matrixFreeMultiplication(const arma::SpMat<std::complex<double>>& _A, const arma::Col<std::complex<double>>& _x, const double _reg);
			// #################################################################################################################################################
		};
	};
};

// *************************************************************************************************************************************************

// #################################################################################################################################################

// METHODS FOR GENERAL SOLVERS

namespace algebra 
{
	namespace Solvers 
	{
		namespace General 
		{            
			/**
			* @brief Solve the linear system using the specified solver type and the matrix-free multiplication function
			* @param _type Type of the solver
			*/
			template <typename _T1, bool _symmetric>
			arma::Col<_T1> solve(Type _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, _symmetric))
			{
				switch (_type) 
				{
				case Type::ARMA:
				{
					LOGINFO("Solvers::Fisher::solve: ARMA solver is not implemented for the matrix-free multiplication.", LOG_TYPES::ERROR, 3);
					return _F;
				}
				case Type::ConjugateGradient:
				{
					if constexpr (_symmetric == true)
						return General::CG::conjugate_gradient<_T1>(_matrixFreeMultiplication, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The ConjugateGradient solver is only implemented for positive semidefinite matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Type::MINRES: 
				{
					if constexpr (_symmetric == true)
						return General::MINRES::minres<_T1>(_matrixFreeMultiplication, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The MINRES solver is only implemented for symmetric matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Type::MINRES_QLP:
				{
					if constexpr (_symmetric == true)
						return General::MINRES_QLP::minres_qlp<_T1>(_matrixFreeMultiplication, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The MINRES_QLP solver is only implemented for symmetric matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Type::PseudoInverse:
				{
					LOGINFO("Solvers::Fisher::solve: PseudoInverse solver is not implemented for the matrix-free multiplication.", LOG_TYPES::ERROR, 3);
					return _F;
				}
				case Type::Direct:
				{
					LOGINFO("Solvers::Fisher::solve: Direct solver is not implemented for the matrix-free multiplication.", LOG_TYPES::ERROR, 3);
					return _F;
				}
				default:
				{
					if constexpr (_symmetric == true)
						return General::CG::conjugate_gradient<_T1>(_matrixFreeMultiplication, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The ConjugateGradient solver is only implemented for positive semidefinite matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				}
			}

			template <typename _T1, bool _symmetric>
			arma::Col<_T1> solve(int _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, _symmetric)) { return General::solve<_T1, _symmetric>(static_cast<Type>(_type), _matrixFreeMultiplication, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
			
			// *********************************************************************************************************************************************
			
			/**
			* @brief Solve the linear system using the specified solver type 
			* @param _type Type of the solver
			*/
			template <typename _T1, bool _symmetric>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(_T1, _symmetric))
			{
				switch (_type) 
				{
				case Solvers::General::Type::ARMA:
				{
					if (_symmetric == true)
						return arma::solve(_A + _reg * arma::eye(_A.n_rows, _A.n_cols), _F, arma::solve_opts::likely_sympd);
					else
						return arma::solve(_A + _reg * arma::eye(_A.n_rows, _A.n_cols), _F);
				}
				case Solvers::General::Type::ConjugateGradient:
				{
					if constexpr (_symmetric == true)
						return General::CG::conjugate_gradient<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The ConjugateGradient solver is only implemented for positive semidefinite matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Solvers::General::Type::MINRES:
				{
					if constexpr (_symmetric == true)
						return General::MINRES::minres<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The MINRES solver is only implemented for symmetric matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Solvers::General::Type::MINRES_QLP:
				{
					if constexpr (_symmetric == true)
						return General::MINRES_QLP::minres_qlp<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The MINRES_QLP solver is only implemented for symmetric matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Solvers::General::Type::PseudoInverse:
					return arma::pinv(_A, _eps) * _F;
				case Solvers::General::Type::Direct:
					return arma::inv(_A) * _F;
				default:
				{
					if constexpr (_symmetric == true)
						return General::CG::conjugate_gradient<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The ConjugateGradient solver is only implemented for positive semidefinite matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				}
			}

			template <typename _T1, bool _symmetric>
			arma::Col<_T1> solve(int _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(_T1, _symmetric)) { return General::solve<_T1, _symmetric>(static_cast<Solvers::General::Type>(_type), _A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }

			// *********************************************************************************************************************************************
			
			/**
			* @brief Solve the linear system using the specified solver type - sparse matrix
			* @param _type Type of the solver
			*/
			template <typename _T1, bool _symmetric>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(_T1, _symmetric))
			{
				switch (_type) 
				{
				case Solvers::General::Type::ARMA:
				{
					if (_symmetric == true)
						return arma::solve(_A + _reg * arma::eye(_A.n_rows, _A.n_cols), _F, arma::solve_opts::likely_sympd);
					else
						return arma::solve(_A + _reg * arma::eye(_A.n_rows, _A.n_cols), _F);
				}
				case Solvers::General::Type::ConjugateGradient:
				{
					if constexpr (_symmetric == true)
						return General::CG::conjugate_gradient<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The ConjugateGradient solver is only implemented for positive semidefinite matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Solvers::General::Type::MINRES:
				{
					if constexpr (_symmetric == true)
						return General::MINRES::minres<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The MINRES solver is only implemented for symmetric matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Solvers::General::Type::MINRES_QLP:
				{
					if constexpr (_symmetric == true)
						return General::MINRES_QLP::minres_qlp<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The MINRES_QLP solver is only implemented for symmetric matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				case Solvers::General::Type::PseudoInverse:
					return arma::pinv(arma::Mat<_T1>(_A), _eps) * _F;
				case Solvers::General::Type::Direct:
					return arma::inv(arma::Mat<_T1>(_A)) * _F;
				default:
				{
					if constexpr (_symmetric == true)
						return General::CG::conjugate_gradient<_T1>(_A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
					else
					{
						LOGINFO("The ConjugateGradient solver is only implemented for positive semidefinite matrices.", LOG_TYPES::ERROR, 3);
						return _F;
					}
				}
				}
			}	

			template <typename _T1, bool _symmetric>
			arma::Col<_T1> solve(int _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(_T1, _symmetric)) { return General::solve<_T1, _symmetric>(static_cast<Solvers::General::Type>(_type), _A, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }		
			
			// *********************************************************************************************************************************************

			template <typename _T1, bool _symmetric>
			inline arma::Col<_T1> solve(Type _type, SOLVE_MATMUL_ARG_TYPES(_T1)){ return General::solve<_T1, _symmetric>(_type, _matrixFreeMultiplication, _F, _x0, nullptr, _eps, _max_iter, _converged, _reg); }
			
			template <typename _T1, bool _symmetric>
			inline arma::Col<_T1> solve(int _type, SOLVE_MATMUL_ARG_TYPES(_T1)) { return General::solve<_T1, _symmetric>(static_cast<Type>(_type), _matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg); }

			// *********************************************************************************************************************************************

			template <typename _T1, bool _symmetric>
			inline arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES(_T1)){ return General::solve<_T1, _symmetric>(_type, _A, _F, _x0, nullptr, _eps, _max_iter, _converged, _reg); }

			template <typename _T1, bool _symmetric>
			inline arma::Col<_T1> solve(int _type, SOLVE_MAT_ARG_TYPES(_T1)) { return General::solve<_T1, _symmetric>(static_cast<Solvers::General::Type>(_type), _A, _F, _x0, _eps, _max_iter, _converged, _reg); }

			// *********************************************************************************************************************************************

			template <typename _T1, bool _symmetric>
			inline arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES(_T1)){ return General::solve<_T1, _symmetric>(_type, _A, _F, _x0, nullptr, _eps, _max_iter, _converged, _reg); }

			template <typename _T1, bool _symmetric>
			inline arma::Col<_T1> solve(int _type, SOLVE_SPMAT_ARG_TYPES(_T1)) { return General::solve<_T1, _symmetric>(static_cast<Solvers::General::Type>(_type), _A, _F, _x0, _eps, _max_iter, _converged, _reg); }

            // -----------------------------------------------------------------------------------------------------------------------------------------

            // define the template specializations
            // double
			template arma::Col<double> solve(Type _type, SOLVE_MATMUL_ARG_TYPES(double));
			template arma::Col<double> solve(int _type, SOLVE_MATMUL_ARG_TYPES(double));
			template arma::Col<double> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES(double));
			template arma::Col<double> solve(int _type, SOLVE_MAT_ARG_TYPES(double));
			template arma::Col<double> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES(double));
			template arma::Col<double> solve(int _type, SOLVE_SPMAT_ARG_TYPES(double));
			// double with preconditioner
			template arma::Col<double> solve(Type _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
			template arma::Col<double> solve(int _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
			template arma::Col<double> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(double, true));
			template arma::Col<double> solve(int _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(double, true));
			template arma::Col<double> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(double, true));
			template arma::Col<double> solve(int _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(double, true));
			// and false
			template arma::Col<double> solve(Type _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, false));
			template arma::Col<double> solve(int _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, false));
			template arma::Col<double> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(double, false));
			template arma::Col<double> solve(int _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(double, false));
			template arma::Col<double> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(double, false));
			template arma::Col<double> solve(int _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(double, false));
			// complex double
			template arma::Col<std::complex<double>> solve(Type _type, SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
			template arma::Col<std::complex<double>> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES(std::complex<double>));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_MAT_ARG_TYPES(std::complex<double>));
			template arma::Col<std::complex<double>> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES(std::complex<double>));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_SPMAT_ARG_TYPES(std::complex<double>));
			// complex double with preconditioner
			template arma::Col<std::complex<double>> solve(Type _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			template arma::Col<std::complex<double>> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			template arma::Col<std::complex<double>> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			// and false
			template arma::Col<std::complex<double>> solve(Type _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, false));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, false));
			template arma::Col<std::complex<double>> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, false));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_MAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, false));
			template arma::Col<std::complex<double>> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, false));
			template arma::Col<std::complex<double>> solve(int _type, SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(std::complex<double>, false));
			
			// #################################################################################################################################################
			
			/**
            * @brief Get the name of the solver type
            * @param _type Type of the solver
            * @return Name of the solver
            */
            std::string name(Type _type)
            {
                switch (_type) 
				{
				case Type::ARMA:
					return "ARMA";
				case Type::ConjugateGradient:
					return "Conjugate Gradient";
				case Type::MINRES:
					return "MINRES";
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

			std::string name(int _type) { return name(static_cast<Type>(_type)); } 

			// -----------------------------------------------------------------------------------------------------------------------------------------
		};
	};
};

// #################################################################################################################################################

// FOR THE FISHER MATRIX SOLVERS - SYMMETRIC MATRICES WITH FORM S = A^T * A, given A

// #################################################################################################################################################

namespace algebra {   
    namespace Solvers {
        namespace FisherMatrix {
			// #################################################################################################################################################

			/**
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
			arma::Col<_T> matrixFreeMultiplication(SOLVE_FISHER_MATRIX(_T), const arma::Col<_T>& _x, const double _reg)
			{
				const size_t _N 			= _DeltaO.n_rows;               	// Number of samples (rows)
				arma::Col<_T> _intermediate = _DeltaO * _x;     				// Calculate \Delta O * x

				// apply regularization on the diagonal
				if (_reg > 0.0)
					return (_DeltaO.t() * _intermediate) / static_cast<_T>(_N)  + _reg * _x;
				
				// no regularization
				return (_DeltaO.t() * _intermediate) / static_cast<_T>(_N);    // Calculate \Delta O^* * (\Delta O * v) / N
			}

			/** 
			* @brief Same as the above method, but with the conjugate transpose of the matrix \Delta O. 
			*/
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(SOLVE_FISHER_MATRICES(_T), const arma::Col<_T>& x, const double _reg)
			{
				const size_t _N 			= _DeltaO.n_rows;               	// Number of samples (rows)
				arma::Col<_T> _intermediate = _DeltaO * x;     					// Calculate \Delta O * x

				// apply regularization on the diagonal
				if (_reg > 0.0)
					return (_DeltaOConjT * _intermediate) / static_cast<_T>(_N)  + _reg * x;
				
				// no regularization
				return (_DeltaOConjT * _intermediate) / static_cast<_T>(_N);    // Calculate \Delta O^* * (\Delta O * v) / N
			}

			// template specializations
			template arma::Col<double> matrixFreeMultiplication(SOLVE_FISHER_MATRIX(double), const arma::Col<double>& _x, const double _reg);
			template arma::Col<std::complex<double>> matrixFreeMultiplication(SOLVE_FISHER_MATRIX(std::complex<double>), const arma::Col<std::complex<double>>& _x, const double _reg);
			// with the conjugate transpose
			template arma::Col<double> matrixFreeMultiplication(SOLVE_FISHER_MATRICES(double), const arma::Col<double>& x, const double _reg);
			template arma::Col<std::complex<double>> matrixFreeMultiplication(SOLVE_FISHER_MATRICES(std::complex<double>), const arma::Col<std::complex<double>>& x, const double _reg);

			// #################################################################################################################################################
        };
    };
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%