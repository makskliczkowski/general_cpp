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

namespace algebra {
	namespace Solvers {
		namespace General {
			// #################################################################################################################################################
			namespace CG {
				/**
				* @brief Conjugate gradient solver for the general case.
				* @param _matrixFreeMultiplication The matrix-vector multiplication function. It is used to calculate the matrix-vector product Sx.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _eps The convergence criterion.
				* @param _max_iter The maximum number of iterations.
				* @param _converged The flag indicating if the solver converged.
				* @param _reg The regularization parameter. (A + \lambda I) x \approx b
				* @return The solution vector x.
				*/
                template <typename _T1>
                arma::Col<_T1> conjugate_gradient(SOLVE_MATMUL_ARG_TYPES(_T1))
				{
					// set the initial values for the solver
					arma::Col<_T1> x 	= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 	= _F - _matrixFreeMultiplication(x, _reg);
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
						Ap 					= _matrixFreeMultiplication(p, _reg);
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
					LOGINFO("Conjugate gradient solver did not converge.", LOG_TYPES::WARNING, 3);
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}

				/**
				* @brief Conjugate gradient solver for the general case with a preconditioner.
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
				arma::Col<_T1> conjugate_gradient(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, true))
				{
					if (_preconditioner == nullptr)
						return Solvers::General::CG::conjugate_gradient<_T1>(_matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg);

					// set the initial values for the solver
					arma::Col<_T1> x 	= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 	= _F - _matrixFreeMultiplication(x, _reg);	// calculate the first residual
					arma::Col<_T1> z 	= _preconditioner->apply(r);										// apply the preconditioner to Mz = r
					arma::Col<_T1> p 	= z;																// set the search direction
					arma::Col<_T1> Ap;																		// matrix-vector multiplication result

					_T1 rs_old 			= arma::cdot(r, z);													// the initial norm of the residual
					// _T1 initial_rs		= std::abs(rs_old);  											// For relative tolerance check
					
					// iterate until convergence
					for (size_t i = 0; i < _max_iter; ++i)
					{
						Ap 						= _matrixFreeMultiplication(p, _reg);
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

					LOGINFO("Conjugate gradient solver did not converge.", LOG_TYPES::WARNING, 3);
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}

                // #################################################################################################################################################

                // define the template specializations
                template arma::Col<double> conjugate_gradient(SOLVE_MATMUL_ARG_TYPES(double));
                template arma::Col<std::complex<double>> conjugate_gradient(SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
                // with preconditioner
                template arma::Col<double> conjugate_gradient(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
                template arma::Col<std::complex<double>> conjugate_gradient(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			};
			// #################################################################################################################################################
			namespace MINRES 
			{
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPES(_T1))
				{
					// !TODO: Implement the MINRES solver
					return General::CG::conjugate_gradient<_T1>(_matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg);
				}

				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, true))
				{
					// !TODO: Implement the MINRES solver with preconditioner
					return General::CG::conjugate_gradient<_T1>(_matrixFreeMultiplication, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}

				// #################################################################################################################################################

				// define the template specializations
				template arma::Col<double> minres(SOLVE_MATMUL_ARG_TYPES(double));
				template arma::Col<std::complex<double>> minres(SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
				// with preconditioner
				template arma::Col<double> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
				template arma::Col<std::complex<double>> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			};
			// #################################################################################################################################################
			namespace MINRES_QLP
			{
				enum class MINRES_QLP_FLAGS {
					PROCESSING 			= -2,   // Processing the MINRES_QLP solver.
					VALUE_BETA_ZERO 	= -1,   // Value: beta_k = 0. F and X are eigenvectors of (A - sigma*I).
					SOLUTION_X_ZERO 	= 0,    // Solution X = 0 was found as F = 0 (beta_km1 = 0).
					SOLUTION_RTOL 		= 1,    // Solution to (A - sigma*I)X = B found within given RTOL tolerance.
					SOLUTION_AR 		= 2,    // Solution to (A - sigma*I)X = B found with AR tolerance.
					SOLUTION_EPS 		= 3,    // Solution found within EPS tolerance (same as Case 1).
					SOLUTION_EPS_AR 	= 4,    // Solution found with EPS tolerance and AR (same as Case 2).
					SOLUTION_EIGEN 		= 5,    // X converged as an eigenvector of (A - sigma*I).
					MAXXNORM 			= 6,    // ||X|| exceeded MAXXNORM, solution may be diverging.
					ACOND 				= 7,    // ACOND exceeded ACONDLIM, system may be ill-conditioned.
					MAXITER 			= 8,    // MAXITER reached; no solution converged yet within given iterations.
					SINGULAR 			= 9     // System appears to be singular or badly scaled.
				};

				const std::vector<std::string> MINRES_QLP_MESSAGES = {
					"Processing the MINRES_QLP solver.",                                                 // Case -2
                    "Value: beta_k = 0. F and X are eigenvectors of (A - sigma*I).",                     // Case -1
                    "Solution X = 0 was found as F = 0 (beta_km1 = 0).",                                 // Case 0
                    "Solution to (A - sigma*I)X = B found within given RTOL tolerance.",                 // Case 1
                    "Solution to (A - sigma*I)X = B found with AR tolerance.",                           // Case 2
                    "Solution found within EPS tolerance (same as Case 1).",                             // Case 3
                    "Solution found with EPS tolerance and AR (same as Case 2).",                        // Case 4
                    "X converged as an eigenvector of (A - sigma*I).",                                   // Case 5
                    "||X|| exceeded MAXXNORM, solution may be diverging.",                               // Case 6
                    "ACOND exceeded ACONDLIM, system may be ill-conditioned.",                           // Case 7
                    "MAXITER reached; no solution converged yet within given iterations.",               // Case 8
                    "System appears to be singular or badly scaled.",  									 // Case 9
				};
                constexpr std::string convergence_message(MINRES_QLP_FLAGS _flag) { return MINRES_QLP_MESSAGES[(int)_flag + (int)MINRES_QLP_FLAGS::PROCESSING]; }
				constexpr double MAXXNORM 	= 1.0e+7;	// maximum norm of the solution
				constexpr double CONLIM [[maybe_unused]] 	= 1.0e+15;	// maximum condition number of the matrix
				constexpr double TRANSCOND 	= 1.0e+7;	// condition number for the transposed matrix
				constexpr double MINNORM 	= 1.0e-14;	// minimum norm of the solution

                // -----------------------------------------------------------------------------------------------------------------------------------------
                
                template <typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MATMUL_ARG_TYPES(_T1))
                {
					// !TODO: Implement the MINRES_QLP solver
					return General::CG::conjugate_gradient<_T1>(_matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
                
                // -----------------------------------------------------------------------------------------------------------------------------------------
				
                /*
				* @brief MINRES_QLP solver for the general case with a preconditioner.
				* The matrix S is symmetric and positive definite, so the MINRES_QLP method can be used.
				* The method is a preconditioned solver for approximating (A-\sigma I) * x = b, where A = S and b = F. 
				* Preconditioning makes it A = M^{-1/2} (A - \sigma I) M^{-1/2} x = M^{-1/2} b, where M is the preconditioner.
				* @ref S.-C. T. Choi and M. A. Saunders MINRES-QLP for Symmetric and Hermitian Linear Equations and Least-Squares Problems
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
				arma::Col<_T1> minres_qlp(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, true))
				{
					if (_preconditioner == nullptr)
						return Solvers::General::MINRES_QLP::minres_qlp<_T1>(_matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg);

					const size_t _n 	= _F.n_elem;										// number of elements			
					if (_max_iter == 0) _max_iter = _n;										// maximum number of iterations
                    
                    auto _flag0 = MINRES_QLP_FLAGS::PROCESSING, _flag = _flag0;				// flags for the convergence criterion

					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					// Lanczos' algorithm for the tridiagonalization of the matrix - vectors that are used in the algorithm
					arma::Col<_T1> z_km2 = arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector z0 - is r1
					arma::Col<_T1> z_km1 = _F; 												// initial vector z1 - is r2
					arma::Col<_T1> z_k; 													// vector z_{k+1} in the loop - updated at each iteration
					// preconditioned Lanczos' vectors
					arma::Col<_T1> q_k = _preconditioner->apply(z_km1);						// initial vector q1 - used for preconditioning - solves M * q_k = z_k
					arma::Col<_T1> p_k;														// vector p_{k+1} in the loop - updated at each iteration - later preconditioned - solves M * p_k = q_k
					_T1 _beta_km1 = 0.0, _beta_k = arma::cdot(_F, q_k);						// beta_k = F'*inv(M)*F - if preconditioner is identity beta_k isnorm of F
					_T1 _alpha_k = 0.0;														// diagonal element of the tridiagonal matrix - at each iteration
					// check if M is indefinite
					if (algebra::real(_beta_k) < 0.0) {
						LOGINFO("MINRES_QLP solver: Preconditioner is indefinite.", LOG_TYPES::ERROR, 3);
						throw std::runtime_error("MINRES_QLP solver: Preconditioner is indefinite.");
					} else {
						_beta_k = std::sqrt(_beta_k);										// beta_k = sqrt(F'*inv(M)*F)
					}
					_T1 _betaStart = _beta_k;												// initial value of beta_k
					// create phi and phi_km1
					_T1 _phi_k = _beta_k, _phi_km1 = _beta_k;								// use them as previous values of phi's
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					// Previous left reflection
					_T1 _delta_k = 0.0;														// delta_k = 0 - updated at each iteration
					_T1 _c_km1_1 = -1.0, _c_km1_2 [[maybe_unused]] = -1.0, _c_km1_3 [[maybe_unused]] = -1.0, _c_k_1, _c_k_2, _c_k_3;
					_T1 _s_km1_1 = 0.0, _s_km1_2 [[maybe_unused]] = 0.0, _s_km1_3 [[maybe_unused]] = 0.0, _s_k_1, _s_k_2, _s_k_3; 
					_T1 _gamma_k = 0.0, _gamma_km1 = 0.0, _gamma_km2 = 0.0, _gamma_km3 = 0.0, _gamma_min = 0.0, _gamma_min_km1, _gamma_min_km2;
					_T1 _eps_k[[maybe_unused]] = 0.0;									
					_T1 _tau_k = 0.0, _tau_km1 = 0.0, _tau_km2 = 0.0;						// use them as previous values of tau's
					_T1 _Ax_norm_k = 0.0, _Ax_norm_km1[[maybe_unused]] = 0.0;								// use them as previous values of Ax_norm's - norm of the matrix-vector multiplication
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					// Previous right reflection
					_T1 _theta_k = 0.0, _theta_km1 = 0.0, _theta_km2 = 0.0;					// use them as previous values of theta's
					_T1 _eta_k = 0.0, _eta_km1 = 0.0, _eta_km2 = 0.0;	
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					_T1 _xnorm_k = 0.0, _xnorm_km2 = 0.0;									// is xi in the algorithm - norm of the solution vector
					_T1 _mu_k = 0.0, _mu_km1 = 0.0, _mu_km2 = 0.0, _mu_km3 = 0.0, _mu_km4 = 0.0;	// use them as previous values of mu'
					_T1 _relres_km1 = 0.0, _relAres_km1 = 0.0;								// use them as previous values of relative residuals
					_T1 _rnorm = _beta_k, _rnorm_km1 = _beta_k, _rnorm_km2[[maybe_unused]] = _beta_k;		// use them as previous values of rnorm's
					_T1 _relres = _rnorm / (_beta_k + 1e-10), _relAres = 0.0;				// relative residual with a safety margin for beta_k = 0
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					// Regarding the wektor w and the solution vector x
					arma::Col<_T1> x_k	= arma::Col<_T1>(_n, arma::fill::zeros);			// final solution vector
					arma::Col<_T1> x_km1= (_x0) ? *_x0 : x_k;								// initial vector x_{k-2}
   					arma::Col<_T1> x_km2= (_x0) ? *_x0 : x_k;         			        	// Initial guess for x
					arma::Col<_T1> _w_k(_n, arma::fill::zeros), _w_km1(_n, arma::fill::zeros), _w_km2(_n, arma::fill::zeros); // use them as previous values of w's
					_T1 _Anorm = 0.0, _Anorm_km1 = 0.0, _Acond = 0.0, _Acond_km1 = 0.0;		// use them as previous values of A's norm and condition number
					_T1 _gammaqlp_k = 0.0, _gammaqlp_km1 = 0.0;
					_T1 _thetaqlp_k = 0.0;
					_T1 _muqlp_k = 0.0, _muqlp_km1 = 0.0;
					_T1 _root_k [[maybe_unused]]= 0.0, _root_km1 = 0.0;
					int _QLP_iter = 0;
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

					// iterate until convergence, if the value is k+1 in the algorithm, we update the corresponding variables.
					// Remember, here the index starts from 0 (k = -1 is the iteration before the loop)
					for (int k = 0; k < _max_iter; ++k)
					{
                        // if the flag has changed - break the loop
                        if (_flag != _flag0)
                            break;

						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// preconditioned Lanczos 
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						_T1 _beta_kp1 = 0.0, _pnorm_rho_k = 0.0;											// beta_{k+1} - at each iteration - to be updated later on
						{
							p_k				= _matrixFreeMultiplication(q_k, -1.0);							// apply the matrix-vector multiplication 
							if (_reg > 0)
								p_k 		-= _reg * q_k; 													// (here the regularization is applied - is sigma * I) 
							
							_alpha_k 		= arma::cdot(q_k, p_k) / _beta_k / _beta_k;						// alpha_k = (q_k', p_k) / beta_k^2 - at each iteration - for Hermintan should be real
							z_k 		 	= p_k / _beta_k - (_alpha_k / _beta_k) * z_km1;					// z_{k+1} = p_k / beta_k - (alpha_k / beta_k) * z_k - updated at each iteration - new orthogonal vector
							if (k > 0)																		// the km2 element exists
								z_k 		-= (_beta_k / _beta_km1) * z_km2;								// z_{k+1} = p_k / beta_k - (alpha_k / beta_k) * z_k - (beta_{k-1} / beta_k) * z_{k-1} - updated at each iteration
							q_k 				= _preconditioner->apply(z_k);								// apply the preconditioner - can be updated now as is not used anymore - otherwise is same as z_k
							_beta_kp1 		= arma::cdot(q_k, z_k);											// beta_{k+1} = q_k' * z_k - at each iteration
							if (algebra::real(_beta_kp1) > 0) {
								_beta_kp1 	= std::sqrt(_beta_kp1);											// beta_{k+1} = sqrt(q_k, z_k) - defined here, beta needs to be updated at each iteration
							} else {
								LOGINFO("MINRES_QLP solver: Preconditioner is indefinite.", LOG_TYPES::ERROR, 3);
								throw std::runtime_error("MINRES_QLP solver: Preconditioner is indefinite.");
							}	
							_pnorm_rho_k = (k > 0) ? norm(_beta_k, _alpha_k, _beta_kp1) : norm(_alpha_k, _beta_kp1); // ||[βk αk βk+1]|| - at each iteration - beta is real
#ifdef _DEBUG
							LOGINFO("MINRES_QLP solver (Lanczos iteration): Iteration " + std::to_string(k) + " - ||[βk αk βk+1]|| = " + STRS(_pnorm_rho_k), LOG_TYPES::DEBUG, 3);
							LOGINFO("MINRES_QLP solver (Lanczos iteration): Iteration " + std::to_string(k) + " - beta_k = " + STRS(_beta_k) + ", alpha_k = " + STRS(_alpha_k) + ", beta_kp1 = " + STRS(_beta_kp1), LOG_TYPES::DEBUG, 3);
#endif
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// apply the previous reflection
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

						// update delta_k ((2) second stage) 
						_T1 _eps_k_p1, _delta_kp1;
						{
							_T1 _dbar 			= _delta_k; 												// [to zero out δ(2)k]
							_delta_k			= (_c_km1_1 * _dbar) + (_s_km1_1 * _alpha_k); 				
							_gamma_k 			= (_s_km1_1 * _dbar) - (_c_km1_1 * _alpha_k);

							//! new value of eta_k - update later
							_eps_k_p1		= _s_km1_1 * _beta_kp1; 										// [produces first two entries in Tk+1ek+1] - local variable
							//! new value of delta_k - update later
							_delta_kp1 		= -_c_km1_1 * _beta_kp1; 										// update delta after the reflection
#ifdef _DEBUG 
							LOGINFO("MINRES_QLP solver (Previous left reflection): Iteration " + STR(k) + " - delta_k = " + STRS(_delta_k) + ", gamma_k = " + STRS(_gamma_k), LOG_TYPES::DEBUG, 3);
							LOGINFO("MINRES_QLP solver (Previous left reflection): Iteration " + STR(k) + " - delta_kp1 = " + STRS(_delta_kp1) + ", eps_k_p1 = " + STRS(_eps_k_p1), LOG_TYPES::DEBUG, 3);
#endif
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// apply the new reflection Q_k
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						{
							// use symortho to get the new reflection 
							// update _gamma_k ((2) second stage)
							algebra::Solvers::sym_ortho(_gamma_k, _beta_kp1, _c_k_1, _s_k_1, _gamma_k); 						// [current left reflection]
							_tau_k 				= _c_k_1 * _phi_km1; 										// [Last element of tk]
							_phi_k 				= _s_k_1 * _phi_km1; 										// 
							_Ax_norm_k			= norm(_Ax_norm_k, _tau_k); 								// [Update ‖Axk‖]
#ifdef _DEBUG
							LOGINFO("MINRES_QLP solver (New left reflection): Iteration " + STR(k) + " - gamma_k = " + STRS(_gamma_k) + ", tau_k = " + STRS(_tau_k), LOG_TYPES::DEBUG, 3);
							LOGINFO("MINRES_QLP solver (New left reflection): Iteration " + STR(k) + " - phi_k = " + STRS(_phi_k) + ", Ax_norm_k = " + STRS(_Ax_norm_k), LOG_TYPES::DEBUG, 3);
#endif
						}

						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// apply the previous right reflection P{k-2,k}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						_T1 _delta_k_tmp[[maybe_unused]];
						if (k > 1)
						{
							// update delta_k ((3) third stage)
							_delta_k_tmp 		= (_s_k_2 * _theta_km1) - (_c_k_2 * _delta_k);
							// update _theta_km1 ((2) second stage)
							_theta_km1 			= (_c_k_2 * _theta_km1) + (_s_k_2 * _delta_k); 				// use delta from (2) stage 
							// update _delta_k ((3) third stage)
							_delta_k 			= _delta_k_tmp;
							// calculate the new value of eta_k
							_eta_k 				= _s_k_2 * _gamma_k; 										// use gamma from (2) stage
							// update _gamma_k ((3) third stage)
							_gamma_k 			= -_c_k_2 * _gamma_k;
#ifdef _DEBUG
							LOGINFO("MINRES_QLP solver (Previous right reflection): Iteration " + STR(k) + " - delta_k = " + STRS(_delta_k) + ", theta_km1 = " + STRS(_theta_km1), LOG_TYPES::DEBUG, 3);
							LOGINFO("MINRES_QLP solver (Previous right reflection): Iteration " + STR(k) + " - eta_k = " + STRS(_eta_k) + ", gamma_k = " + STRS(_gamma_k), LOG_TYPES::DEBUG, 3);
#endif
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// Compute the new reflection P{k-1,k}, P_12, P_23, P_34...
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 0)
						{
							sym_ortho(_gamma_km1, _delta_k, _c_k_3, _s_k_3, _gamma_km1);					// [Second right reflection]
							// calculate theta -
							_theta_k 			= _s_k_3 * _gamma_k;
							// update _gamma_k ((4) fourth stage)
							_gamma_k 			= -_c_k_3 * _gamma_k;
#ifdef _DEBUG
							LOGINFO("MINRES_QLP solver (New right reflection): Iteration " + std::to_string(k) + " - theta_k = " + STRS(_theta_k) + ", gamma_k = " + STRS(_gamma_k), LOG_TYPES::DEBUG, 3);
#endif
						}			
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// update the xnorm
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						{
							if (k > 1) 
								_mu_km2 = (_tau_km2 - _eta_km2 * _mu_km4 - _theta_km2 * _mu_km3) / _gamma_km2;	// [Update µk−2]
							if (k > 0)
								_mu_km1 = (_tau_km1 - _eta_km1 * _mu_km3 - _theta_km1 * _mu_km2) / _gamma_km1;	// [Update µk−1]

							_T1 _xnorm_tmp 	= norm(_xnorm_km2, _mu_km2, _mu_km1);								// [Update ‖xk−2‖]
							bool _likeLS 	= (algebra::real(_relres_km1) >= algebra::real(_relAres_km1));		// [Check for like least-squares problem]

							if (std::abs(_gamma_k) > MINNORM && algebra::gr(_xnorm_tmp, MAXXNORM))
							{
								_mu_k 		= (_tau_k - _eta_k * _mu_km2 - _theta_k * _mu_km1) / _gamma_k;		// [Update µk]
								if (norm(_xnorm_tmp, _mu_k) > MAXXNORM && !_likeLS)
									_flag = MINRES_QLP_FLAGS::MAXXNORM;											// [X converged as an eigenvector of (A - sigma*I)]
							}
							else {
								_mu_k 		= 0.0;
								_flag 		= MINRES_QLP_FLAGS::SINGULAR;										// [System appears to be singular or badly scaled]
							}
							// update the xnorm
							_xnorm_km2 		= norm(_xnorm_tmp, _mu_km2);										// update xi_km2 - norm of the solution vector
							_xnorm_k 		= norm(_xnorm_km2, _mu_km1, _mu_k);									// update xi_k - norm of the solution vector
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// Update w. Update x except if it is too big.
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						{
							if (algebra::real(_Acond) < TRANSCOND && _flag != _flag0 && _QLP_iter == 0)			// MINRES updates
							{
								_w_k 		= -(_c_k_2 / _beta_k) * q_k + _w_km2 * _s_k_2;						// [Update wk]
								if (algebra::real(_xnorm_k) < MAXXNORM)
									x_k 	+= _tau_k * _w_k;													// [Update xk]
								else 
									_flag 	= MINRES_QLP_FLAGS::MAXXNORM;										// [X converged as an eigenvector of (A - sigma*I)]

							}
							else 																				// MINRES-QLP updates
							{
								_QLP_iter++;
								if (_QLP_iter == 1) {
									if (k > 0) {																// construct w_km3, w_km2, w_km1
										if (k > 3) 																// w_km3 exists
											_w_km2 = _gamma_km3 * _w_km2 + _theta_km2 * _w_km1 + _eta_km1 * _w_k;
										if (k > 2) 																// w_km2 exists
											_w_km1 = _gammaqlp_km1 * _w_km1 + _thetaqlp_k * _w_k;
										_w_k = _gammaqlp_k * _w_k;
										x_km2 = x_k - _w_km1 * _muqlp_km1 - _w_k * _muqlp_k;
									}
								}
								if (k == 0)
								{
									_w_km2 = q_k * (_s_k_2 / _beta_k);
									_w_k = q_k * (-_c_k_2 / _beta_k);
								}
								else if (k == 1)
								{
									_w_km1 = _w_k;
									_w_k = q_k * (-_c_k_2 / _beta_k) + _w_km2 * (_s_k_2 / _beta_k);
								}
								else
								{
									_w_km2 = _w_km1;
									_w_km1 = _w_k;
									_w_k = q_k * (-_c_k_2 / _beta_k) + _w_km2 * (_s_k_2 / _beta_k);
								}
								// update the solution
								x_km2 = x_km2 + _w_km2 * _mu_km2;
								x_k = x_km2 + _w_km1 * _mu_km1 + _w_k * _mu_k;
							}
							#ifdef _DEBUG
							#endif
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// Compute the next right reflection P{k+1,k+1}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						_T1 _gamma_tmp = _gamma_km1;
						{
							sym_ortho(_gamma_km1, _eps_k_p1, _c_k_2, _s_k_2, _gamma_km1);		
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// Store quantities for the next iteration
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						{
							_gammaqlp_km1 = _gamma_tmp;
							_thetaqlp_k = _theta_k;
							_gammaqlp_k = _gamma_k;
							_muqlp_km1 = _mu_km1;
							_muqlp_k = _mu_k;
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// Estimate various norms
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						{
							auto _abs_gamma = std::abs(_gamma_k);
							_Anorm_km1 = _Anorm;
							_Anorm = algebra::max(_Anorm_km1, _gamma_km1, _abs_gamma, _pnorm_rho_k);
							if (k == 0)
							{
								_gamma_min = _gamma_k;
								_gamma_min_km1 = _gamma_k;
							}
							else 
							{
								_gamma_min_km2 = _gamma_min_km1;
								_gamma_min_km1 = _gamma_min;
								_gamma_min = algebra::min(_gamma_min_km2, _gamma_min_km1, _abs_gamma);
							}
							_Acond_km1 	= _Acond;
							_Acond 		= _Anorm / _gamma_min;
							_rnorm_km1 	= _rnorm;
							_relres_km1 = _relres;
							if (_flag != MINRES_QLP_FLAGS::SINGULAR)
								_rnorm 	= _phi_k;
							_relres 	= _rnorm / (_Anorm * _xnorm_k + _beta_k + 1e-10);
							_root_km1   = algebra::norm(_delta_k, _delta_kp1);
							_Anorm_km1  = _rnorm_km1 * _root_km1;
							_relAres_km1= _root_km1 / _Anorm;
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// Check for convergence
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
						{
							_T1 _epsx = _Anorm * _xnorm_k * _eps; 											// [Estimate of ‖(A−σI)xk−b‖]
							if (_flag == _flag0 || _flag == MINRES_QLP_FLAGS::SINGULAR)
							{
								_T1 t1 = 1.0 + _relres;
								_T1 t2 = 1.0 + _relAres;
								if (k >= _max_iter - 1) 					_flag = MINRES_QLP_FLAGS::MAXITER;
								if (algebra::gr(_Acond, TRANSCOND)) 		_flag = MINRES_QLP_FLAGS::ACOND;
								if (algebra::geq(_xnorm_k, MAXXNORM)) 		_flag = MINRES_QLP_FLAGS::MAXXNORM;
								if (algebra::geq(_epsx, _betaStart))		_flag = MINRES_QLP_FLAGS::SOLUTION_EIGEN;
								if (algebra::leq(t2, 1.0)) 					_flag = MINRES_QLP_FLAGS::SOLUTION_EPS_AR;
								if (algebra::leq(t1, 1.0)) 					_flag = MINRES_QLP_FLAGS::SOLUTION_EPS;
								if (algebra::leq(_relAres, _eps)) 			_flag = MINRES_QLP_FLAGS::SOLUTION_AR;
								if (algebra::leq(_relres, _eps)) 			_flag = MINRES_QLP_FLAGS::SOLUTION_RTOL;
							#ifdef _DEBUG
							LOGINFO("MINRES_QLP solver (Check for convergence): Iteration " + STR(k) + " - epsx = " + STRS(_epsx) + ", betaStart = " + STRS(_betaStart), LOG_TYPES::DEBUG, 3);
							#endif
							}
						}
					}

					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					// Check for convergence
					// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					auto _msg = convergence_message(_flag);
					LOGINFO("MINRES_QLP solver: " + _msg, LOG_TYPES::ERROR, 3);
					if (_converged != nullptr)
						*_converged = _flag != MINRES_QLP_FLAGS::MAXITER && _flag != MINRES_QLP_FLAGS::MAXXNORM && _flag != MINRES_QLP_FLAGS::SINGULAR;
					return x_k;
				}
                
                // -----------------------------------------------------------------------------------------------------------------------------------------

                // define the template specializations
                // double
                template arma::Col<double> minres_qlp(SOLVE_MATMUL_ARG_TYPES(double));
                template arma::Col<double> minres_qlp(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
                // complex double
                template arma::Col<std::complex<double>> minres_qlp(SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
                template arma::Col<std::complex<double>> minres_qlp(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));

                // -----------------------------------------------------------------------------------------------------------------------------------------
			};
			// #################################################################################################################################################
            
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