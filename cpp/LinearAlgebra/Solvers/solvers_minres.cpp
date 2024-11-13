#include "../../../src/lin_alg.h"
#include "../../../src/flog.h"

// #################################################################################################################################################

namespace algebra 
{
	namespace Solvers 
    {
		namespace General 
        {
			namespace MINRES 
			{
				/**
				* @brief MINRES solver for the general case without a preconditioner.
				* @param _matrixFreeMultiplication The matrix-vector multiplication function. It is used to calculate the matrix-vector product Sx.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _eps The convergence criterion.
				* @param _max_iter The maximum number of iterations.
				* @param _converged The flag indicating if the solver converged.
				* @param _reg The regularization parameter. (A + \lambda I) x \approx b
				* @return The solution vector x.
				* @theory The MINRES algorithm is used to solve the system Sx = F or the minimization problem ||Sx - F||_2 where S is a symmetric matrix. 
				* The algorithm is based on the Lanczos algorithm and is used to solve symmetric indefinite systems. 
				*/
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPES(_T1))
				{
					// Initialize solution x, setting it to zero if _x0 is nullptr (no initial guess)
					arma::Col<_T1> x 		= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 		= _F;
					if (_x0 != nullptr) r 	-= _matrixFreeMultiplication(x, _reg);							// Initial residual r = b - A*x
					arma::Col<_T1> pkm1 	= r, pk = pkm1, pkp1;											// is A^0 * r
					arma::Col<_T1> Ap_km1 	= _matrixFreeMultiplication(pkm1, _reg), Ap_k = Ap_km1, Ap_kp1;	// matrix-vector multiplication result - is the same as s_0 in the MINRES algorithm
					_T1 _rnorm 				= arma::norm(r);												// the norm of the residual - is the same as beta_0 in the MINRES algorithm

					for (size_t i = 0; i < _max_iter; ++i)													// iterate until convergence
					{
						// update the search direction
						pkp1 = pk; pk = pkm1;																// update search directions using the Lanczos coefficients
						Ap_kp1 = Ap_k; Ap_k = Ap_km1;														// update the matrix-vector multiplication result
						_T1 alpha 			= arma::cdot(r, Ap_k) / arma::cdot(Ap_k, Ap_k);					// is the overlap of r and Ap so that x can be updated with the correct step
						x 					+= alpha * pk;													// update the solution
						r 					-= alpha * Ap_k;												// update the residual
						_T1 beta 			= arma::norm(r) / _rnorm;										// is the norm of the residual
						if (std::abs(beta) < _eps) {
							if (_converged != nullptr)														// Check for convergence
								*_converged = true;
							return x;
						}
						// update the search direction
						pkm1 				= Ap_k;															// update the search direction - p_{k-1} = Ap_k
						Ap_km1 				= _matrixFreeMultiplication(Ap_k, _reg);						// update the matrix-vector multiplication result
						beta 				= arma::cdot(Ap_km1, Ap_k) / arma::cdot(Ap_k, Ap_k);			// is the overlap of Ap_km1 and Ap_k
						// update the search direction													
						pkm1 				-= beta * pk;
						Ap_km1 				-= beta * Ap_k;

						if (i > 0)																			// Update the second Lanczos vector
						{
							beta = arma::cdot(Ap_km1, Ap_kp1) / arma::cdot(Ap_kp1, Ap_kp1);
							pkm1 			-= beta * pkp1;
							Ap_km1 			-= beta * Ap_kp1;
						}
					}
					LOGINFO("MINRES solver did not converge.", LOG_TYPES::WARNING, 3);
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}

				/*
				* @brief MINRES solver for the general case with a preconditioner.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _preconditioner The preconditioner for the MINRES method.
				* @param _eps The convergence criterion.
				* @param _max_iter The maximum number of iterations.
				* @param _converged The flag indicating if the solver converged.
				* @param _reg The regularization parameter. (A + \lambda I) x \approx b
				* @return The solution vector x.
				* @theory The MINRES algorithm is used to solve the system Sx = F or the minimization problem ||Sx - F||_2 where S is a symmetric matrix.
				* The algorithm is based on the Lanczos algorithm and is used to solve symmetric indefinite systems.
				* @note When a preconditioner is provided, the MINRES algorithm is used to solve the system M^{-1}Sx = M^{-1}F or the minimization problem ||M^{-1}Sx - M^{-1}F||_2 where M is the preconditioner.
				*/
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, true))
				{
					if (_preconditioner == nullptr)
						return Solvers::General::MINRES::minres<_T1>(_matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg);
   		 			// Initialize solution x, setting it to zero if _x0 is nullptr (no initial guess)
					arma::Col<_T1> x 		= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 		= _F;
					if (_x0 != nullptr) r 	-= _matrixFreeMultiplication(x, _reg);							// Initial residual r = b - A*x

					_T1 beta_old 			= 0.0;
					_T1 alpha 				= 0.0;

					arma::Col<_T1> z 		= _preconditioner->apply(r);
					arma::Col<_T1> pkm1 	= z, pk = pkm1;													// Variables for Krylov subspace basis and matrix-vector product results
					arma::Col<_T1> Ap_kp1;																	// matrix-vector multiplication result - is the same as s_0 in the MINRES algorithm
					_T1 _rnorm 				= arma::norm(r);												// the norm of the residual - is the same as beta_0 in the MINRES algorithm

					for (size_t i = 0; i < _max_iter; ++i)													// iterate until convergence
					{
						// Apply the matrix vector multiplication
						Ap_kp1         		= _matrixFreeMultiplication(pk, _reg);							// is A^k * p_{k}
						
						// compute alpha and update the residual
						alpha 				= arma::cdot(pk, Ap_kp1);										// is the overlap of r and Ap so that x can be updated with the correct step
						Ap_kp1 				-= alpha * pk + beta_old * pkm1;								

						// update solution
						_T1 beta 			= arma::norm(Ap_kp1);											// is the norm of the residual
						if (std::abs(beta) < _eps) {
							if (_converged != nullptr)														// Check for convergence
								*_converged = true;
							return x;
						}

						_T1 omega			= _rnorm / alpha;
						x 					+= omega * pk;													// update the solution
						_rnorm         		= beta * omega;													// update the residual norm

						// prepare for the next iteration
						pkm1 				= pk; 
						pk 					= Ap_kp1 / beta;
						beta_old 			= beta;
					}

					LOGINFO("MINRES solver did not converge.", LOG_TYPES::WARNING, 3);
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}

				// #################################################################################################################################################

				// define the template specializations
				template arma::Col<double> minres(SOLVE_MATMUL_ARG_TYPES(double));
				template arma::Col<std::complex<double>> minres(SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
				// with preconditioner
				template arma::Col<double> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
				template arma::Col<std::complex<double>> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			};
        };
    };
};