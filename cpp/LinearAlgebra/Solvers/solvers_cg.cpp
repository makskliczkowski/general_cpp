#include "../../../src/lin_alg.h"
#include "../../../src/flog.h"

// #################################################################################################################################################

namespace algebra 
{
	namespace Solvers 
    {
		namespace General 
        {
			namespace CG 
            {
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
        };
    };
};