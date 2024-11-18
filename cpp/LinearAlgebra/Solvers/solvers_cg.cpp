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
			
				// #################################################################################################################################################

			};
			// FOR THE CLASS
			namespace CG
			{
				// ----------------------------------------------------------------------------------------------------------------------------------------

				template <typename _T, bool _symmetric>
				void ConjugateGradient_s<_T, _symmetric>::init(const arma::Col<_T>& _F, arma::Col<_T>* _x0)
				{
					this->converged_ = false;
					if (this->N_ != _F.n_elem)
						this->N_ = _F.n_elem;

					this->x_ 		= (_x0 == nullptr) ? arma::Col<_T>(_F.n_elem, arma::fill::zeros) : *_x0;
					this->r 		= _F;
					if (_x0 != nullptr)
						this->r -= this->matVecFun_(this->x_, this->reg_);
					
					// check the preconditioner
					if (this->precond_ != nullptr)
					{
						this->z 		= this->precond_->apply(this->r);
						this->p 		= this->z;
						this->rs_old 	= arma::cdot(this->r, this->z);
					}
					else
					{
						this->rs_old 	= arma::cdot(this->r, this->r);
						this->p 		= this->r;
					}
					
					// check the convergence
					if (std::abs(this->rs_old) < this->eps_)
						this->converged_ = true;
				}

				// ----------------------------------------------------------------------------------------------------------------------------------------

				template <typename _T, bool _symmetric>
				void ConjugateGradient_s<_T, _symmetric>::solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
				{
					if (!this->matVecFun_)
						throw std::runtime_error("Conjugate gradient solver: matrix-vector multiplication function is not set.");
					
					if (_precond != nullptr)
					{
						this->precond_ 			= _precond;
						this->isPreconditioned_ = true;
					}

					// initialize the solver
					this->init(_F, _x0);

					// check the convergence
					if (this->converged_)
						return;

					// iterate until convergence
					for (size_t i = 0; i < this->max_iter_; ++i)
					{
						this->Ap 				= this->matVecFun_(this->p, this->reg_);
						_T alpha 				= this->rs_old / arma::cdot(this->p, this->Ap);
						this->x_ 				+= alpha * this->p;
						this->r 				-= alpha * this->Ap;
						_T rs_new 				= arma::cdot(r, r);
						// Check for convergence
						if (std::abs(rs_new) < this->eps_) {
							this->converged_ = true;
							return;
						}

						if (this->precond_ != nullptr)
						{
							this->z 				= this->precond_->apply(this->r);
							rs_new 					= arma::cdot(this->r, this->z);
							this->p 				= this->z + (rs_new / this->rs_old) * this->p;
						}
						else
							this->p 				= this->r + (rs_new / this->rs_old) * this->p;
						this->rs_old 			= rs_new;
					}
					// check the convergence
					LOGINFO("Conjugate gradient solver did not converge.", LOG_TYPES::WARNING, 3);
					this->converged_ = false;
				}

				// ############################################################################################################################################

				// define the template specializations
				template class ConjugateGradient_s<double, true>;
				template class ConjugateGradient_s<std::complex<double>, true>;
				template class ConjugateGradient_s<double, false>;
				template class ConjugateGradient_s<std::complex<double>, false>;
			};
        };
    };
};