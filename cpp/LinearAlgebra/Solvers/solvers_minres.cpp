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

				/*
					# Preconditioned MINRES Method

					The **Preconditioned Minimal Residual Method (MINRES)** is used to solve a system of linear equations involving a self-adjoint operator, modified by preconditioning matrices.

					---

					## Given System

					The original system to solve is:

					\[
					A x = b,
					\]

					where \( A \) is a self-adjoint operator. The goal is to find the solution \( x \) that satisfies this equation.

					To apply preconditioning, introduce **left** and **right preconditioners** \( M_l \) and \( M_r \), and define \( y \) such that:

					\[
					x = M_r y.
					\]

					The system is reformulated as:

					\[
					M M_l A M_r y = M M_l b,
					\]

					where \( M \) is an additional preconditioner applied to the residual to improve convergence.

					---

					## Self-Adjoint Operator in Preconditioned Space

					The operator \( M M_l A M_r \) is self-adjoint with respect to the **inner product**:

					\[
					\langle u, v \rangle_{M^{-1}} := \langle M^{-1} u, v \rangle.
					\]

					The corresponding norm is:

					\[
					\| u \|_{M^{-1}} = \sqrt{\langle u, u \rangle_{M^{-1}}}.
					\]

					---

					## Optimization Problem Solved by MINRES

					The **preconditioned MINRES method** computes iterates \( x_k \) such that:

					\[
					\| M M_l (b - A x_k) \|_{M^{-1}} =
					\min_{z \in x_0 + M_r K_k} \| M M_l (b - A z) \|_{M^{-1}},
					\]

					where \( K_k := K_k(M M_l A M_r, r_0) \) is the Krylov subspace spanned by the first \( k \) Lanczos vectors.

					The Lanczos algorithm is applied to the operator \( M M_l A M_r \) with the inner product:

					\[
					\langle u, v \rangle_{M^{-1}} := \langle M^{-1} u, v \rangle.
					\]

					The initial residual is given by:

					\[
					r_0 = M M_l (b - A x_0).
					\]

					---
				*/

				// #################################################################################################################################################
				
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
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPES(_T1)) {
					// Initialize solution x, setting it to zero if _x0 is nullptr (no initial guess)
					arma::Col<_T1> x 		= (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					arma::Col<_T1> r 		= _F;
					if (_x0 != nullptr) r 	-= _matrixFreeMultiplication(x, _reg);							// Initial residual r = b - A*x

					_T1 _beta0 				= arma::norm(r);												// is the norm of the residual
					if (std::abs(_beta0) < _eps) {															// is an eigenvector
						if (_converged != nullptr) *_converged = true;
						return _F;
					}

					arma::Col<_T1> pkm1 	= r / _beta0, pk = pkm1, pkp1;									// is p0, p1, p2 in the MINRES algorithm
					arma::Col<_T1> Ap_km1 	= _matrixFreeMultiplication(pkm1, _reg);						// is s0 in the MINRES algorithm
					arma::Col<_T1> Ap_k = Ap_km1, Ap_kp1;													// matrix-vector multiplication result - is the same as s_1, s_2 in the MINRES algorithm

					for (size_t i = 0; i < _max_iter; ++i)													// iterate until convergence
					{
						// update the search direction
						pkp1 = pk; pk = pkm1;																
						Ap_kp1 = Ap_k; Ap_k = Ap_km1;														// update the matrix-vector multiplication result
						_T1 alpha 			= arma::cdot(r, Ap_k) / arma::cdot(Ap_k, Ap_k);					// is the overlap of r and Ap so that x can be updated with the correct step, previous vector
						x 					+= alpha * pk;													// update the solution
						r 					-= alpha * Ap_k;												// update the residual
						_T1 beta 			= arma::norm(r);												// is the norm of the residual
						if (std::abs(beta) < _eps) {
							if (_converged != nullptr)														// Check for convergence
								*_converged = true;
							return x;
						}

						// update the search direction
						pkm1 				= Ap_k;															// update the search direction - p_{k-1} = Ap_k
						Ap_km1 				= _matrixFreeMultiplication(Ap_k, _reg);						// update the matrix-vector multiplication result
						_T1 beta1 			= arma::cdot(Ap_km1, Ap_k) / arma::cdot(Ap_k, Ap_k);			// is the overlap of Ap_km1 and Ap_k
						// update the search direction													
						pkm1 				-= beta1 * pk;
						Ap_km1 				-= beta1 * Ap_k;

						if (i > 0)																			// Update the second Lanczos vector
						{
							_T1 beta2 		= arma::cdot(Ap_km1, Ap_kp1) / arma::cdot(Ap_kp1, Ap_kp1);
							pkm1 			-= beta2 * pkp1;
							Ap_km1 			-= beta2 * Ap_kp1;
						}
					}
					
					LOGINFO("MINRES solver did not converge.", LOG_TYPES::WARNING, 3);
					if (_converged != nullptr)
						*_converged = false;
					return x;
				}
				// #################################################################################################################################################

				/**
				* @brief MINRES solver for the general case with a preconditioner.
				* @param _F The right-hand side vector.
				* @param _x0 The initial guess for the solution.
				* @param _preconditioner The preconditioner for the MINRES method.       M M_l A M_r y = M M_l b. We will use 
				*	The preconditioned MINRES method then computes (in exact arithmetics!)
				*	iterates :math:`x_k \in x_0 + M_r K_k` with
				*	:math:`K_k:= K_k(M M_l A M_r, r_0)` such that
					.. math::

					\|M M_l(b - A x_k)\|_{M^{-1}} =
					\min_{z \in x_0 + M_r K_k} \|M M_l (b - A z)\|_{M^{-1}}.

					The Lanczos alorithm is used with the operator
					:math:`M M_l A M_r` and the inner product defined by
					:math:`\langle x,y \rangle_{M^{-1}} = \langle M^{-1}x,y \rangle`.
					The initial vector for Lanczos is
					:math:`r_0 = M M_l (b - Ax_0)` - note that :math:`M_r` is not used for
					the initial vector.
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
					if (_preconditioner == nullptr || true) {
						return minres<_T1>(_matrixFreeMultiplication, _F, _x0, _eps, _max_iter, _converged, _reg);
					}
					// !TODO Implement the MINRES method for symmetric positive definite matrices A (not singular) when preconditioner is provided
					const size_t _n = _F.n_elem;														// number of elements
					if (_max_iter == 0) _max_iter = _n;													// maximum number of iterations

					arma::Mat<_T1> W = arma::Mat<_T1> (_n, 2, arma::fill::zeros);						// vectors for the residuals - updating y
					arma::Col<_T1> yk = _x0 == nullptr ? arma::Col<_T1>(_n, arma::fill::zeros) : *_x0;	// initial guess for the solution
					arma::Col<_T1> y = _F - _matrixFreeMultiplication(yk, _reg);						// initial residual r = b - A*x
					arma::Col<_T1> r = y;																// initial residual r = b - A*x

					// for the givens rotation
					// _T1 ck = -1.0, sk = 0.0, rho = 0.0;
					// _T1 ckm1 = -1.0, skm1 = 0.0, rhom1 = 0.0;

					for (int k = 0; k < _max_iter; ++k)
					{

					}
					return yk;
				}

				// ############################################################################################################################################

				// define the template specializations
				template arma::Col<double> minres(SOLVE_MATMUL_ARG_TYPES(double));
				template arma::Col<std::complex<double>> minres(SOLVE_MATMUL_ARG_TYPES(std::complex<double>));
				// with preconditioner
				template arma::Col<double> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(double, true));
				template arma::Col<std::complex<double>> minres(SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(std::complex<double>, true));
			};

			// #################################################################################################################################################

			namespace MINRES
			{
				// ############################################################################################################################################
			
				template <typename _T1, bool _symmetric>
				void MINRES_s<_T1, _symmetric>::init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0)
				{
					this->converged_ = false;
					if (this->N_ != _F.n_elem)
						this->N_ = _F.n_elem;

					this->x_ = (_x0 == nullptr) ? arma::Col<_T1>(_F.n_elem, arma::fill::zeros) : *_x0;
					this->r  = _F;
					if (_x0 != nullptr) this->r -= this->matVecFun_(this->x_, this->reg_);

					// calculate the norms
					this->beta0_ 	=	arma::norm(this->r);
					if (std::abs(this->beta0_) < this->eps_) {
						this->converged_ = true;
						return;
					}

					// initialize the multiplications 
					this->pkm1 		= this->r / this->beta0_;
					this->pk 		= this->pkm1;
					this->Ap_km1    = this->matVecFun_(this->pkm1, this->reg_);
					this->Ap_k 		= this->Ap_km1;
				}

				// ############################################################################################################################################

				template <typename _T1, bool _symmetric>
				void MINRES_s<_T1, _symmetric>::solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0, Precond<_T1, _symmetric>* _preconditioner)
				{
					if (!this->matVecFun_)
						throw std::runtime_error("MINRES solver: matrix-vector multiplication function is not set.");

					if (_preconditioner != nullptr) {
						this->precond_ = _preconditioner;
						this->isPreconditioned_ = true;
					}

					// Initialize solution x, setting it to zero if _x0 is nullptr (no initial guess)
					this->init(_F, _x0);

					// check the convergence
					if (this->converged_)
						return;

					// GO!
					for (size_t i = 0; i < this->max_iter_; ++i)													// iterate until convergence
					{
						// update the search direction
						this->pkp1 = this->pk; this->pk = this->pkm1;																
						this->Ap_kp1 = this->Ap_k; this->Ap_k = this->Ap_km1;										// update the matrix-vector multiplication result
						_T1 alpha 			= arma::cdot(this->r, this->Ap_k) / arma::cdot(this->Ap_k, this->Ap_k);	// is the overlap of r and Ap so that x can be updated with the correct step, previous vector
						this->x_			+= alpha * this->pk;													// update the solution
						this->r 			-= alpha * this->Ap_k;													// update the residual
						_T1 beta 			= arma::norm(this->r);													// is the norm of the residual
						if (std::abs(beta) < this->eps_) {
							this->converged_ = true;
							return;
						}

						// update the search direction
						this->pkm1 			= this->Ap_k;															// update the search direction - p_{k-1} = Ap_k
						this->Ap_km1 		= this->matVecFun_(this->Ap_k, this->reg_);								// update the matrix-vector multiplication result
						_T1 beta1 			= arma::cdot(this->Ap_km1, this->Ap_k) / arma::cdot(this->Ap_k, this->Ap_k); // is the overlap of Ap_km1 and Ap_k
						// update the search direction													
						this->pkm1 			-= beta1 * this->pk;
						this->Ap_km1 		-= beta1 * this->Ap_k;

						if (i > 0)																					// Update the second Lanczos vector
						{
							_T1 beta2 		= arma::cdot(this->Ap_km1, this->Ap_kp1) / arma::cdot(this->Ap_kp1, this->Ap_kp1);
							this->pkm1 		-= beta2 * this->pkp1;
							this->Ap_km1 	-= beta2 * this->Ap_kp1;
						}
					}
					
					LOGINFO("MINRES solver did not converge.", LOG_TYPES::WARNING, 3);
					this->converged_ = false;
				}
				// ############################################################################################################################################
			
				// define the template specializations
				template class MINRES_s<double, true>;
				template class MINRES_s<std::complex<double>, true>;
				template class MINRES_s<double, false>;
				template class MINRES_s<std::complex<double>, false>;
			};
        };
    };
};