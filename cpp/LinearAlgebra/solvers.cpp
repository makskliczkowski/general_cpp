#include "../../src/lin_alg.h"
#include <cassert>

// #################################################################################################################################################

// PRECONDITIONERS FOR THE SOLVERS

// #################################################################################################################################################

namespace algebra
{
    namespace Solvers
    {
        namespace Preconditioners
        {
            // #################################################################################################################################################
            
            // define the template specializations 
            // double, false
            template class Preconditioner<double, false>;
            // double, true
            template class Preconditioner<double, true>;
            // complex, false
            template class Preconditioner<std::complex<double>, false>;
            // complex, true
            template class Preconditioner<std::complex<double>, true>;
        };
    };
};

// #################################################################################################################################################

// FOR THE FISHER MATRIX SOLVERS - SYMMETRIC MATRICES WITH FORM S = A^T * A, given A

// #################################################################################################################################################

/**
* @brief Check the sign of a value
* @param val value to be checked
* @return sign of a variable
*/
static double sgn(double val) {
    if (val == 0.0) 
        return 0.0;
    return (0.0 < val) - (val < 0.0);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

namespace algebra
{   
    namespace Solvers
    {
        namespace FisherMatrix
        {
            namespace CG
            {

            };

            namespace MINRES_QLP
            {
				const std::vector<std::string> MINRES_QLP_MESSAGES = {
					" Value: beta_k = 0. If M = I, b and x are eigenvectors     		",
					" Given rtol, a solution to Ax = b was found.						",
					" Min-length solution for singular LS problem found, given rtol. 	",
					" A solution to Ax = b was found, given eps.						",	
					" Min-length solution for singular LS problem found, given eps.		",
					" x has converged to an eigenvector.								",
					" |x| has exceeded MAXXNORM.										",
					" Acond has exceeded CONLIM.										",
					" Least-squares problem but no converged solution yet.				"
				};

				constexpr double MAXXNORM 	= 1.0e+7;	// maximum norm of the solution
				constexpr double CONLIM 	= 1.0e+15;	// maximum condition number of the matrix
				constexpr double TRANSCOND 	= 1.0e+7;	// condition number for the transposed matrix

                // -----------------------------------------------------------------------------------------------------------------------------------------
                
                template <typename _T1>
				arma::Col<_T1> minres_qlp(  const arma::Mat<_T1>& _DeltaO,
                                            const arma::Mat<_T1>& _DeltaOConjT,
                                            const arma::Col<_T1>& _F,
                                            arma::Col<_T1>* _x0,
                                            double _eps,
                                            size_t _max_iter,
                                            bool* _converged, 
                                            double _reg)
                {
					// !TODO: Implement the MINRES_QLP solver
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
                
                // -----------------------------------------------------------------------------------------------------------------------------------------
				
                /*
				* @brief MINRES_QLP solver for the Fisher matrix inversion. This method is used whenever the matrix can be
				* decomposed into the form S = \Delta O^* \Delta O, where \Delta O is the derivative of the observable with respect to the parameters.
				* The matrix S is symmetric and positive definite, so the MINRES_QLP method can be used.
				* The method is a preconditioned solver for approximating (A-\sigma I) * x = b, where A = S and b = F. 
				* Preconditioning makes it A = M^{-1/2} (A - \sigma I) M^{-1/2} x = M^{-1/2} b, where M is the preconditioner.
				* @ref S.-C. T. Choi and M. A. Saunders MINRES-QLP for Symmetric and Hermitian Linear Equations and Least-Squares Problems
				* @equation S_{ij} = <\Delta O^*_i \Delta O_j> / N
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
				arma::Col<_T1> minres_qlp(SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(_T1))
				{
					assert(_DeltaO.n_cols == _F.n_elem); 									// check the dimensions of the matrix and the vector

					if (_preconditioner == nullptr)
						return minres_qlp<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);

					const size_t _n 	= _F.n_elem;										// number of elements			
					if (_max_iter == 0)
						_max_iter 		= _n;												// maximum number of iterations

					// initial values of the vectors - solution vectors
					arma::Col<_T1> x_km2= arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector x_{k-2}
					arma::Col<_T1> x_km1= arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector x_{k-1}
					arma::Col<_T1> x_k	= arma::Col<_T1>(_n, arma::fill::zeros);			// final solution vector
   					arma::Col<_T1> x_km3= (_x0) ? *_x0 : x_k;         			            // Initial guess for x

					// initial values of the vectors 
					arma::Col<_T1> _w_k(_n, arma::fill::zeros), _w_km1(_n, arma::fill::zeros), _w_km2(_n, arma::fill::zeros); // use them as previous values of w's
					arma::Col<_T1> z_km2= arma::Col<_T1>(_n, arma::fill::zeros);			// initial vector z0 
					arma::Col<_T1> z_km1 = _F; 												// initial vector z1
					arma::Col<_T1> z_k; 													// vector z_{k+1} in the loop - updated at each iteration
					arma::Col<_T1> q_k 	= _preconditioner->apply(z_km2);					// initial vector q1
					arma::Col<_T1> p_k 	= q_k;												// initial vector p1
					
   		 			// Initial values for β and φ
					double _beta_km1 = 0.0, _beta_k = algebra::real(std::sqrt(arma::cdot(_F, q_k))); // beta_0 = 0, beta_1 = sqrt(_F^T * q1)
					double _phi_k = 0.0, _phi_km1 = _beta_k;									// use them as previous values of phi's
		
					// for the reflection and Lanczos tridiagonalization 
					double _c_km1_1 = -1.0, _c_km1_2 [[maybe_unused]]= -1.0, _c_km1_3 [[maybe_unused]] = -1.0, _c_k_1, _c_k_2, _c_k_3; // use them as previous values of c's
					double _s_km1_1 = 0.0, _s_km1_2 [[maybe_unused]] = 0.0, _s_km1_3 [[maybe_unused]] = 0.0, _s_k_1, _s_k_2, _s_k_3; // use them as previous values of s's
					double _tau_k = 0.0, _tau_km1 = 0.0, _tau_km2;							// use them as previous values of tau's
					double _om_k = 0.0, _om_km1 = 0.0;										// use them as previous values of omega's
					double _chi_k = 0.0, _chi_km1 = 0.0, _chi_km2 = 0.0, _chi_km3 = 0.0;	// use them as previous values of chi's
					double _kappa_k [[maybe_unused]] = 0.0;													// use them as previous values of kappa's
					double _A_k = 0.0, _A_km1 = 0.0;										// use them as previous values of A's
					double _delta_k = 0.0;														// use them as previous values of delta's
					double _gamm_k = 0.0, _gamm_km1 = 0.0, _gamm_km2 = 0.0;					// use them as previous values of gamma's
					_T1 _alpha_k = 0.0;														// alpha_k = 0 - at each iteration
					double _eta_k = 0.0, _eta_km1 = 0.0, _eta_km2 = 0.0;					// use them as previous values of eta's
					double _theta_k = 0.0, _theta_km1 = 0.0, _theta_km2;					// use them as previous values of theta's
					double _mu_k = 0.0, _mu_km1 = 0.0, _mu_km2 = 0.0, _mu_km3, _mu_km4 = 0.0;	// use them as previous values of mu'
					
					double _rho_k = 0.0;													// rho_k = 0 - at each iteration
					double _gamm_min = 0.0;
					double _eps_k = 0.0;									// use them as previous values of eps's
					double _psi_k = 0.0, _psi_km1 [[maybe_unused]] = 0.0;										// use them as previous values of psi's

					// iterate until convergence, if the value is k+1 in the algorithm, we update the corresponding variables.
					// Remember, here the index starts from 0 (k = -1 is the iteration before the loop)
					for (int k = 0; k < _max_iter; ++k)
					{
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// preconditioned Lanczos 
						// MINRES and MINRES-QLP use the symmetric Lanczos process [Lanczos 1950] to reduce
						// A to a tridiagonal form Tk. The process is initialized with v0 ≡ 0, β1 = ‖b‖, and β1v1 = b.
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						
						p_k 				= matrixFreeMultiplication(_DeltaO, _DeltaOConjT, q_k, -1.0) - _reg * q_k;	// apply the matrix-vector multiplication (here the regularization is applied)
						_alpha_k 			= arma::cdot(q_k, p_k) / _beta_k / _beta_k;									// alpha_k = (q_k, p_k) / beta_k^2 - at each iteration
						z_k					= p_k / _beta_k - (_alpha_k / _beta_k) * z_km1 - (_beta_km1 / _beta_k) * z_km2;	// z_{k+1} = p_k / beta_k - (alpha_k / beta_k) * z_k - (beta_{k-1} / beta_k) * z_{k-1} - updated at each iteration 
						q_k 				= _preconditioner->apply(z_k);												// apply the preconditioner - can be updated now as is not used anymore
						//! new value of beta_k - update later
						double _beta_kp1 	= algebra::real(std::sqrt(arma::cdot(q_k, z_k)));							// beta_{k+1} = sqrt(q_k, z_k) - defined here, beta needs to be updated at each iteration
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k == 0) _rho_k 	= std::sqrt(std::abs(_alpha_k) * std::abs(_alpha_k) + (_beta_kp1 * _beta_kp1));		// ||[αk βk+1]|| - at the first iteration - beta is real
						else _rho_k			= std::sqrt(std::abs(_alpha_k) * std::abs(_alpha_k) + (_beta_kp1 * _beta_kp1) + (_beta_k * _beta_k)); // ||[βk αk βk+1]|| - at each iteration - beta is real
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						// update delta_k ((2) second stage) 
						_delta_k 			= (_c_km1_1 * _delta_k) + algebra::real(_s_km1_1 * _alpha_k); 							// previous left reflection - at each iteration - is a second stage for this variable
						_gamm_k 			= (_s_km1_1 * _delta_k) - algebra::real(_c_km1_1 * _alpha_k); 				// [on middle two entries of Tk ek...] - at each iteration
						//! new value of eta_k - update later
						double _eps_k_p1	= (_s_km1_1 * _beta_kp1); 													// [produces first two entries in Tk+1ek+1] - local variable
						//! new value of delta_k - update later
						double _delta_kp1 	= (-_c_km1_1 * _beta_kp1); 													// update delta after the reflection
						
						// use symortho to get the new reflection
						// update _gamma_k ((2) second stage)
						sym_ortho(_gamm_k, _beta_kp1, _c_k_1, _s_k_1, _gamm_k); 										// [current left reflection]
						// update _gamma_km2 and _gamma_km1 ((5) fifth and (6) sixth stage)
						sym_ortho(_gamm_km2, _eps_k, _c_k_2, _s_k_2, _gamm_km2); 										// [First right reflection]		
						// update _theta_km1 ((2) second stage)
						_theta_km1 			= (_c_k_2 * _theta_km1) + (_s_k_2 * _delta_k); 									// use delta from (2) stage 
						// update delta_k ((3) third stage)
						_delta_k 			= (_s_k_2 * _theta_km1) - (_c_k_2 * _delta_k);
						_eta_k 				= _s_k_2 * _gamm_k;															// use gamma from (2) stage
						// update _gamma_k ((3) third stage)
						_gamm_k 			= -_c_k_2 * _gamm_k;
						// update _gamma_km1 ((4) fourth stage and (5) fifth stage)
						sym_ortho(_gamm_km1, _delta_k, _c_k_3, _s_k_3, _gamm_km1); 										// [Second right reflection]
						_theta_k 			= _s_k_3 * _gamm_k; 														// use gamma from (3) stage	
						// update _gamma_k ((4) fourth stage)
						_gamm_k 			= -(_c_k_3 * _gamm_k);														// [to zero out δ(3)k]
						_tau_k 				= _c_k_1 * _phi_km1; 														// [Last element of tk]
						_phi_k 				= _s_k_1 * _phi_km1; 														// [Update ‖ ¯r k‖, ‖ ˜A¯r k−1‖]
						_psi_km1 			= _phi_km1 * std::sqrt(_gamm_k * _gamm_k + _delta_kp1 * _delta_kp1);

						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k == 0)
							_gamm_min 		= _gamm_k;
						else
							_gamm_min 		= std::min(_gamm_km2, std::min(_gamm_km1, std::min(_gamm_min, std::abs(_gamm_k))));
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!

						_A_k 				= std::max(_A_km1, std::max(_rho_k, std::max(_gamm_km2, std::max(_gamm_km1, std::abs(_gamm_k)))));	// [Update ‖A‖]
						_om_k 				= std::sqrt(_om_km1 * _om_km1 + _tau_k * _tau_k);										// [Update ‖ ˜Axk‖, cond( ˜A)]
						_kappa_k 			= _A_k / _gamm_min;															// [Update wk−2, wk−1, wk]
						_w_k 				= -(_c_k_2 / _beta_k) * q_k + _s_k_2 * _w_km2;								// [Update wk], use w_km2 from (3) stage
						_w_km2 				= (_s_k_2 / _beta_k) * q_k + _c_k_2 * _w_km2;								// [Update wk−2], use w_km2 from (3) stage
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 1)
						{
							// update w_k ((2) second stage)
							_w_k = _s_k_3 * _w_km1 - _c_k_3 * _w_k;														// [Update wk], use w_km1 from (2) stage
							_w_km1 = _c_k_3 * _w_km1 + _s_k_3 * _w_k;													// [Update wk−1], use w_km1 from (2) stage
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 1)
						{
							_mu_km2 = (_tau_km2 - _eta_km2 * _mu_km4 - _theta_km2 * _mu_km3) / _gamm_km2;				// [Update µk−2]
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (k > 0)
						{
							_mu_km1 = (_tau_km1 - _eta_km1 * _mu_km3 - _theta_km1 * _mu_km2) / _gamm_km1;				// [Update µk−1]
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						if (std::abs(_gamm_k) > _eps)
						{
							_mu_k = (_tau_k - _eta_k * _mu_km2 - _theta_k * _mu_km1) / _gamm_k;							// [Update µk]
						} else {
							_mu_k = 0.0;
						}
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!
						
						// update the solution
						x_km2 		= x_km3 + _mu_km2 * _w_km2;															// [Update xk−2]
						x_k 		= x_km2 + _mu_km1 * _w_km1 + _mu_k * _w_k;											// [Update xk]

						// check for convergence
						_chi_km2 	= std::sqrt(_chi_km3 * _chi_km3 + _mu_km2 * _mu_km2);											// [Update ‖xk−2‖]
						_chi_k 		= std::sqrt(_chi_km2 * _chi_km2 + _mu_km1 * _mu_km1 + _mu_k * _mu_k);							// [Update ‖xk‖]
						// !!!!!!!!!!!!!!!!!!!!!!!!!!!!

						// update the values for the next iteration
						{
							// values of delta
							_delta_k 	= _delta_kp1;
							// values of beta
							_beta_km1 	= _beta_k;
							_beta_k 	= _beta_kp1;
							// values of c - tridiagonal matrix
							_c_km1_1 	= _c_k_1;
							_c_km1_2 	= _c_k_2;
							_c_km1_3 	= _c_k_3;
							// values of s - tridiagonal matrix
							_s_km1_1 	= _s_k_1;
							_s_km1_2 	= _s_k_2;
							_s_km1_3 	= _s_k_3;
							// values of gamma
							_gamm_km2 	= _gamm_km1;
							_gamm_km1 	= _gamm_k;
							// values of delta
							_eta_km2 	= _eta_km1;
							_eta_km1 	= _eta_k;
							// values of eps
							_eps_k 		= _eps_k_p1;
							// values of theta
							_theta_km2 = _theta_km1;
							_theta_km1 = _theta_k;
							// values of tau
							_tau_km2 = _tau_km1;
							_tau_km1 = _tau_k;
							// values of phi
							_phi_km1 = _phi_k;
							// values of psi
							_psi_km1 = _psi_k;
							// values of A
							_A_km1 = _A_k;
							// values of omega
							_om_km1 = _om_k;
							// values of mu
							_mu_km4 = _mu_km3;
							_mu_km3 = _mu_km2;
							_mu_km2 = _mu_km1;
							_mu_km1 = _mu_k;
							// values of chi
							_chi_km3 = _chi_km2;
							_chi_km2 = _chi_km1;
							_chi_km1 = _chi_k;
							// vector W
							_w_km2 = _w_km1;
							_w_km1 = _w_k;
						}

						if (std::abs(_phi_k) < _eps)
						{
							if (_converged != nullptr)
								*_converged = true;
							return x_k;
						}
					}
					
					std::cerr << "\t\t\tMINRES_QLP solver did not converge." << std::endl;
					if (_converged != nullptr)
						*_converged = false;
					return x_k;
				}
                
                // -----------------------------------------------------------------------------------------------------------------------------------------

                // define the template specializations
                // double
                template arma::Col<double> minres_qlp(SOLVE_FISHER_ARG_TYPES(double));
                template arma::Col<double> minres_qlp(SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(double));
                // complex double
                template arma::Col<std::complex<double>> minres_qlp(SOLVE_FISHER_ARG_TYPES(std::complex<double>));
                template arma::Col<std::complex<double>> minres_qlp(SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(std::complex<double>));

                // -----------------------------------------------------------------------------------------------------------------------------------------

            };
        };
    };
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// !GENERAL SOLVER FOR THIS TYPE OF PROBLEM

namespace algebra
{
    namespace Solvers
    {
        namespace FisherMatrix
        {
            template <typename _T1>
			arma::Col<_T1> solve(Type _type, SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(_T1))
			{
				switch (_type) 
				{
				case Type::ARMA:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::solve(_S, _F);
				}
				case Type::ConjugateGradient:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				case Type::MINRES_QLP:
					return MINRES_QLP::minres_qlp<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				case Type::PseudoInverse:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::pinv(_S, _eps) * _DeltaOConjT * _F;
				}
				case Type::Direct:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::inv(_S) * _F;
				}
				default:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}
			}

			template <typename _T1>
			inline arma::Col<_T1> solve(int _type, SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(_T1)) { return solve<_T1>(static_cast<Type>(_type), _DeltaO, _DeltaOConjT, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }

			template <typename _T1>
			inline arma::Col<_T1> solve(Type _type, SOLVE_FISHER_ARG_TYPES(_T1))
			{
				switch (_type) 
				{
				case Type::ARMA:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::solve(_S, _F);
				}
				case Type::ConjugateGradient:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				case Type::MINRES_QLP:
					return MINRES_QLP::minres_qlp<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				case Type::PseudoInverse:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::pinv(_S, _eps) * _DeltaOConjT * _F;
				}
				case Type::Direct:
				{
					arma::Mat<_T1> _S 	= 	_DeltaOConjT * _DeltaO;
					_S.diag() 			+= 	_reg;
					return arma::inv(_S) * _F;
				}
				default:
					return CG::conjugate_gradient<_T1>(_DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
			}
			
			template <typename _T1>
			inline arma::Col<_T1> solve(int _type, SOLVE_FISHER_ARG_TYPES(_T1)) { return solve<_T1>(static_cast<Type>(_type), _DeltaO, _DeltaOConjT, _F, _x0, _eps, _max_iter, _converged, _reg); }

            // -----------------------------------------------------------------------------------------------------------------------------------------

            // define the template specializations
            // double
            template arma::Col<double> solve(Type _type, SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(double));
            template arma::Col<double> solve(int _type, SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(double));
            template arma::Col<double> solve(Type _type, SOLVE_FISHER_ARG_TYPES(double));
            template arma::Col<double> solve(int _type, SOLVE_FISHER_ARG_TYPES(double));

            // complex double
            template arma::Col<std::complex<double>> solve(Type _type, SOLVE_FISHER_ARG_TYPES(std::complex<double>));
            template arma::Col<std::complex<double>> solve(int _type, SOLVE_FISHER_ARG_TYPES(std::complex<double>));
            template arma::Col<std::complex<double>> solve(Type _type, SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(std::complex<double>));
            template arma::Col<std::complex<double>> solve(int _type, SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(std::complex<double>));
            
            // -----------------------------------------------------------------------------------------------------------------------------------------

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

        }
    };
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%