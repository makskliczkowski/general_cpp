#include "../../../src/lin_alg.h"
#include "../../../src/flog.h"
#include "../../../src/common.h"
#include <cmath>

// #################################################################################################################################################

namespace algebra
{
    namespace Solvers
    {
        namespace General
        {
            namespace PseudoInverse
            {
                // ############################################################################################################################################

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::init(const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->converged_ = false;
                    if (this->N_ != _F.n_elem)
                        this->N_ = _F.n_elem;
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::init(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = _A;
                    this->N_        = _A.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::init(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_A);
                    this->N_        = _A.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::init(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_Sp * _S) / _Sp.n_cols;
                    this->N_        = _F.n_elem;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::init(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_Sp * _S) / _Sp.n_cols;
                    this->N_        = _F.n_elem;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                // ############################################################################################################################################

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        this->init(_F, _x0);
                        if (this->reg_ > 0)
                        {
                            this->x_ = arma::pinv(this->Amat_ + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), this->eps_) * _F;
                        } else {
                            this->x_ = arma::pinv(this->Amat_, this->eps_) * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::solve(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        if (this->reg_ > 0)
                        {
                            this->x_ = arma::pinv(_A + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), this->eps_) * _F;
                        } else {
                            this->x_ = arma::pinv(_A, this->reg_) * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::solve(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        if (this->reg_ > 0)
                        {
                            this->x_ = arma::pinv(arma::Mat<_T>(_A) + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), this->eps_) * _F;
                        } else {
                            this->x_ = arma::pinv(arma::Mat<_T>(_A), this->eps_) * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::solve(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }

                        if (this->reg_ > 0)
                        {
                            this->x_ = arma::pinv(_Sp * _S / _Sp.n_cols + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), this->eps_) * _F;
                        } else {
                            this->x_ = arma::pinv(_Sp * _S / _Sp.n_cols, this->eps_) * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void PseudoInverse_s<_T, _symmetric>::solve(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        if (this->reg_ > 0)
                        {
                            this->x_ = arma::pinv(arma::Mat<_T>(_Sp * _S) / _Sp.n_cols + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), this->eps_) * _F;
                        } else {
                            this->x_ = arma::pinv(arma::Mat<_T>(_Sp * _S) / _Sp.n_cols, this->eps_) * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                // ############################################################################################################################################
            
                // define the template specializations
                template class PseudoInverse_s<double, true>;
                template class PseudoInverse_s<std::complex<double>, true>;
                template class PseudoInverse_s<double, false>;
                template class PseudoInverse_s<std::complex<double>, false>;
            };
        };
    };
};