#include "../../../src/lin_alg.h"
#include "../../../src/flog.h"
#include "../../../src/common.h"
#include "armadillo"
#include <cmath>

// #################################################################################################################################################

namespace algebra
{
    namespace Solvers
    {
        namespace General
        {
            namespace ARMA
            {
                // ############################################################################################################################################

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::init(const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->converged_ = false;
                    if (this->N_ != _F.n_elem)
                        this->N_ = _F.n_elem;
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::init(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = _A;
                    this->N_        = _A.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::init(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_A);
                    this->N_        = _A.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::init(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_Sp * _S) / _Sp.n_cols;
                    this->N_        = this->Amat_.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::init(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_Sp * _S) / _Sp.n_cols;
                    this->N_        = _S.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                // ############################################################################################################################################

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        this->init(_F, _x0);
                        if (this->isSymmetric_)
                        {
                            if (this->reg_ > 0)
                                this->x_ = arma::solve(this->Amat_ + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), _F, arma::solve_opts::likely_sympd);
                            else
                                this->x_ = arma::solve(this->Amat_, _F, arma::solve_opts::likely_sympd);
                        } else {
                            if (this->reg_ > 0)
                                this->x_ = arma::solve(this->Amat_ + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), _F);
                            else
                                this->x_ = arma::solve(this->Amat_, _F);
                        
                        }
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::solve(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        if (this->isSymmetric_)
                        {
                            if (this->reg_ > 0)
                                this->x_ = arma::solve(_A + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), _F, arma::solve_opts::likely_sympd);
                            else
                                this->x_ = arma::solve(_A, _F, arma::solve_opts::likely_sympd);
                        } else {
                            if (this->reg_ > 0)
                                this->x_ = arma::solve(_A + this->reg_ * arma::eye<arma::Mat<_T>>(this->N_, this->N_), _F);
                            else
                                this->x_ = arma::solve(_A, _F);
                        }
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::solve(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try 
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        if (this->reg_ > 0)
                            this->x_ = arma::spsolve(_A + this->reg_ * arma::speye<arma::SpMat<_T>>(this->N_, this->N_), _F);
                        else
                            this->x_ = arma::spsolve(_A, _F);
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::solve(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        this->isGram_   = true;
                        this->init(_S, _Sp, _F, _x0);    
                        if (this->isSymmetric_)
                        {
                            if (this->reg_ > 0)
                                this->x_ = arma::solve(_Sp * _S / _Sp.n_cols + this->reg_ * arma::eye<arma::Mat<_T>>(_F.n_elem, _F.n_elem), _F, arma::solve_opts::likely_sympd);
                            else
                                this->x_ = arma::solve(_Sp * _S / _Sp.n_cols, _F, arma::solve_opts::likely_sympd);
                        } 
                        else 
                        {
                            if (this->reg_ > 0)
                                this->x_ = arma::solve(_Sp * _S / _Sp.n_cols + this->reg_ * arma::eye<arma::Mat<_T>>(_F.n_elem, _F.n_elem), _F);
                            else
                                this->x_ = arma::solve(_Sp * _S / _Sp.n_cols, _F);
                        }
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void ARMA_s<_T, _symmetric>::solve(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try 
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        this->isGram_ = true;
                        if (this->reg_ > 0)
                            this->x_ = arma::spsolve(_Sp * _S / _Sp.n_cols + this->reg_ * arma::speye<arma::SpMat<_T>>(_F.n_elem, _F.n_elem), _F);
                        else
                            this->x_ = arma::spsolve(_Sp * _S / _Sp.n_cols, _F);
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                // ############################################################################################################################################
            
                // define the template specializations
                template class ARMA_s<double, true>;
                template class ARMA_s<std::complex<double>, true>;
                template class ARMA_s<double, false>;
                template class ARMA_s<std::complex<double>, false>;
            };
        };
    };
};