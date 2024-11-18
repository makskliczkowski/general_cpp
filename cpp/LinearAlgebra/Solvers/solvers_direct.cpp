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
            namespace Direct
            {
                // ############################################################################################################################################

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::init(const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->converged_ = false;
                    if (this->N_ != _F.n_elem)
                        this->N_ = _F.n_elem;
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::init(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = _A;
                    this->N_        = _A.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::init(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_A);
                    this->N_        = _A.n_cols;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::init(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_Sp * _S) / _Sp.n_cols;
                    this->N_        = _F.n_elem;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::init(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0)
                {
                    this->Amat_     = arma::Mat<_T>(_Sp * _S) / _Sp.n_cols;
                    this->N_        = _F.n_elem;
                    if (this->N_ == 0)
                        throw std::invalid_argument("PseudoInverse: The input matrix is empty.");
                    this->init(_F, _x0);
                }

                // ############################################################################################################################################

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (_precond != nullptr)
                        {
                            this->precond_          = _precond;
                            this->isPreconditioned_ = true;
                        }
                        this->init(_F, _x0);
                        this->x_ = this->Amat_.i() * _F;
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::solve(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {

                    if (this->reg_ > 0)
                    {
                        this->x_ = arma::inv(_A + this->reg_ * arma::eye<arma::Mat<_T>>(_A.n_rows, _A.n_cols)) * _F;
                    } else {
                        this->x_ = _A.i() * _F;
                    }
                    }
                    catch (std::exception& e)
                    {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::solve(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (this->reg_ > 0)
                        {
                            this->x_ = arma::inv(arma::Mat<_T>(_A + this->reg_ * arma::speye<arma::SpMat<_T>>(_A.n_rows, _A.n_cols))) * _F;
                        } else {
                            this->x_ = arma::inv(arma::Mat<_T>(_A)) * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::solve(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try
                    {
                        if (this->reg_ > 0)
                        {
                            this->x_ = (_Sp * _S + this->reg_ * arma::eye<arma::Mat<_T>>(_F.n_elem, _F.n_elem)).i() * _F;
                        } else {
                            this->x_ = (_Sp * _S).i() * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                    this->converged_ = true;
                }

                template <typename _T, bool _symmetric>
                void Direct_s<_T, _symmetric>::solve(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
                {
                    try {
                        if (this->reg_ > 0)
                        {
                            this->x_ = (arma::Mat<_T>(_Sp * _S) / _Sp.n_cols + this->reg_ * arma::eye(_F.n_elem, _F.n_elem)).i() * _F;
                        } else {
                            this->x_ = arma::Mat<_T>(_Sp * _S / _Sp.n_cols).i() * _F;
                        }
                    } catch (std::exception& e) {
                        LOGINFO("Direct solver: " + std::string(e.what()), LOG_TYPES::ERROR, 2);
                        this->converged_ = false;
                        return;
                    }
                }

                // ############################################################################################################################################
            
                // define the template specializations
                template class Direct_s<double, true>;
                template class Direct_s<std::complex<double>, true>;
                template class Direct_s<double, false>;
                template class Direct_s<std::complex<double>, false>;
            };
        };
    };
};