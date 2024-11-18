#include "../../../src/lin_alg.h"
#include "../../../src/flog.h"

// #################################################################################################################################################

namespace algebra
{
    namespace Solvers
    {
        // #################################################################################################################################################

        // class template instantiation
        template class Arnoldi<double, true, false>;
        template class Arnoldi<std::complex<double>, true, false>;
        
        template class Arnoldi<double, false, false>;
        template class Arnoldi<std::complex<double>, false, false>;

        template class Arnoldi<double, true, true>;
        template class Arnoldi<std::complex<double>, true, true>;

        template class Arnoldi<double, false, true>;
        template class Arnoldi<std::complex<double>, false, true>;

        // #################################################################################################################################################

        template <typename _T, bool _symmetric, bool _reorthogonalize>
        void Arnoldi<_T, _symmetric, _reorthogonalize>::init(const arma::Col<_T>& _F, arma::Col<_T>* _x0)
        {
            this->N_    = _F.n_elem;
            if (this->N_ == 0)
                throw std::invalid_argument("Arnoldi: The input vector is empty.");

            if (this->max_iter_ == 0)
                this->max_iter_ = this->N_;
            
            // initialize the basis
            this->V_ = arma::zeros<arma::Mat<_T>>(this->N_, this->max_iter_ + 1);
            if (this->precond_ != nullptr)
            {
                this->isPreconditioned_ = true;
                this->P_ = arma::zeros<arma::Mat<_T>>(this->N_, this->max_iter_ + 1);
            }

            // initialize the Hessenberg matrix
            this->H_ = arma::zeros<arma::Mat<_T>>(this->max_iter_ + 1, this->max_iter_);

            // generate the first basis vector
            this->v_ = _F;
            if (this->precond_ != nullptr)
            {
                this->p_        = this->v_;
                this->v_        = this->precond_->apply(this->v_, this->reg_);
                this->vnorm_    = arma::norm(this->v_);
                if (this->vnorm_ > 0)
                    this->P_.col(0) = this->p_ / this->vnorm_;
            } else {
                this->vnorm_    = arma::norm(this->v_);
            }

            // check if the norm is ok
            if (this->vnorm_ > 0)
                this->V_.col(0) = this->v_ / this->vnorm_;
            else
            {
                LOGINFO("Arnoldi: The input vector is zero. It is invarian to A...", LOG_TYPES::ERROR, 2);
                this->invariant_ = true;
            }
        }

        // #################################################################################################################################################

        template <typename _T, bool _symmetric, bool _reorthogonalize>
        Arnoldi<_T, _symmetric, _reorthogonalize>::Arnoldi(size_t _N, double _eps, size_t _max_iter, double _reg, Precond<_T, _symmetric>* _preconditioner)
            : General::Solver<_T, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
        {
            this->isPreconditioned_ = (_preconditioner != nullptr);
        }

        // #################################################################################################################################################

        template <typename _T, bool _symmetric, bool _reorthogonalize>
        void Arnoldi<_T, _symmetric, _reorthogonalize>::solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0, Precond<_T, _symmetric>* _precond)
        {
            if (_precond != nullptr)
            {
                this->precond_          = _precond;
                this->isPreconditioned_ = true;
            }
            this->init(_F, _x0);
            this->iterate();
        }

        // #################################################################################################################################################

        /**
        * @brief Perform the Arnoldi iteration as a single step of the algorithm 
        * @tparam _T data type
        * @tparam _symmetric flag for the symmetric matrix
        * @tparam _reorthogonalize flag for the reorthogonalization        
        */
        template <typename _T, bool _symmetric, bool _reorthogonalize>
        void Arnoldi<_T, _symmetric, _reorthogonalize>::advance()
        {
            // check if the iteration is finished
            if (this->iter_ >= this->max_iter_)
                return;

            if (this->invariant_)
            {
                LOGINFO("Arnoldi: The input vector is zero. It is invariant to A...", LOG_TYPES::ERROR, 2);
                return;
            }

            this->N_ = this->V_.n_rows;
            size_t k = this->iter_;

            // apply matrix - vector multiplication
            this->Av_ = this->matVecFun_(this->V_.col(k), this->reg_);

            // check if is symmetric -> Lanczos
            if constexpr (_symmetric)
            {
                if (k > 0)
                {
                    this->H_(k - 1, k) = this->H_(k, k - 1);                    // copy the last element
                    if (this->precond_ != nullptr)
                    {
                        this->Av_ -= this->H_(k, k - 1) * this->P_.col(k - 1);  // subtract the last element of the last column
                    } else {
                        this->Av_ -= this->H_(k, k - 1) * this->V_.col(k - 1);  // subtract the last element of the last column
                    }
                }
            }

            // modified Gram-Schmidt to the previous vectors - reorthogonalization to the previous vectors
            size_t _start = 0;
            if (reorthogonalize_ || !_symmetric)                                // or perform the Arnoldi iteration
            {
                for (size_t i = _start; i < k; ++i)
                {
                    _T _alpha = arma::cdot(this->V_.col(i), this->Av_);
                    if (std::abs(_alpha) > 1e-14)
                    {
                        if (this->isPreconditioned_)
                        {
                            this->Av_ -= _alpha * this->P_.col(i);
                        } else {
                            this->Av_ -= _alpha * this->V_.col(i);
                        }
                        if constexpr (_symmetric)
                            this->H_(i, k) += _alpha;
                    }
                }
            }

            // check the final preconditioned vector
            if (this->precond_ != nullptr)
            {
                this->MAv_          = this->precond_->apply(this->Av_, this->reg_);
                this->vnorm_        = arma::norm(this->MAv_);
            } else {
                this->vnorm_ = arma::norm(this->Av_);
            }
            this->H_(k + 1, k)  = this->vnorm_;

            if (std::abs(this->vnorm_) < 1e-14)
            {
                LOGINFO("Arnoldi: The vector is linearly dependent.", LOG_TYPES::ERROR, 2);
                this->invariant_ = true;
                return;
            }
            else {
                if (this->precond_ != nullptr)
                {
                    this->P_.col(k + 1) = this->Av_ / this->vnorm_;
                    this->V_.col(k + 1) = this->MAv_ / this->vnorm_;
                }
                else
                {
                    this->V_.col(k + 1) = this->Av_ / this->vnorm_;
                }
            }

            // increase the iteration counter
            ++this->iter_;
        }

        // #################################################################################################################################################

        /**
        * @brief Perform the Arnoldi iteration as a full step of the algorithm
        * @tparam _T data type
        * @tparam _symmetric flag for the symmetric matrix
        * @tparam _reorthogonalize flag for the reorthogonalization        
        */
        template <typename _T, bool _symmetric, bool _reorthogonalize>
        void Arnoldi<_T, _symmetric, _reorthogonalize>::iterate()
        {
            while (this->iter_ < this->max_iter_)
                this->advance();
        }

        // #################################################################################################################################################


    };



};
