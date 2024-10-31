/*
* This file contains all the important methods for the Machine Learning part of the library.
*/

#ifndef ML_H
#define ML_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <complex>

namespace MachineLearning {

    namespace EarlyStoppings {

        // ##########################################################################################################################################

        struct EarlyStopping
        {
            size_t patience_            = 0;                   // number of epochs to wait before reducing the learning rate or stopping the training
            double minDelta_            = 1e-3;                // minimum delta for the early stopping
            double best_metric_         = std::numeric_limits<double>::max();
            size_t epoch_since_best_    = 0;
            bool stop_                  = false;

            // ##########################
            EarlyStopping()             = default;
            EarlyStopping(size_t patience, double _minDelta = 1e-3)
                : patience_(patience), minDelta_(_minDelta) {};

            bool operator()(size_t epoch, double _metric = 0.0)
            {
                // check if the metric is nan or inf
                if (std::isnan(_metric) || std::isinf(_metric))
                    return true;
                
                // consider the early stopping only after the patience (zero means no early stopping) 
                if (patience_ == 0) 
                    return false;

                // Check if current_metric has improved
                if (_metric < this->best_metric_) {
                    this->best_metric_              = _metric;
                    this->epoch_since_best_         = 0;
                } else {
                    this->epoch_since_best_++;
                }

                // If we're on a plateau, reduce the learning rate
                if (this->epoch_since_best_ >= this->patience_) {
                    this->stop_ = true;
                }
                return this->stop_;
            }
        };

        // ##########################################################################################################################################
    };

    // ##########################################################################################################################################

    struct Parameters
    {
        double lr_              = 1e-2;         // the initial learning rate
        double lr_decay_        = 0.9;          // the learning rate decay
        size_t max_epochs_      = 100;          // the maximum number of epochs
        size_t patience_        = 5;            // the number of epochs to wait before reducing the learning rate or stopping the training
    
        // setters
        void set_lr(double lr)                  { this->lr_ = lr; }
        void set_decay(double d)                { this->lr_decay_ = d; }
        void set_max_epochs(size_t epochs)      { this->max_epochs_ = epochs; }
        void set_patience(size_t p)             { this->patience_ = p; }

        // learning rate scheduler
        virtual double operator()(size_t epoch, double _metric = 0.0) = 0;

        virtual ~Parameters()   = default;
        Parameters()            = default;
        Parameters(double lr, double decay, size_t max_epochs, size_t patience = 5)
            : lr_(lr), lr_decay_(decay), max_epochs_(max_epochs), patience_(patience) 
        {
            this->early_stopping_ = EarlyStoppings::EarlyStopping();
        }

        // --------------------------------------------------------------------------------------------

        // for the early stopping
        EarlyStoppings::EarlyStopping early_stopping_;
        auto set_early_stopping(size_t patience, double minDelta = 1e-3) -> void    { this->early_stopping_ = EarlyStoppings::EarlyStopping(patience, minDelta); }
        auto stop(size_t epoch, double _metric = 0.0) -> bool                       { return this->early_stopping_(epoch, _metric); }
        auto stop(size_t epoch, std::complex<double> _metric) -> bool               { return this->early_stopping_(epoch, std::real(_metric)); }
    };

        
    // ---------- S C H E D U L E S ----------
    namespace Schedulers {
        
        enum class SchedulerType
        {
            Constant    = 0,
            Exponential = 1,
            Step        = 2,
            Cosine      = 3,
            Adaptive    = 4
        };

        // ##########################################################################################################################################
        
        struct ConstantScheduler : public Parameters
        {
            ConstantScheduler(double initial_lr, double decay_rate = 0.1, size_t max_epochs = 100)
                : Parameters(initial_lr, decay_rate, max_epochs) {};

            double operator()(size_t epoch, double _metric = 0.0) override final { return this->lr_; }
        };

        struct ExponentialDecayScheduler : public Parameters
        {
            ExponentialDecayScheduler(double initial_lr, double decay_rate = 0.1, size_t max_epochs = 100)
                : Parameters(initial_lr, decay_rate, max_epochs) {};

            double operator()(size_t epoch, double _metric = 0.0) override final { return this->lr_ * std::exp(-this->lr_decay_ * epoch); }
        };

        struct StepDecayScheduler : public Parameters
        {
            size_t step_size_ = 10;

            StepDecayScheduler(double initial_lr, double decay_rate = 0.1, size_t max_epochs = 100, size_t step_size = 10)
                : Parameters(initial_lr, decay_rate, max_epochs), step_size_(step_size) {};

            double operator()(size_t epoch, double _metric = 0.0) override final { return this->lr_ * std::pow(this->lr_decay_, std::floor(epoch / this->step_size_)); }
        };

        struct CosineAnnealingScheduler : public Parameters
        {
            CosineAnnealingScheduler(double initial_lr, double decay_rate = 0.1, size_t max_epochs = 100)
                : Parameters(initial_lr, decay_rate, max_epochs) {};

            double operator()(size_t epoch, double _metric = 0.0) override final { return this->lr_ / 2.0 * (1.0 + std::cos(M_PI * epoch / this->max_epochs_)); }
        };

        struct AdaptiveScheduler : public Parameters
        {
            size_t cooldown_;                       // Number of epochs to wait after a reduction
            double min_lr_                  = 1e-5; // Minimum learning rate
            size_t epoch_since_reduction_   = 0;
            size_t cooldown_counter_        = 0;
            double best_metric_;

            AdaptiveScheduler(double initial_lr, double decay_rate = 0.1, size_t max_epochs = 100, double min_lr = 1e-5, size_t _patience = 5, size_t _cooldown = 5)
                : Parameters(initial_lr, decay_rate, max_epochs, _patience), cooldown_(_cooldown), min_lr_(min_lr), best_metric_(std::numeric_limits<double>::max())
            {};

            double operator()(size_t epoch, double _metric = 0.0) override final
            {
                // If we're cooling down, skip adjustment
                if (this->cooldown_counter_ > 0) {
                    this->cooldown_counter_--;
                    return this->lr_;
                }

                // Check if current_metric has improved
                if (_metric < this->best_metric_) {
                    this->best_metric_              = _metric;
                    this->epoch_since_reduction_    = 0;
                } else {
                    this->epoch_since_reduction_++;
                }

                // If we're on a plateau, reduce the learning rate
                if (this->epoch_since_reduction_ >= this->patience_) {
                    this->lr_                       = std::max(this->min_lr_, this->lr_ * this->lr_decay_);
                    this->epoch_since_reduction_    = 0;
                    this->cooldown_counter_         = this->cooldown_;
                }
                return this->lr_;
            }

        };

        // ##########################################################################################################################################

        /*
        * @brief Creates a scheduler based on the type of scheduler requested by the user. Remember to delete the scheduler after use.
        * @param scheduler The type of scheduler to be used.
        * @param lr The initial learning rate.
        * @param decay The decay rate of the learning rate.
        * @param max_epochs The maximum number of epochs.
        * @param patience The number of epochs to wait before reducing the learning rate.
        * @param cooldown The number of epochs to wait after a reduction.
        * @return A pointer to the scheduler.
        */
        inline Parameters* get_scheduler(const std::string& scheduler, double lr, size_t max_epochs = 100, double decay = 0.9, size_t patience = 10, size_t cooldown = 10)
        {
            if (scheduler == "exponential")
                return new ExponentialDecayScheduler(lr, decay, max_epochs);
            else if (scheduler == "step")
                return new StepDecayScheduler(lr, decay, max_epochs);
            else if (scheduler == "cosine")
                return new CosineAnnealingScheduler(lr, decay, max_epochs);
            else if (scheduler == "adaptive")
                return new AdaptiveScheduler(lr, decay, max_epochs, 1e-5, patience, cooldown);
            else if (scheduler == "constant")
                return new ConstantScheduler(lr, decay, max_epochs);
            else
                return new ExponentialDecayScheduler(lr, decay, max_epochs);
        }

        inline Parameters* get_scheduler(int scheduler, double lr, size_t max_epochs = 100, double decay = 0.9, size_t patience = 10, size_t cooldown = 10)
        {
            switch (SchedulerType(scheduler))
            {
            case SchedulerType::Constant:
                return new ConstantScheduler(lr, decay, max_epochs);
            case SchedulerType::Exponential:
                return new ExponentialDecayScheduler(lr, decay, max_epochs);
            case SchedulerType::Step:
                return new StepDecayScheduler(lr, decay, max_epochs);
            case SchedulerType::Cosine:
                return new CosineAnnealingScheduler(lr, decay, max_epochs);
            case SchedulerType::Adaptive:
                return new AdaptiveScheduler(lr, decay, max_epochs, 1e-5, patience, cooldown);
            default:
                return new ExponentialDecayScheduler(lr, decay, max_epochs);
            }
        }

        // ##########################################################################################################################################
    };
};
#endif

//#ifndef COMMON_H
//#include "../src/common.h"
//#endif
//
//#ifndef ML_H
//#define ML_H
//
//template<typename _type>
//class Adam {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta1_0 = 0.9;						// 1st order exponential decay starting parameter
//	double beta1 = 0.9;							// 1st order exponential decay
//	double beta2_0 = 0.99;						// 2nd order exponential decay starting parameter
//	double beta2 = 0.99;						// 2nd order exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//	cpx alpha = 0;								// true learning step
//	arma::Col<_type> m;							// moment vector
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~Adam() = default;
//	Adam() = default;
//	Adam(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->alpha = lr;
//		this->initialize();
//	};
//	Adam(double beta1, double beta2, double lr, double eps, uint size)
//		: beta1(beta1), beta2(beta2), lr(lr), eps(eps), size(size)
//	{
//		this->beta1_0 = beta1;
//		this->beta2_0 = beta2;
//		this->alpha = lr;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->m.zeros();
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->m = arma::Col<_type>(size, arma::fill::zeros);
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad) {
//		this->current_time += 1;
//		this->m = this->beta1_0 * this->m + (1.0 - this->beta1_0) * grad;
//		this->v = this->beta2_0 * this->v + (1.0 - this->beta2_0) * arma::square(grad);
//		// update decays
//		this->beta1 *= this->beta1_0;
//		this->beta2 *= this->beta2_0;
//
//		this->alpha = this->lr * (1.0 - this->beta2) / (1.0 - this->beta1);
//		// calculate the new gradient according to Adam
//		this->gradient = this->alpha * this->m / (arma::sqrt(this->v) + this->eps);
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//template<typename _type>
//class RMSprop_mod {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta_0 = 0.9;						// exponential decay starting parameter
//	double beta = 0.9;							// exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~RMSprop_mod() = default;
//	RMSprop_mod() = default;
//	RMSprop_mod(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta_0 = 0.9;
//		this->beta = 0.9;
//		this->initialize();
//	};
//	RMSprop_mod(double beta, double lr, double eps, uint size)
//		: beta(beta), lr(lr), eps(eps), size(size)
//	{
//		this->beta_0 = beta;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta = this->beta_0;
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad, const arma::Col<_type>& O) {
//		this->current_time += 1;
//		this->v = this->beta * this->v + (1.0 - this->beta) * O % arma::conj(O);
//
//		//this->v.print("v");
//		// calculate the new gradient according to RMSProp arXiv:1910.11163v2
//		this->gradient = this->lr * grad / (arma::sqrt(this->v) + this->eps);
//		//this->gradient.print("grad");
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//
//
//
//#endif#pragma once
//#ifndef COMMON_H
//#include "../src/common.h"
//#endif
//
//#ifndef ML_H
//#define ML_H
//
//template<typename _type>
//class Adam {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta1_0 = 0.9;						// 1st order exponential decay starting parameter
//	double beta1 = 0.9;							// 1st order exponential decay
//	double beta2_0 = 0.99;						// 2nd order exponential decay starting parameter
//	double beta2 = 0.99;						// 2nd order exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//	cpx alpha = 0;								// true learning step
//	arma::Col<_type> m;							// moment vector
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~Adam() = default;
//	Adam() = default;
//	Adam(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->alpha = lr;
//		this->initialize();
//	};
//	Adam(double beta1, double beta2, double lr, double eps, uint size)
//		: beta1(beta1), beta2(beta2), lr(lr), eps(eps), size(size)
//	{
//		this->beta1_0 = beta1;
//		this->beta2_0 = beta2;
//		this->alpha = lr;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta1 = this->beta1_0;
//		this->beta2 = this->beta2_0;
//		this->m.zeros();
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->m = arma::Col<_type>(size, arma::fill::zeros);
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad) {
//		this->current_time += 1;
//		this->m = this->beta1_0 * this->m + (1.0 - this->beta1_0) * grad;
//		this->v = this->beta2_0 * this->v + (1.0 - this->beta2_0) * arma::square(grad);
//		// update decays
//		this->beta1 *= this->beta1_0;
//		this->beta2 *= this->beta2_0;
//
//		this->alpha = this->lr * (1.0 - this->beta2) / (1.0 - this->beta1);
//		// calculate the new gradient according to Adam
//		this->gradient = this->alpha * this->m / (arma::sqrt(this->v) + this->eps);
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//template<typename _type>
//class RMSprop_mod {
//private:
//	uint size;								// size of the gradient
//	uint current_time = 0;					// current iteration - starts from zero
//	double beta_0 = 0.9;						// exponential decay starting parameter
//	double beta = 0.9;							// exponential decay
//	double eps = 1e-8;							// prevent zero-division
//	double lr;									// learning step rate
//
//	arma::Col<_type> v;							// norm
//	arma::Col<_type> gradient;					// gradient
//public:
//	// ---------------------------
//	~RMSprop_mod() = default;
//	RMSprop_mod() = default;
//	RMSprop_mod(double lr, uint size)
//		: lr(lr), size(size)
//	{
//		this->beta_0 = 0.9;
//		this->beta = 0.9;
//		this->initialize();
//	};
//	RMSprop_mod(double beta, double lr, double eps, uint size)
//		: beta(beta), lr(lr), eps(eps), size(size)
//	{
//		this->beta_0 = beta;
//		this->initialize();
//	};
//	/*
//	* resets Adam
//	*/
//	void reset() {
//		this->current_time = 0;
//		this->beta = this->beta_0;
//		this->v.zeros();
//		this->gradient.zeros();
//	}
//	/*
//	* initialize Adam
//	*/
//	void initialize() {
//		// initialize to zeros
//		this->v = arma::Col<_type>(size, arma::fill::zeros);
//		this->gradient = arma::Col<_type>(size, arma::fill::zeros);
//	}
//
//	/*
//	* updates Adam
//	*/
//	void update(const arma::Col<_type>& grad, const arma::Col<_type>& O) {
//		this->current_time += 1;
//		this->v = this->beta * this->v + (1.0 - this->beta) * O % arma::conj(O);
//
//		//this->v.print("v");
//		// calculate the new gradient according to RMSProp arXiv:1910.11163v2
//		this->gradient = this->lr * grad / (arma::sqrt(this->v) + this->eps);
//		//this->gradient.print("grad");
//	};
//	/*
//	* get the gradient :)
//	*/
//	const arma::Col<_type>& get_grad()				const { return this->gradient; };
//	arma::Col<_type> get_grad_cpy()					const { return this->gradient; };
//};
//
//
//
//
//
//
//#endif