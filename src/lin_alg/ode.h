#pragma once

	// ################################################################### ODE #########################################################################

	// #################################################################################################################################################

	namespace ODE
	{
		// #############################################################################################################################################

		template <typename _T, typename _CT>
		struct IVP_Functions {
			using fun_r_t 	= std::function<_CT(double, double, const _CT&)>;
			using fun_t 	= std::function<void(double, double, const _CT&, _CT&)>;
			using fun_jac_t = std::function<arma::Mat<_T>(const _CT&, const _CT&)>;
		};

		/**
		* @class IVP
		* @brief A template class for solving Initial Value Problems (IVP) for Ordinary Differential Equations (ODE).
		* 
		* This class provides an interface for implementing various ODE solvers. It defines the necessary function types
		* and methods required for single-step ODE solving and updating the state of the system.
		* 
		* @tparam Derived The derived class type.
		* @tparam _T The data type for numerical values, default is double.
		* @tparam _CT The container type for storing state vectors, default is arma::Col<_T>.
		* 
		* @note This class is intended to be used as a base class for specific ODE solver implementations.
		*/
		template <typename _T = double, typename _CT = arma::Col<_T>>
		class IVP
		{
		public:
			using fun_r_t 		= typename IVP_Functions<_T, _CT>::fun_r_t;											// function type - returns the derivative at time t
			using fun_t 		= typename IVP_Functions<_T, _CT>::fun_t;											// function type - updates the state
			using fun_jac_t 	= typename IVP_Functions<_T, _CT>::fun_jac_t;										// function type - returns the Jacobian matrix
		public:																								// _SINGLE STEP_
			virtual ~IVP() 		= default;																	// destructor
			// -----------------------------------------------------------------------------------------------------------------------------------------
			virtual void step(const fun_r_t& _f, double _t, double _h, const _CT& _y, _CT& _yout) = 0; 		// single step of the ODE solver - does not update the inner state
			virtual void step(const fun_t& _f, double _t, double _h, const _CT& _y, _CT& _yout) = 0;	 	// single step of the ODE solver - does not update the inner state
			
			_CT step(const fun_r_t& _f, double _t, double _h, const _CT& _y) {
				_CT yout;
				step(_f, _t, _h, _y, yout);
				return yout;
			}
			
			_CT step(const fun_t& _f, double _t, double _h, const _CT& _y) {
				_CT yout;
				step(_f, _t, _h, _y, yout);
				return yout;
			}
			// -----------------------------------------------------------------------------------------------------------------------------------------
			virtual void update(_CT& _y, double _h) = 0; 													// update the inner state
			virtual _CT update(const _CT& _y, double _h) = 0; 												// update the inner state
			virtual double dt(double _h, uint i) const = 0; 												// get the timestep
			const uint getOrder() 											const { return 1; }

			// -----------------------------------------------------------------------------------------------------------------------------------------
		};

		// #############################################################################################################################################

		template <typename Derived, uint order = 1, typename _T = double, typename _CT = arma::Col<_T>>
		class RK_Base : public IVP<_T, _CT>
		{
		public:
			using fun_r_t 	= typename IVP_Functions<_T, _CT>::fun_r_t;
			using fun_t 	= typename IVP_Functions<_T, _CT>::fun_t;
			using fun_jac_t = typename IVP_Functions<_T, _CT>::fun_jac_t;
		protected:
			v_1d<_CT> k_;					// k values of the RK method
			_CT kout_;						// helper for returning the k values
			v_1d<double> coefficients_;		// coefficients of the RK method - for single step
			v_1d<double> timesteps_;		// timesteps of the RK method (for single step)	- multipliers of the timestep
			uint order_ = order;			// order of the RK method
		public:
			// -----------------------------------------------------------------------------------------------------------------------------------------
			RK_Base() 
			{
				k_.resize(order);
				coefficients_.resize(order);
				timesteps_.resize(order);
			}

			// -----------------------------------------------------------------------------------------------------------------------------------------
			void step(const fun_r_t& _f, double _t, double _h, const _CT& _y, _CT& _yout) override {
				static_cast<Derived*>(this)->step_impl(_f, _t, _h, _y, _yout);
			}

			void step(const fun_t& _f, double _t, double _h, const _CT& _y, _CT& _yout) override {
				static_cast<Derived*>(this)->step_impl(_f, _t, _h, _y, _yout);
			}

			_CT step(const fun_r_t& _f, double _t, double _h, const _CT& _y) {
				return static_cast<Derived*>(this)->step_impl(_f, _t, _h, _y);
			}

			_CT step(const fun_t& _f, double _t, double _h, const _CT& _y) {
				return static_cast<Derived*>(this)->step_impl(_f, _t, _h, _y);
			}

			// -----------------------------------------------------------------------------------------------------------------------------------------
			virtual void update(_CT& _y, double _h) override;
			virtual _CT update(const _CT& _y, double _h) override;

			double dt(double _h, uint i) const override						{ return _h * this->timesteps_[i]; }
			// -----------------------------------------------------------------------------------------------------------------------------------------
			// getters
			v_1d<double> getCoefficients() 									const { return this->coefficients_; };
			v_1d<double> getTimesteps() 									const { return this->timesteps_; };
			const uint getOrder() 											const { return this->order_; }
			// set
			void setCoefficients(const v_1d<double>& _coefficients) 		{ if(_coefficients.size() == this->order_) this->coefficients_ = _coefficients; }
			void setTimesteps(const v_1d<double>& _timesteps) 				{ if(_timesteps.size() == this->order_) this->timesteps_ = _timesteps; }
			// -----------------------------------------------------------------------------------------------------------------------------------------
		};

		/**
        * @brief Updates the value of _y using the Runge-Kutta method.
        *
        * This function updates the value of _y by iterating over the coefficients
        * and performing the necessary calculations with the k_ array.
        *
		* @tparam Derived The derived class of the RK_Base class.
        * @tparam _order The order of the Runge-Kutta method.
        * @tparam _T The data type of the coefficients.
        * @tparam _CT The data type of the value to be updated.
        * @param _y The value to be updated.
        * @note This function assumes that the k_ array has been properly initialized and does not check for its size.
        */
        template <typename Derived, uint _order, typename _T, typename _CT>
        inline void RK_Base<Derived, _order, _T, _CT>::update(_CT& _y, double _h)
        {
			// const auto _dt = this->dt(_h, this->order_ - 1);				// calculate the time step from the last coefficient
            for (int _c = 0; _c < this->order_; _c++)						// iterate over the coefficients and update the value
                _y += (this->coefficients_[_c] * _h) * this->k_[_c];		// update the value using the coefficients and the k_ array
        }

		/**
		* @brief Updates the solution using Runge-Kutta method
		* 
		* Performs one step of the Runge-Kutta integration by combining previous stage
		* values (k_) with their corresponding coefficients. The method calculates
		* the weighted sum of stage values and adds it to the current solution.
		* 
		* @param _y Current value/state of the system
		* @param _h Step size for the integration
		* @return _CT Updated value/state after one RK step
		* 
		* @details The method:
		* 1. Calculates the time step using dt()
		* 2. Initializes output with current state
		* 3. Adds weighted contributions from each RK stage
		* 4. Returns the final updated state
		* 
		* @note Part of RK_Base template class implementation for Runge-Kutta methods
		*/
		template <typename Derived, uint _order, typename _T, typename _CT>
		inline _CT RK_Base<Derived, _order, _T, _CT>::update(const _CT& _y, double _h)
		{
			// const auto _dt 	= this->dt(_h, this->order_ - 1);				// calculate the time step from the last coefficient
			this->kout_ 	= _y;											// initialize the output with the current state
			for (int _c = 0; _c < this->order_; _c++)
				this->kout_ += (_h * this->coefficients_[_c]) * this->k_[_c];
			return _y + this->kout_;										// return the updated state
		}

		// #############################################################################################################################################

		template <uint _order, typename _T = double, typename _CT = arma::Col<_T>>
		class RK : public RK_Base<RK<_order, _T, _CT>, _order, _T, _CT>
		{
		public:
			using fun_r_t 	= typename IVP_Functions<_T, _CT>::fun_r_t;
			using fun_t 	= typename IVP_Functions<_T, _CT>::fun_t;
			using fun_jac_t = typename IVP_Functions<_T, _CT>::fun_jac_t;
		public:
			// -----------------------------------------------------------------------------------------------------------------------------------------
			RK() : RK_Base<RK<_order, _T, _CT>, _order, _T, _CT>()
			{
				if constexpr (_order == 1) {
					this->coefficients_ = {1.0};
					this->timesteps_ 	= {1.0};
				} else if constexpr (_order == 2) {
					this->coefficients_ = {0.5, 0.5};
					this->timesteps_ 	= {0.5, 0.5};
				} else if constexpr (_order == 4) {
					this->coefficients_ = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
					this->timesteps_ 	= {0.5, 0.5, 1.0, 1.0};
				}
			};
			// -----------------------------------------------------------------------------------------------------------------------------------------
			void step_impl(const fun_r_t& _f, double _t, double _h, const _CT& _y, _CT& _yout);
			void step_impl(const fun_t& _f, double _t, double _h, const _CT& _y, _CT& _yout);
			_CT step_impl(const fun_r_t& _f, double _t, double _h, const _CT& _y);
			_CT step_impl(const fun_t& _f, double _t, double _h, const _CT& _y);
			// -----------------------------------------------------------------------------------------------------------------------------------------
			// Adaptive step-size control
			void adaptive_step(const fun_r_t& _f, double& _t, double& _h, _CT& _y, double _tol);
			void adaptive_step(const fun_t& _f, double& _t, double& _h, _CT& _y, double _tol);
		};

		// Adaptive step-size control implementation
		// !TODO: Implement adaptive step-size control for the Runge-Kutta solver - check the error and adjust the step size
		// !TODO: Create tests for the adaptive step-size control
		template <uint _order, typename _T, typename _CT>
		inline void RK<_order, _T, _CT>::adaptive_step(const fun_r_t& _f, double& _t, double& _h, _CT& _y, double _tol)
		{
			_CT y_temp, y_err;
			double h_new, err;
			do {
				step_impl(_f, _t, _h, _y, y_temp);
				step_impl(_f, _t, _h / 2, _y, y_err);
				step_impl(_f, _t + _h / 2, _h / 2, y_err, y_err);
				y_err = y_temp - y_err;
				err = arma::norm(y_err, "inf");
				h_new = _h * std::pow(_tol / err, 1.0 / (_order + 1));
				if (err > _tol) {
					_h = h_new;
				}
			} while (err > _tol);
			_y = y_temp;
			_t += _h;
			_h = h_new;
		}


		// #############################################################################################################################################

		enum class ODE_Solvers 
		{
			Euler,
			Heun,
			RK2,
			RK4 		= 4,
			RKAdaptive 	= 5
		};

		/**
		* @brief Factory function to create a Runge-Kutta solver of a specified order.
		* 
		* This function creates and returns a pointer to a Runge-Kutta solver object
		* of the specified order. The supported orders are 1, 2, and 4. If an unsupported
		* order is provided, the function throws an invalid_argument exception.
		* 
		* @tparam _T The data type used for the solver (default is double).
		* @tparam _CT The container type used for the solver (default is arma::Col<_T>).
		* @param order The order of the Runge-Kutta solver to create.
		* @return IVP<_T, _CT>* Pointer to the created Runge-Kutta solver.
		* @throws std::invalid_argument If the specified order is not supported.
		*/
		template <typename _T = double, typename _CT = arma::Col<_T>>
		inline IVP<_T, _CT>* createRKsolver(ODE_Solvers solverType, double _tol = 1e-10)
		{
			switch (solverType) {
				case ODE_Solvers::Euler:
					return new RK<1, _T, _CT>();
				case ODE_Solvers::Heun:
				case ODE_Solvers::RK2:
					return new RK<2, _T, _CT>();
				case ODE_Solvers::RK4:
					return new RK<4, _T, _CT>();
				case ODE_Solvers::RKAdaptive:
					throw std::runtime_error("Adaptive Runge-Kutta solver not implemented yet. This is work in progress.");
					break;
				default:
					throw std::invalid_argument("Unsupported Runge-Kutta solver type. Supported types are Euler, Heun, RK2, RK4, and RKAdaptive.");
			}
		}

		// #############################################################################################################################################
	};

