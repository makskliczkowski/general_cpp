#include "../../src/lin_alg.h"
#include <complex>

// #################################################################################################################################

namespace algebra
{
    namespace ODE
    {
        // #########################################################################################################################

        template<typename _T, typename _CT>
        using fun_r_t = IVP_Functions<_T, _CT>::fun_r_t;
        using fun_r_t_col_dd = fun_r_t<double, arma::Col<double>>;
        using fun_r_t_col_cc = fun_r_t<std::complex<double>, arma::Col<std::complex<double>>>;

        template<typename _T, typename _CT>
        using fun_t = IVP_Functions<_T, _CT>::fun_t;
        using fun_t_col_dd = fun_t<double, arma::Col<double>>;
        using fun_t_col_cc = fun_t<std::complex<double>, arma::Col<std::complex<double>>>;

        template<typename _T, typename _CT>
        using fun_jac_t = IVP_Functions<_T, _CT>::fun_jac_t;

        // #########################################################################################################################

        /**
        * @brief Implements a single step of the Runge-Kutta method for solving ordinary differential equations (ODEs).
        *
        * @tparam _order The order of the Runge-Kutta method (e.g., 1 for RK1, 2 for RK2, 4 for RK4).
        * @tparam _T The type of the time variable.
        * @tparam _CT The type of the container holding the state variables.
        * @param _f The function representing the ODE system. It takes the current time, an integer index, and the current state as arguments.
        * @param _t The current time.
        * @param _h The time step size.
        * @param _y The current state of the system.
        * @param _yout The output state after taking the step.
        *
        * @throws std::runtime_error If the order of the Runge-Kutta method is not supported.
        */
        template <uint _order, typename _T, typename _CT>
        void RK<_order, _T, _CT>::step_impl(const fun_r_t& _f, double _t, double _h, const _CT& _y, _CT& _yout)
        {
            _yout = _y;
            if constexpr (_order == 1)
            {
            this->k_[0] = _f(_t, 0, _y);
            }
            else if constexpr (_order == 2)
            {
            this->k_[0] = _f(_t, 0, _y);
            this->k_[1] = _f(_t + this->dt(_h, 1), 0, _y + this->dt(_h, 1) * this->k_[0]);
            }
            else if constexpr (_order == 4)
            {
            this->k_[0] = _f(_t, 0, _y);
            this->k_[1] = _f(_t + this->dt(_h, 1), 0, _y + this->dt(_h, 1) * this->k_[0]);
            this->k_[2] = _f(_t + this->dt(_h, 2), 0, _y + this->dt(_h, 2) * this->k_[1]);
            this->k_[3] = _f(_t + this->dt(_h, 3), 0, _y + this->dt(_h, 3) * this->k_[2]);
            }
            else
            {
            throw std::runtime_error("Invalid order for the Runge-Kutta method.");
            }
            this->update(_yout, _h);
        }

        // template instantiation of the function above
        template void RK<1, double, arma::Col<double>>::step_impl(const fun_r_t_col_dd&, double, double, const arma::Col<double>&, arma::Col<double>&);
        template void RK<1, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_r_t_col_cc&, double, double, const arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&);
        template void RK<2, double, arma::Col<double>>::step_impl(const fun_r_t_col_dd&, double, double, const arma::Col<double>&, arma::Col<double>&);
        template void RK<2, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_r_t_col_cc&, double, double, const arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&);
        template void RK<4, double, arma::Col<double>>::step_impl(const fun_r_t_col_dd&, double, double, const arma::Col<double>&, arma::Col<double>&);
        template void RK<4, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_r_t_col_cc&, double, double, const arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&);
        // #########################################################################################################################

        template <uint _order, typename _T, typename _CT>
        void RK<_order, _T, _CT>::step_impl(const fun_t& _f, double _t, double _h, const _CT& _y, _CT& _yout)
        {
            _yout = _y;
            if constexpr (_order == 1)
            {
            _f(_t, this->dt(_h, 0), _y, this->k_[0]);
            }
            else if constexpr (_order == 2)
            {
            _f(_t, this->dt(_h, 0), _y, this->k_[0]);
            _f(_t + this->dt(_h, 1), this->dt(_h, 1), _y + this->dt(_h, 1) * this->k_[0], this->k_[1]);
            }
            else if constexpr (_order == 4)
            {
            _f(_t, this->dt(_h, 0), _y, this->k_[0]);
            _f(_t + this->dt(_h, 1), this->dt(_h, 1), _y + this->dt(_h, 1) * this->k_[0], this->k_[1]);
            _f(_t + this->dt(_h, 2), this->dt(_h, 2), _y + this->dt(_h, 2) * this->k_[1], this->k_[2]);
            _f(_t + this->dt(_h, 3), this->dt(_h, 3), _y + this->dt(_h, 3) * this->k_[2], this->k_[3]);
            }
            else
            {
            throw std::runtime_error("Invalid order for the Runge-Kutta method.");
            }
            this->update(_yout, _h);
        }
        // template instantiation of the function above
        template void RK<1, double, arma::Col<double>>::step_impl(const fun_t_col_dd&, double, double, const arma::Col<double>&, arma::Col<double>&);
        template void RK<1, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_t_col_cc&, double, double, const arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&);
        template void RK<2, double, arma::Col<double>>::step_impl(const fun_t_col_dd&, double, double, const arma::Col<double>&, arma::Col<double>&);
        template void RK<2, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_t_col_cc&, double, double, const arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&);
        template void RK<4, double, arma::Col<double>>::step_impl(const fun_t_col_dd&, double, double, const arma::Col<double>&, arma::Col<double>&);
        template void RK<4, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_t_col_cc&, double, double, const arma::Col<std::complex<double>>&, arma::Col<std::complex<double>>&);
        // #########################################################################################################################

        template <uint _order, typename _T, typename _CT>
        _CT RK<_order, _T, _CT>::step_impl(const fun_r_t& _f, double _t, double _h, const _CT& _y)
        {
            if constexpr (_order == 1)
            {
                this->k_[0] = _f(_t, this->dt(_h, 0), _y);
            }
            else if constexpr (_order == 2)
            {
                this->k_[0] = _f(_t, this->dt(_h, 0), _y);
                this->k_[1] = _f(_t + this->dt(_h, 1), this->dt(_h, 1), _y + this->dt(_h, 1) * this->k_[0]);
            }
            else if constexpr (_order == 4)
            {
                this->k_[0] = _f(_t, this->dt(_h, 0), _y);
                this->k_[1] = _f(_t + this->dt(_h, 1), this->dt(_h, 1), _y + this->dt(_h, 1) * this->k_[0]);
                this->k_[2] = _f(_t + this->dt(_h, 2), this->dt(_h, 2), _y + this->dt(_h, 2) * this->k_[1]);
                this->k_[3] = _f(_t + this->dt(_h, 3), this->dt(_h, 3), _y + this->dt(_h, 3) * this->k_[2]);
            }
            else
                throw std::runtime_error("Invalid order for the Runge-Kutta method.");
            return this->update(_y, _h);
        }
        // template instantiation of the function above
        template arma::Col<double> RK<1, double, arma::Col<double>>::step_impl(const fun_r_t_col_dd&, double, double, const arma::Col<double>&);
        template arma::Col<std::complex<double>> RK<1, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_r_t_col_cc&, double, double, const arma::Col<std::complex<double>>&);
        template arma::Col<double> RK<2, double, arma::Col<double>>::step_impl(const fun_r_t_col_dd&, double, double, const arma::Col<double>&);
        template arma::Col<std::complex<double>> RK<2, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_r_t_col_cc&, double, double, const arma::Col<std::complex<double>>&);
        template arma::Col<double> RK<4, double, arma::Col<double>>::step_impl(const fun_r_t_col_dd&, double, double, const arma::Col<double>&);
        template arma::Col<std::complex<double>> RK<4, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_r_t_col_cc&, double, double, const arma::Col<std::complex<double>>&);
        // #########################################################################################################################

        template <uint _order, typename _T, typename _CT>
        _CT RK<_order, _T, _CT>::step_impl(const fun_t& _f, double _t, double _h, const _CT& _y)
        {
            if constexpr (_order == 1)
            {
                _f(_t, this->dt(_h, 0), _y, this->k_[0]);
            }
            else if constexpr (_order == 2)
            {
                _f(_t, this->dt(_h, 0), _y, this->k_[0]);
                _f(_t + this->dt(_h, 1), this->dt(_h, 1), _y + this->dt(_h, 1) * this->k_[0], this->k_[1]);
            }
            else if constexpr (_order == 4)
            {
                _f(_t, this->dt(_h, 0), _y, this->k_[0]);
                _f(_t + this->dt(_h, 1), this->dt(_h, 1), _y + this->dt(_h, 1) * this->k_[0], this->k_[1]);
                _f(_t + this->dt(_h, 2), this->dt(_h, 2), _y + this->dt(_h, 2) * this->k_[1], this->k_[2]);
                _f(_t + this->dt(_h, 3), this->dt(_h, 3), _y + this->dt(_h, 3) * this->k_[2], this->k_[3]);
            }
            else
                throw std::runtime_error("Invalid order for the Runge-Kutta method.");
            return this->update(_y, _h);
        }
        // template instantiation of the function above
        template arma::Col<double> RK<1, double, arma::Col<double>>::step_impl(const fun_t_col_dd&, double, double, const arma::Col<double>&);
        template arma::Col<std::complex<double>> RK<1, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_t_col_cc&, double, double, const arma::Col<std::complex<double>>&);
        template arma::Col<double> RK<2, double, arma::Col<double>>::step_impl(const fun_t_col_dd&, double, double, const arma::Col<double>&);
        template arma::Col<std::complex<double>> RK<2, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_t_col_cc&, double, double, const arma::Col<std::complex<double>>&);
        template arma::Col<double> RK<4, double, arma::Col<double>>::step_impl(const fun_t_col_dd&, double, double, const arma::Col<double>&);
        template arma::Col<std::complex<double>> RK<4, std::complex<double>, arma::Col<std::complex<double>>>::step_impl(const fun_t_col_cc&, double, double, const arma::Col<std::complex<double>>&);
        // #########################################################################################################################
    };
    // #############################################################################################################################
};

// #################################################################################################################################