/******************************************************************************
 *
 *  @file src/common.h
 *  @brief Common utilities, type aliases, and definitions used throughout the project.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#include <thread>
#include <limits>
#include <iomanip>
#include <iostream>
#include <variant>
#include <random>
#include <complex>
#include <vector>

#ifndef COMMON_H
#define COMMON_H

#if defined DEBUG and not defined _DEBUG
#	define _DEBUG
#endif

// #########################################################################
// STL utilities and type aliases
// #########################################################################


#include "common/str.h"
#include "common/signatures.h"
#include "common/exceptions.h"
#include "common/time.h"
#include "common/algorithms.h"

// Mathematical utilities and constants
#include "maths/maths.h"

// Slurm utilities for job management and parallel execution
#include "runtime/slurm.h"

// -------------------------------------------------------------------------
// Logging utilities
// -------------------------------------------------------------------------
#ifndef FLOG_H
#	include "common/flog.h"
#endif

// -------------------------------------------------------------------------
#ifndef __APPLE__
#	include <omp.h>
#endif

// #########################################################################
// Project-specific type aliases and utilities

template<class T> struct is_complex						: std::false_type	{};
template<class T> struct is_complex<std::complex<T>>	: std::true_type	{};
template <typename> constexpr bool always_false 		= false;

// using types
using cpx	= std::complex<double>;
using uint	= unsigned int;
using ul	= unsigned long;
using ull	= unsigned long long;
using u64	= ull;
using ld	= long double;

// using types
template <typename _T, typename _U = std::allocator<std::shared_ptr<_T>>>
using v_sp_t = std::vector<std::shared_ptr<_T>, _U>;							// vector of shared pointers
template <typename _T, typename _U = std::allocator<std::unique_ptr<_T>>>
using v_up_t = std::vector<std::unique_ptr<_T>, _U>;							// vector of unique pointers
template <typename _T, typename _U = std::allocator<_T*>>
using v_p_t = std::vector<_T*, _U>; 											// vector of pointers

// constexpressions

// Mathematical constants
constexpr long double PI		= 3.141592653589793238462643383279502884L;		// it is me, pi
constexpr long double TWOPI		= 2.0L * PI;									// it is me, 2pi
constexpr long double PIHALF	= PI / 2.0L;									// it is me, half a pi
constexpr long double LOG_TWO	= 0.69314718055994530941723212145818L;
constexpr long double LOG_HALF	= -LOG_TWO;

// imaginary unit
constexpr cpx I					= cpx(0, 1);									// imaginary unit

// global random seed for classes that need it
inline const auto global_seed	= std::random_device{}();						// global seed for classes

// end lines
#define EL std::endl
#define stout std::cout << std::setprecision(8) << std::fixed									// standard out
#define stoutc(c) if(c) stout <<  std::setprecision(8) << std::fixed	

// debug printers
#ifdef DEBUG
    #define stoutd(str) do { stout << str << EL; } while(0)
    #define PRT(time_point, cond) do { stoutc(cond) << #cond << " -> time : " << tim_mus(time_point) << "mus" << EL; } while (0)
#else
    #define stoutd(str) do { } while (0)
    #define PRT(time_point, cond) do { } while (0)
#endif


// #########################################################################
// UTILITY FUNCTIONS
// #########################################################################

#define SPACE_VEC(Lx, Ly, Lz, T) v_3d<T>(Lx, v_2d<T>(Ly, v_1d<T>(Lz)))

// #########################################################################
// VECTOR PRINTING
// #########################################################################

/**
*@brief Overwritten standard stream redirection operator for 2D vectors separated by commas
*@param out outstream to be used
*@param v 1D vector
*/
template <typename T>
inline std::ostream& operator<< (std::ostream& out, const v_1d<T>& v) {
	if (!v.empty()) 
	{
		for (int i = 0; i < v.size(); i++)
			out << STRP(v[i], 10) << ",";
		out << "\b"; 
		// use two ANSI backspace characters '\b' to overwrite final ", "
	}
	return out;
}

/**
* @brief Overwritten standard stream redirection operator for 2D vectors
* @param out outstream to be used
* @param v 2D vector
*/
template <typename T>
inline std::ostream& operator<< (std::ostream& out, const v_2d<T>& v) {
	if (!v.empty())
		for (auto it : v)
			out << "\t" << it << EL;
	return out;
}

// #########################################################################
// Value printing and comparison utilities
// #########################################################################

/**
* @brief Checks if value is equal to some param up to given tolerance
*/
template <typename _T1, typename _T2, typename _T3>
inline auto valueEqualsPrecision(_T1 value, _T2 equals, _T3 tolerance) RETURNS(std::abs(value - equals) <= tolerance);
template <typename _T1, typename _T2>
inline auto valueEqualsPrecision(_T1 value, _T2 equals) RETURNS(value == equals);

#define VEQ(name)					valueEquals(#name,(name)	, 2)
#define VEQS(name)					valueEqualsS(#name,(name)	, 2)
#define VEQV(name,val)				valueEquals(#name,(val)		, 2)
#define VEQVS(name,val)				valueEqualsS(#name,(val)	, 2)
#define VEQP(name,prec)				valueEquals(#name,(name)	, prec)
#define VEQPS(name,prec)			valueEqualsS(#name,(name)	, prec)
#define VEQVP(name,val,prec)		valueEquals(#name,(val)		, prec)
#define EQP(value, equals, prec)	valueEqualsPrecision(value, equals, prec)
#define EQ(value, equals) 			valueEqualsPrecision(value, equals)

// #########################################################################
// Complex number printing
// #########################################################################

/**
* @brief Overloads printing to standard stream for complex numbers
*/
std::ostream& operator<< (std::ostream& out, const cpx v);

// #########################################################################
// Value comparison and printing utilities
// #########################################################################

/**
* @brief Given the char* name it prints its value in a format "name=val"
* @param name name of the variable
* @param value value of the variable
* @returns "name=val" string
*/
template <typename T>
inline auto valueEquals(const char name[], T value, int prec = 2)			RETURNS(std::string(name) + "=" + str_p(value, prec));
template <typename T>
inline auto valueEqualsS(const char name[], T value, int prec = 2)			RETURNS(std::string(name) + "=" + str_p(value, prec, true));
inline auto valueEquals(const char name[], std::string value, int prec)		RETURNS(std::string(name) + "=" + value);

// #########################################################################

// #########################################################################
// Progress bar implementation
// #########################################################################

#ifndef PROGRESS_H
#define PROGRESS_H
#include <mutex>

class pBar 
{
public:
	std::mutex _mutex;
	void update(double newProgress);
	void print();
	void printWithTime(std::string message);
	~pBar()							=		default;
	pBar() : timer(NOW) 
	{ 
		this->currUpdateVal		= 0;
		this->currentProgress	= 0;
		this->amountOfFiller	= 0;
	};
	pBar(const pBar& other)
		: timer(other.timer), percentage(other.percentage), percentageSteps(other.percentageSteps)

	{ 
		this->currUpdateVal		= 0;
		this->currentProgress	= 0;
		this->amountOfFiller	= 0;
	};
	pBar(double percentage, int discreteSteps, clk::time_point _time = NOW)
		: timer(_time)
		, percentage(percentage)
		, percentageSteps((int)std::ceil(percentage * discreteSteps / 100.0))
	{
		// check if we can even make the progress bar
		if (percentage * discreteSteps < 100 || percentageSteps == 0)
		{
			this->percentage		=	100.0 / discreteSteps;
			this->percentageSteps	=	(int)std::ceil(this->percentage * discreteSteps / 100.0);
		}
		else {
			this->percentage		= 100 * (double)percentageSteps / discreteSteps;
		}
		this->currUpdateVal		= 0;
		this->currentProgress	= 0;
		this->amountOfFiller	= 0;
		this->update(percentage);
	};

	pBar& operator=(const pBar& other)
	{
		this->timer				= other.timer;
		this->percentage		= other.percentage;
		this->percentageSteps	= other.percentageSteps;
		this->currUpdateVal		= other.currUpdateVal;
		this->currentProgress	= other.currentProgress;
		this->amountOfFiller	= other.amountOfFiller;
		return *this;
	}
protected:
	// --------------------------- STRING ENDS
	std::string startingTabs		=		"\t\t\t\t";
	std::string firstPartOfpBar		=		startingTabs + "[";
	std::string lastPartOfpBar		=		"]";
	std::string pBarFiller			=		"|";
	std::string pBarUpdater			=		"|\\/";
	// --------------------------- PROGRESS
	clk::time_point timer;														            // inner clock
	int amountOfFiller		= 0;															// length of filled elements
	int pBarLength			= 50;														    // length of a progress bar
	int currUpdateVal		= 0;														    // current value of the updated progress
	double currentProgress	= 0;													        // current progress
	double neededProgress	= 100;												            // final progress
public:
	auto get_start_time()			const	{ return this->timer; };
	double percentage 		= 34;															// print percentage
	int percentageSteps 	= 1;
};

#define PROGRESS_UPD(X, PBAR, TEXT)		BEGIN_CATCH_HANDLER{								\
											if (X % PBAR.percentageSteps == 0)				\
												PBAR.printWithTime(LOG_LVL1 + SSTR(TEXT));}	\
										END_CATCH_HANDLER("Couldn't print progress: ", ;)		
#define PROGRESS_UPD_DO(X, PBAR, TXT, D)BEGIN_CATCH_HANDLER{								\
											if (X % PBAR.percentageSteps == 0)				\
												PBAR.printWithTime(LOG_LVL1 + SSTR(TXT));	\
												D;										}	\
										END_CATCH_HANDLER("Couldn't print progress: ", ;)		
#define PROGRESS_UPD_Q(X, PBAR, TEXT, Q)if(Q){												\
										BEGIN_CATCH_HANDLER{								\
											if (X % PBAR.percentageSteps == 0)				\
												PBAR.printWithTime(LOG_LVL1 + SSTR(TEXT));}	\
										END_CATCH_HANDLER("Couldn't print progress: ", ;)}	\

#endif // !PROGRESS_H

// ##########################################################################################################################################

// ############################################################# O T H E R S ! ##############################################################

// ##########################################################################################################################################

/*
* @brief Allows to visit the variant and get the type of the variant
*/
template<class V>
std::type_info const& var_type(V const& v)
{
	return std::visit( [](auto&&x)->decltype(auto){ return typeid(x); }, v );
}

// #########################################################################

#endif // !COMMON_H

// -------------------------------------------------------------------------
//! EOF
// -------------------------------------------------------------------------