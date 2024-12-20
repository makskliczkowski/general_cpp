#pragma once
/*******************************
* Stores all the common functions
* used throughtly in the codes
* REV : 01/12/2023 - Maks Kliczkowski
*******************************/

#ifndef COMMON_H
#define COMMON_H

#if defined DEBUG and not defined _DEBUG
#	define _DEBUG
#endif
// #if defined _DEBUG and not defined DEBUG
// #	define DEBUG
// #endif 

// ########################################################				  ARMA					########################################################

#include "Include/statistical.h"

// ########################################################				  OTHER					########################################################

#include "Include/random.h"
#include "Include/maths.h"

#ifndef FLOG_H
#	include "flog.h"
#endif

#include <omp.h>
#include <thread>
#include <iomanip>
#include <iostream>
#include <limits>
#include <variant>

// ########################################################				 CONCEPTS				########################################################

template<class T> struct is_complex						: std::false_type	{};
template<class T> struct is_complex<std::complex<T>>	: std::true_type	{};

// ########################################################			    DEFINITIONS				########################################################

// using types
using cpx						=					std::complex<double>;
using uint						=					unsigned int;
using ul						=					unsigned long;
using ull						=					unsigned long long;
using u64						=					ull;
using ld						=					long double;

// constexpressions
constexpr long double PI		=					3.141592653589793238462643383279502884L;			// it is me, pi
constexpr long double TWOPI		=					2.0L * PI;											// it is me, 2pi
constexpr long double PIHALF	=					PI / 2.0L;											// it is me, half a pi
constexpr long double LOG_TWO	=					0.69314718055994530941723212145818L;
constexpr long double LOG_HALF	=					-LOG_TWO;
constexpr cpx I					=					cpx(0, 1);											// imaginary unit
const auto global_seed			=					std::random_device{}();								// global seed for classes

// end lines
#define EL std::endl
#define stout std::cout << std::setprecision(8) << std::fixed											// standard out
#define stoutc(c) if(c) stout <<  std::setprecision(8) << std::fixed	

// debug printers
#ifdef DEBUG
    #define stoutd(str) do { stout << str << EL } while(0)
    #define PRT(time_point, cond) do { stoutc(cond) << #cond << " -> time : " << tim_mus(time_point) << "mus" << EL; } while (0);
#else
    #define stoutd(str) do { break; } while (0)
    #define PRT(time_point, cond) do { } while (0)
#endif


// ##########################################################################################################################################

// ############################################################# U T I L I T Y ##############################################################

// ##########################################################################################################################################

#define SPACE_VEC(Lx, Ly, Lz, T) v_3d<T>(Lx, v_2d<T>(Ly, v_1d<T>(Lz)))

template<class T>
using v_Mat						=					v_1d<arma::Mat<T>>;									// 1d vector of arma::mat

// ##########################################################################################################################################

// ############################################################# V E C T O R S ##############################################################

// ##########################################################################################################################################

/*
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

/*
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

// ############################################################ VALUE EQUALS ################################################################

/*
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
#define EQ(value, equals) valueEqualsPrecision(value, equals)

// ##########################################################################################################################################

// ############################################################# C O M P L E X ##############################################################

// ##########################################################################################################################################

/*
* @brief Overloads printing to standard stream for complex numbers
*/
inline std::ostream& operator<< (std::ostream& out, const cpx v)
{
	auto prec = 1e-4;
	auto phase = std::arg(v) / PI;
	while (phase < 0) phase += 2.0;
	std::string absolute = "+" + STRP(std::abs(v), 2);
	std::string phase_str = "";
	if (EQP(phase, 0.0, prec) || EQP(phase, 2.0, prec))
		phase_str = "";
	else if (EQP(phase, 1.0, prec)) {
		absolute = "-" + STRP(std::abs(v), 2);
		phase_str = "";
	}
	else
		phase_str = "*exp(" + STRP(phase, 2) + "*pi*i)";
	out << absolute + phase_str;
	return out;
}


// ##########################################################################################################################################

// ############################################################# V A L U E S ! ##############################################################

// ##########################################################################################################################################

/*
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

// ##########################################################################################################################################

// ########################################################## B I N   S E A R C H ###########################################################

// ##########################################################################################################################################

/*
* @brief Finding index of base vector in mapping to reduced basis
* @typeparam T
* @param arr arary/vector conataing the mapping to the reduced basis
* @param l_point left maring for binary search
* @param r_point right margin for binary search
* @param element element to search in the array
* @returns -1 if not found else index of @ref element
*/
template <typename _T>
inline long long binarySearch(const v_1d<_T>& arr, ull l_point, ull r_point, _T elem) {
	if (l_point < 0 || r_point >= arr.size())
		return -1;

	if (l_point <= r_point) {
		ull middle = l_point + (r_point - l_point) / 2;												// find the middle point
		if (arr[middle] == elem) return middle;														// if found return
		else if (arr[middle] < elem) return binarySearch(arr, middle + 1, r_point, elem);			// else check the other boundaries
		else return binarySearch(arr, l_point, middle - 1, elem);
	}
	return -1;
}

template <>
inline long long binarySearch(const v_1d<double>& arr, ull l_point, ull r_point, double elem) {
	if (l_point < 0 || r_point >= arr.size())
		return -1;

	if (l_point <= r_point) {
		ull middle = l_point + (r_point - l_point) / 2;
		if (EQP(arr[middle], elem, 1e-12)) return middle;
		else if (arr[middle] < elem) return binarySearch(arr, middle + 1, r_point, elem);
		else return binarySearch(arr, l_point, middle - 1, elem);
	}
	return -1;
}


// ##########################################################################################################################################

// ############################################################ P R O G R E S S #############################################################

// ##########################################################################################################################################


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
		, percentageSteps(static_cast<int>(percentage * discreteSteps / 100.0))
	{
		// check if we can even make the progress bar
		if (percentage * discreteSteps < 100 || percentageSteps == 0)
		{
			this->percentage		=	100 / discreteSteps;
			this->percentageSteps	=	(int)std::ceil(this->percentage * discreteSteps / 100.0);
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
	double percentage = 34;																	// print percentage
	int percentageSteps = 1;
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

#endif // !COMMON_H
