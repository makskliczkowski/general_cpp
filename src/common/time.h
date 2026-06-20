/******************************************************************************
 *
 *  @file src/Include/time.h
 *  @brief High-resolution timer, scoped timing, and time formatting.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#include <chrono>
#include <iostream>
#include <algorithm>
#include <string>
#include <time.h>
#include <type_traits>
#include "exceptions.h"
#include "str.h"

// ########################################################	T I M E   F U N C T I O N S ########################################################
#if defined (_MSC_VER)
	using clk					=				std::chrono::steady_clock;
#elif defined (__linux__)
	using clk					=				std::chrono::system_clock;
#else
	using clk 					=				std::chrono::steady_clock;
#endif

using clkS						=				std::chrono::system_clock;
#define DUR										std::chrono::duration
#define DURCAST									std::chrono::duration_cast
#define NOW										std::chrono::high_resolution_clock::now()	    			
#define DURATION(t1, t2)						static_cast<long double>(DURCAST<std::chrono::microseconds>(DUR(t1 - t2)).count())
#define DURATIONS(t1, t2)						static_cast<long double>(DURCAST<std::chrono::seconds>(DUR(t1 - t2)).count())
#define DURATIONMS(t1, t2)						static_cast<long double>(DURCAST<std::chrono::milliseconds>(DUR(t1 - t2)).count())
#define DURATIONMUS(t1, t2)						DURATION(t1, t2)

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_s(clk::time_point start)			RETURNS(DURATIONS(NOW, start));
inline auto t_s(clk::time_point start, 
				clk::time_point end)			RETURNS(DURATIONS(end, start));
inline auto TS(clk::time_point start)			-> std::string { return STRP(t_s(start), 3) + " s";				};
inline auto TS(clk::time_point start,
				clk::time_point end)			-> std::string { return STRP(t_s(start, end), 3) + " s";		};
/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_ms(clk::time_point start)			RETURNS(DURATIONMS(NOW, start));
inline auto t_ms(clk::time_point start, 
				 clk::time_point end)			RETURNS(DURATIONMS(end, start));
inline auto TMS(clk::time_point start)			-> std::string { return STRP(t_ms(start), 3) + " ms";			};
inline auto TMS(clk::time_point start,	
				clk::time_point end)			-> std::string { return STRP(t_ms(start, end), 3) + " ms";		};
/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_mus(clk::time_point start)		RETURNS(DURATION(NOW, start));
inline auto t_mus(clk::time_point start,
				clk::time_point end)			RETURNS(DURATION(end, start));
inline auto TMUS(clk::time_point start)			-> std::string { return STRP(t_mus(start), 3) + " mus";			};
inline auto TMUS(clk::time_point start,
				clk::time_point end)			-> std::string { return STRP(t_mus(start, end), 3) + " mus"; 	};

#define stouts(text, start)						stout	<< text <<	" -> time : " << tim_s(start)	<< " s"		<< EL					
#define stoutms(text, start)					stout	<< text <<	" -> time : " << tim_ms(start)	<< " ms"	<< EL			
#define stoutmus(text, start)					stout	<< text <<	" -> time : " << tim_mus(start) << " mus"	<< EL	

constexpr auto PRETTY_TIME_FORMAT				= "%Y-%m-%d_%H:%M:%S"; 	// strftime format
constexpr auto PRETTY_TIME_FORMAT_MS			= ".%03Id"; 			// printf format

// ##################################################################################################################################

// ######################################################## T I M E R ###############################################################

// ##################################################################################################################################

#include <map>
class Timer
{
public:
	enum class TimePrecision { MICROSECONDS = 0, MILLISECONDS = 1, SECONDS = 2 };
protected:
	const static inline std::string _startName 	= "start";
	std::string _last 							= "start";
	size_t _iter 								= 0;
	clk::time_point	_start;
	std::vector<clk::time_point> _timestamps;
	std::map<std::string, size_t> _timestampNames;

public:
	
	// ########### C O N S T R C T ###########

	Timer()										{ this->reset(); };
	void reset(clk::time_point _t = NOW);

	// ############# S E T T E R S #############

	void checkpoint(const std::string& _name);

	// ############# G E T T E R S #############

	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	clk::time_point point(_T1 _idx);

	clk::time_point point(const std::string& _name);
	std::vector<clk::time_point> point() const;
	clk::time_point start() const;
	clk::time_point end() const;

	// ############# E L A P S E D #############

	template <typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type, typename _R = std::string>
	_R elapsed(_T1 _point, _T1 _start = 0, TimePrecision _prec = TimePrecision::MICROSECONDS);
	template <typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type, typename _R = std::string>
	_R elapsed(_T1 _point, TimePrecision _prec = TimePrecision::MICROSECONDS);

	template <typename _R = std::string>
	_R elapsed(const std::string& _point, const std::string& _since, TimePrecision _prec = TimePrecision::MICROSECONDS);
	template <typename _R = std::string>
	_R elapsed(const std::string& _point, TimePrecision _prec = TimePrecision::MICROSECONDS);
};

// ##################################################################################################################################

std::string prettyTime(std::time_t now = std::time(0));

// ##################################################################################################################################

// ######################################################## S C O P E D   T I M E R #################################################

// ##################################################################################################################################

/*
* @brief RAII scope timer: times the lifetime of the object and, on
* destruction, either reports the elapsed time to stdout (with a label) or
* writes it into a caller-owned sink. The C++ equivalent of the Python
* Timer used as a context manager.
*/
class ScopedTimer
{
public:
	// report to stdout on destruction
	explicit ScopedTimer(std::string _label)
		: _label(std::move(_label)), _start(NOW) {}
	// write the elapsed seconds into _sink on destruction (no printing)
	explicit ScopedTimer(long double* _sink)
		: _sink(_sink), _start(NOW) {}

	ScopedTimer(const ScopedTimer&)				= delete;
	ScopedTimer& operator=(const ScopedTimer&)	= delete;

	// elapsed seconds so far, without ending the scope
	[[nodiscard]] long double seconds() const	{ return t_s(this->_start); }

	~ScopedTimer()
	{
		const long double _elapsed = t_s(this->_start);
		if (this->_sink)
			*this->_sink = _elapsed;
		else
			std::cout << this->_label << " -> time : " << _elapsed << " s" << std::endl;
	}

private:
	std::string			_label;
	long double*		_sink	= nullptr;
	clk::time_point		_start;
};

#include <functional>
#include <utility>
#include <type_traits>

/**
* @brief Times the execution of a callable object.
* @param func Callable object (lambda, function pointer, functor).
* @param args Arguments to pass to the callable.
* @return If the callable returns void, returns elapsed time in seconds as double.
*         Otherwise, returns std::pair of the result and elapsed time in seconds.
*/
template <typename Func, typename... Args>
inline auto timeCall(Func&& func, Args&&... args) {
    auto start = clk::now();
    if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) {
        std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        auto end = clk::now();
        return std::chrono::duration<double>(end - start).count();
    } else {
        auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        auto end = clk::now();
        return std::make_pair(result, std::chrono::duration<double>(end - start).count());
    }
}

// ##################################################################################################################################

#ifdef _DEBUG
	#define TIMER_CREATE(TIMER) Timer TIMER;
	#define TIMER_START_MEASURE(FUN, IF, TIMER, NAME) 	{ 		std::string tmp = "";																					\
																if(TIMER && IF) TIMER->checkpoint(NAME);																\
																	FUN; 																								\
																if(TIMER && IF) std::cout << "\t\t\t\t\t->" << #FUN << " took: " << TIMER->elapsed(NAME) << std::endl;	\
														} 
#else
	#define TIMER_CREATE(TIMER)
	#define TIMER_START_MEASURE(FUN, IF, TIMER, NAME) FUN;
#endif