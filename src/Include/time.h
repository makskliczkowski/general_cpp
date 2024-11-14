#include <chrono>
#include <iostream>
#include <algorithm>
#include <string>
#include <time.h>
#include <type_traits>
#include "exceptions.h"
#include "str.h"


/*******************************
* Contains the possible methods
* for handling the sim time.
*******************************/
#pragma once

// ########################################################	T I M E   F U N C T I O N S ########################################################
#if defined (_MSC_VER)
	using clk					=				std::chrono::steady_clock;
#else
	using clk					=				std::chrono::system_clock;
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

	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	std::string elapsed(_T1 _point, _T1 _start = 0, TimePrecision _prec = TimePrecision::MICROSECONDS);
	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	std::string elapsed(_T1 _point, TimePrecision _prec = TimePrecision::MICROSECONDS);

	std::string elapsed(const std::string& _point, const std::string& _since, TimePrecision _prec = TimePrecision::MICROSECONDS);
	std::string elapsed(const std::string& _point, TimePrecision _prec = TimePrecision::MICROSECONDS);
};

// ##################################################################################################################################

std::string prettyTime(std::time_t now = std::time(0));

// ##################################################################################################################################

#ifdef _DEBUG
	#define TIMER_CREATE(TIMER) Timer TIMER;
	#define TIMER_START_MEASURE(FUN, IF, TIMER, NAME) 	{ 		std::string tmp = "";																		\
																if(IF) TIMER.checkpoint(NAME);																\
															   	FUN; 																						\
																if(IF) std::cout << "\t\t\t\t\t->" << #FUN << " took: " << TIMER.elapsed(NAME) << std::endl;\
														} 
#else
	#define TIMER_CREATE(TIMER)
	#define TIMER_START_MEASURE(FUN, IF, TIMER, NAME) FUN;
#endif