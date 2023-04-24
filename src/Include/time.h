#pragma once
#include <chrono>
#include <iostream>
#include <algorithm>
#include <time.h>
#include "exceptions.h"

/*
* Define function signatures to use in debug scenarios
*/
#if !defined(LOCALTIME_S)
	#if defined(__unix__)
		#define LOCALTIME_S(TIMER_T, ST) localtime_r(&TIMER_T, &ST)
		#pragma message ("--> Using localtime_r")
	#elif defined (_MSC_VER)
		#define LOCALTIME_S(TIMER_T, ST) localtime_s(&ST, &TIMER_T)
		#pragma message ("--> Using localtime_s")
	#else 
		#define LOCALTIME_S     static std::mutex mtx; std::lock_guard<std::mutex> lock(mtx); bt = *std::localtime(&timer);
		#pragma message ("--> Using weird mutex")
	#endif
#endif

/*******************************
* Contains the possible methods
* for handling the sim time.
*******************************/

// ########################################################	T I M E   F U N C T I O N S ########################################################
#if defined (_MSC_VER)
	using clk						=				std::chrono::steady_clock;
#else
	using clk						=				std::chrono::system_clock;
#endif

using clkS						=				std::chrono::system_clock;
#define DUR										std::chrono::duration
#define DURCAST									std::chrono::duration_cast
#define NOW										std::chrono::high_resolution_clock::now()	    											// standard out microseconds
#define DURATION(t1, t2)						static_cast<long double>(DURCAST<std::chrono::microseconds>(DUR(t1 - t2)).count())
/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_s(clk::time_point start)			RETURNS(DURATION(NOW, start) / 1e6);
inline auto TS(clk::time_point start)			-> std::string { return STRP(t_s(start), 3) + "s"; };

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_ms(clk::time_point start)			RETURNS(DURATION(NOW, start) / 1e3);
inline auto TMS(clk::time_point start)			-> std::string { return STRP(t_ms(start), 3) + "ms"; };

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_mus(clk::time_point start)		RETURNS(DURATION(NOW, start));
inline auto TMUS(clk::time_point start)			-> std::string { return STRP(t_mus(start), 3) + "mus"; };

#define stouts(text, start)		stout	<< text <<	" -> time : " << tim_s(start)	<< "s"		<< EL					// standard out seconds
#define stoutms(text, start)	stout	<< text <<	" -> time : " << tim_ms(start)	<< "ms"		<< EL					// standard out miliseconds
#define stoutmus(text, start)	stout	<< text <<	" -> time : " << tim_mus(start) << "mus"	<< EL	

// strftime format
constexpr auto PRETTY_TIME_FORMAT				= "%Y-%m-%d_%H:%M:%S";

// printf format
constexpr auto PRETTY_TIME_FORMAT_MS			= ".%03Id";

/*
* @brief convert current time to milliseconds since unix epoch
*/
template <typename _T>
static auto to_ms(const std::chrono::time_point<_T>& tp)
{
	return DURCAST<std::chrono::milliseconds>(tp.time_since_epoch()).count();
};

/*
* @brief pretty prints the time point
*/
static std::string prettyTime(clk::time_point _tp)
{
#ifdef HAS_FORMAT
	return std::format("{0:%F_%T}", _tp);
#else
	auto curTime	=	\
		clkS::to_time_t(clkS::now() + DURCAST<clkS::duration>(_tp - clk::now()));

	// this function use static global pointer. so it is not thread safe solution
	std::tm timeInfo;
	LOCALTIME_S(curTime, timeInfo);
	

	// create a buffer
	char buffer[128];

	auto size			=	strftime(buffer,		
									(int)sizeof(buffer),
									PRETTY_TIME_FORMAT,
									&timeInfo
									);

	//auto ms				=	to_ms(_tp) % 1000;

	//size				+=	std::snprintf(buffer + size, 
	//								(size_t)sizeof(buffer) - size,
	//								PRETTY_TIME_FORMAT_MS, 
	//								ms
	//								);
	return std::string(buffer, buffer + size);
#endif
}
