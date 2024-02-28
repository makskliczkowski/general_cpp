#pragma once
#include <chrono>
#include <iostream>
#include <algorithm>
#include <time.h>
#include "exceptions.h"

/*
* Define function signatures to use in debug scenarios
*/
//#if !defined(LOCALTIME_S)
//	#if defined(__unix__)
//		#define LOCALTIME_S(TIMER_T, ST) localtime_r(&TIMER_T, &ST)
//		#pragma message ("--> Using localtime_r")
//	#elif defined (_MSC_VER)
//		#define LOCALTIME_S(TIMER_T, ST) localtime_s(&ST, &TIMER_T)
//		#pragma message ("--> Using localtime_s")
//	#else 
//		#define LOCALTIME_S     static std::mutex mtx; std::lock_guard<std::mutex> lock(mtx); bt = *std::localtime(&timer);
//		#pragma message ("--> Using weird mutex")
//	#endif
//#endif

/*******************************
* Contains the possible methods
* for handling the sim time.
*******************************/

// ########################################################	T I M E   F U N C T I O N S ########################################################
#if defined (_MSC_VER)
	using clk					=				std::chrono::steady_clock;
#else
	using clk					=				std::chrono::system_clock;
#endif

using clkS						=				std::chrono::system_clock;
#define DUR										std::chrono::duration
#define DURCAST								std::chrono::duration_cast
#define NOW										std::chrono::high_resolution_clock::now()	    			
#define DURATION(t1, t2)					static_cast<long double>(DURCAST<std::chrono::microseconds>(DUR(t1 - t2)).count())
#define DURATIONS(t1, t2)					static_cast<long double>(DURCAST<std::chrono::seconds>(DUR(t1 - t2)).count())
#define DURATIONMS(t1, t2)					static_cast<long double>(DURCAST<std::chrono::milliseconds>(DUR(t1 - t2)).count())
#define DURATIONMUS(t1, t2)				DURATION(t1, t2)

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_s(clk::time_point start)			RETURNS(DURATIONS(NOW, start));
inline auto t_s(clk::time_point start, 
				clk::time_point end)					RETURNS(DURATIONS(end, start));
inline auto TS(clk::time_point start)			-> std::string { return STRP(t_s(start), 3) + " s";				};
inline auto TS(clk::time_point start,
				 clk::time_point end)				-> std::string { return STRP(t_s(start, end), 3) + " s";		};
/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_ms(clk::time_point start)		RETURNS(DURATIONMS(NOW, start));
inline auto t_ms(clk::time_point start, 
				 clk::time_point end)				RETURNS(DURATIONMS(end, start));
inline auto TMS(clk::time_point start)			-> std::string { return STRP(t_ms(start), 3) + " ms";			};
inline auto TMS(clk::time_point start,	
				 clk::time_point end)				-> std::string { return STRP(t_ms(start, end), 3) + " ms";		};
/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline auto t_mus(clk::time_point start)		RETURNS(DURATION(NOW, start));
inline auto t_mus(clk::time_point start,
				  clk::time_point end)				RETURNS(DURATION(end, start));
inline auto TMUS(clk::time_point start)		-> std::string { return STRP(t_mus(start), 3) + " mus";			};
inline auto TMUS(clk::time_point start,
				 clk::time_point end)				-> std::string { return STRP(t_mus(start, end), 3) + " mus"; }	;

#define stouts(text, start)		stout	<< text <<	" -> time : " << tim_s(start)	<< " s"		<< EL					
#define stoutms(text, start)	stout	<< text <<	" -> time : " << tim_ms(start)	<< " ms"	<< EL			
#define stoutmus(text, start)	stout	<< text <<	" -> time : " << tim_mus(start) << " mus"	<< EL	

// strftime format
constexpr auto PRETTY_TIME_FORMAT				= "%Y-%m-%d_%H:%M:%S";

// printf format
constexpr auto PRETTY_TIME_FORMAT_MS			= ".%03Id";

// ##################################################################################################################################
// ##################################################################################################################################
// ######################################################## T I M E R ###############################################################
// ##################################################################################################################################
// ##################################################################################################################################

#include <map>
class Timer
{
	enum class TimePrecision { MICROSECONDS = 0, MILLISECONDS = 1, SECONDS = 2 };
protected:
	const static inline std::string _startName = "start";
	clk::time_point	_start;
	std::vector<clk::time_point> _timestamps;
	std::map<std::string, size_t> _timestampNames;

public:
	
	// ########### C O N S T R C T ###########

	/*
	* @brief Constructs the timer
	*/
	Timer()
	{
		this->reset();
	};

	/*
	* @brief Reset the timer
	*/
	void reset(clk::time_point _t = NOW)
	{
		_start = _t;
		_timestampNames.clear();
		_timestamps.clear();

		// set the starting point
		_timestampNames[_startName] = 0;
		_timestamps.push_back(_start);
	}

	// ############# S E T T E R S #############

	/*
	* @brief Creates a checkpoint for the timer.
	*/
	void checkpoint(const std::string& _name)
	{
		_timestampNames[_name] = _timestamps.size();
		_timestamps.push_back(NOW);
	}

	// ############# G E T T E R S #############

	/*
	* @brief Returns the specific timepoint at a given index
	* @param _name specific timepoint idx
	* @returns timepoint
	*/
	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	clk::time_point point(_T1 _idx)
	{
		if (_idx >= this->_timestamps.size())
			throw std::runtime_error("Not enough timestamps in the vector.");
		return this->_timestamps[_idx];
	}

	/*
	* @brief Returns the specific timepoint with a given name
	* @param _name specific timepoint name
	* @returns timepoint
	*/
	clk::time_point point(const std::string& _name)
	{
		return this->_timestamps[this->_timestampNames[_name]];
	}

	/*
	* @brief Returns all the timestamps
	*/
	std::vector<clk::time_point> point() const
	{
		return this->_timestamps;
	}

	/*
	* @brief Returns the starting timestamp
	*/
	clk::time_point start() const
	{
		return this->_timestamps[0];
	}

	/*
	* @brief Returns the ending timestamp
	*/
	clk::time_point end() const
	{
		return this->_timestamps[this->_timestamps.size() - 1];
	}

	// ############# E L A P S E D #############

	/*
	* @brief Get the elapsed time at given indices
	* @param _point specific ending timepoint idx
	* @param _point specific staring timepoint idx
	* @param _prec precision to be used
	* @returns string with a timestamp
	*/
	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	std::string elapsed(_T1 _point,
						_T1 _start			= 0,
						TimePrecision _prec = TimePrecision::MICROSECONDS)
	{
		switch (_prec)
		{
		case TimePrecision::MICROSECONDS:
			return TMUS(this->_timestamps[_point], this->_timestamps[_start]);
			break;
		case TimePrecision::MILLISECONDS:
			return TMS(this->_timestamps[_point], this->_timestamps[_start]);
			break;
		case TimePrecision::SECONDS:
			return TS(this->_timestamps[_point], this->_timestamps[_start]);
			break;
		default:
			return TMUS(this->_timestamps[_point], this->_timestamps[_start]);
			break;
		}
	}

	/*
	* @brief Get the elapsed time at given names
	* @param _point specific ending timepoint name
	* @param _point specific staring timepoint name
	* @param _prec precision to be used
	* @returns string with a timestamp
	*/
	std::string elapsed(const std::string& _point, 
						const std::string& _since	= _startName, 
						TimePrecision _prec			= TimePrecision::MICROSECONDS)
	{
		switch (_prec) 
		{
		case TimePrecision::MICROSECONDS:
			return TMUS(this->_timestamps[this->_timestampNames[_point]], this->_timestamps[this->_timestampNames[_since]]);
			break;
		case TimePrecision::MILLISECONDS:
			return TMS(this->_timestamps[this->_timestampNames[_point]], this->_timestamps[this->_timestampNames[_since]]);
			break;
		case TimePrecision::SECONDS:
			return TS(this->_timestamps[this->_timestampNames[_point]], this->_timestamps[this->_timestampNames[_since]]);
			break;
		default:
			return TMUS(this->_timestamps[this->_timestampNames[_point]], this->_timestamps[this->_timestampNames[_since]]);
			break;
		}
	}
};

/*
* @brief pretty prints the time point
* @param _tp specific timepoint
* @returns string time in a given format %Y-%m-%d:%X
*/
static std::string prettyTime(std::time_t now)
{
	// take the time
	char buf[42];
#ifdef _WIN32
	std::tm* now_tm		= new tm;
	gmtime_s(now_tm, &now);
#elif defined __linux__ 
	std::tm* now_tm 	= std::localtime(&now);
#endif
	std::strftime(buf, 42, "%Y-%m-%d:%X", now_tm);
	// clear memory
#ifdef _WIN32
	delete now_tm;
#endif
	return std::string(buf);
}

/*
* @brief pretty prints the time point
* @returns string time in a given format %Y-%m-%d:%X
*/
static std::string prettyTime()
{
	return prettyTime(std::time(0));
}