#ifndef __TIME_H__
#define __TIME_H__

#include "../src/Include/time.h"
#include <map>
#include <type_traits>

// #################################################################################################################################################


/**
* @brief Reset the timer to the current time
*/
void Timer::reset(clk::time_point _t)
{
    _start = _t;
    _timestampNames.clear();
    _timestamps.clear();
    _iter = 0;

    // set the starting point
    _timestampNames[_startName] = 0;
    _timestamps.push_back(_start);
    _iter++;
}

// #################################################################################################################################################

/*
* @brief Creates a checkpoint for the timer.
*/
void Timer::checkpoint(const std::string& _name)
{
    _timestampNames[_name] = _timestamps.size();
    _timestamps.push_back(NOW);
    _last = _name;
    _iter++;
}

// #################################################################################################################################################

/*
* @brief Returns the specific timepoint at a given index
* @param _name specific timepoint idx
* @returns timepoint
*/
template<typename _T1, typename _T2>
clk::time_point Timer::point(_T1 _idx)
{
    if (_idx >= this->_timestamps.size())
        throw std::runtime_error("Not enough timestamps in the vector.");
    return this->_timestamps[_idx];
}

// size_t
template clk::time_point Timer::point(size_t _idx);
// int
template clk::time_point Timer::point(int _idx);
// long
template clk::time_point Timer::point(long _idx);
// long long
template clk::time_point Timer::point(long long _idx);

// #################################################################################################################################################

/*
* @brief Returns the specific timepoint with a given name
* @param _name specific timepoint name
* @returns timepoint
*/
clk::time_point Timer::point(const std::string& _name)
{
    return this->_timestamps[this->_timestampNames[_name]];
}

// #################################################################################################################################################

/*
* @brief Returns all the timestamps
*/
std::vector<clk::time_point> Timer::point() const
{
    return this->_timestamps;
}

// #################################################################################################################################################

/*
* @brief Returns the starting timestamp
*/
clk::time_point Timer::start() const
{
    return this->_timestamps[0];
}

// #################################################################################################################################################

/*
* @brief Returns the ending timestamp
*/
clk::time_point Timer::end() const
{
    return this->_timestamps[this->_timestamps.size() - 1];
}

// #################################################################################################################################################

/**
* @brief Get the elapsed time at given indices
* @param _point specific ending timepoint idx
* @param _point specific staring timepoint idx
* @param _prec precision to be used
* @returns string with a timestamp
*/
template<typename _T1, typename _T2, typename _R>
_R Timer::elapsed(_T1 _point, _T1 _start, TimePrecision _prec)
{
    if constexpr (std::is_same_v<_R, std::string>)
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
    else if constexpr (std::is_arithmetic<_R>::value)
    {
        switch (_prec)
        {
        case TimePrecision::MICROSECONDS:
            return std::chrono::duration_cast<std::chrono::microseconds>(this->_timestamps[_point] - this->_timestamps[_start]).count();
            break;
        case TimePrecision::MILLISECONDS:
            return std::chrono::duration_cast<std::chrono::milliseconds>(this->_timestamps[_point] - this->_timestamps[_start]).count();
            break;
        case TimePrecision::SECONDS:
            return std::chrono::duration_cast<std::chrono::seconds>(this->_timestamps[_point] - this->_timestamps[_start]).count();
            break;
        default:
            return std::chrono::duration_cast<std::chrono::microseconds>(this->_timestamps[_point] - this->_timestamps[_start]).count();
            break;
        }
    }
    else 
        throw std::runtime_error("Unknown return type.");
}

template std::string Timer::elapsed(size_t _point, size_t _start, TimePrecision _prec);
template std::string Timer::elapsed(int _point, int _start, TimePrecision _prec);
template std::string Timer::elapsed(long _point, long _start, TimePrecision _prec);
template std::string Timer::elapsed(long long _point, long long _start, TimePrecision _prec);
// different return type
template long Timer::elapsed(size_t _point, size_t _start, TimePrecision _prec);
template long Timer::elapsed(int _point, int _start, TimePrecision _prec);
template long Timer::elapsed(long _point, long _start, TimePrecision _prec);
template long Timer::elapsed(long long _point, long long _start, TimePrecision _prec);


// #################################################################################################################################################

/**
* @brief Measures the elapsed time from a given timestamp to the current time.
* 
* @tparam _T1 Type of the timestamp identifier.
* @tparam _T2 Type of the timestamp value (not used in the function).
* @tparam _R Return type of the elapsed time (either std::string or an arithmetic type).
* @param _point Identifier for the timestamp to measure from.
* @param _prec Precision of the time measurement (microseconds, milliseconds, or seconds).
* @return _R Elapsed time in the specified precision. If _R is std::string, the time is returned as a formatted string.
* If _R is an arithmetic type, the time is returned as a numeric value.
* @throws std::runtime_error If the return type _R is neither std::string nor an arithmetic type.
*/
template<typename _T1, typename _T2, typename _R>
_R Timer::elapsed(_T1 _point, TimePrecision _prec)
{
    if constexpr (std::is_same_v<_R, std::string>)
    {
        switch (_prec)
        {
        case TimePrecision::MICROSECONDS:
            return TMUS(NOW, this->_timestamps[_point]);
            break;
        case TimePrecision::MILLISECONDS:
            return TMS(NOW, this->_timestamps[_point]);
            break;
        case TimePrecision::SECONDS:
            return TS(NOW, this->_timestamps[_point]);
            break;
        default:
            return TMUS(NOW, this->_timestamps[_point]);
            break;
        }
    }
    else if constexpr (std::is_arithmetic<_R>::value)
    {
        switch (_prec)
        {
        case TimePrecision::MICROSECONDS:
            return std::chrono::duration_cast<std::chrono::microseconds>(NOW - this->_timestamps[_point]).count();
            break;
        case TimePrecision::MILLISECONDS:
            return std::chrono::duration_cast<std::chrono::milliseconds>(NOW - this->_timestamps[_point]).count();
            break;
        case TimePrecision::SECONDS:
            return std::chrono::duration_cast<std::chrono::seconds>(NOW - this->_timestamps[_point]).count();
            break;
        default:
            return std::chrono::duration_cast<std::chrono::microseconds>(NOW - this->_timestamps[_point]).count();
            break;
        }
    }
    else 
        throw std::runtime_error("Unknown return type.");
}

// size_t
template std::string Timer::elapsed(size_t _point, TimePrecision _prec);
template std::string Timer::elapsed(int _point, TimePrecision _prec);
template std::string Timer::elapsed(long _point, TimePrecision _prec);
template std::string Timer::elapsed(long long _point, TimePrecision _prec);
// different return type
template long Timer::elapsed(size_t _point, TimePrecision _prec);
template long Timer::elapsed(int _point, TimePrecision _prec);
template long Timer::elapsed(long _point, TimePrecision _prec);
template long Timer::elapsed(long long _point, TimePrecision _prec);

// #################################################################################################################################################

/**
* @brief Get the elapsed time between two named time points.
* @param _point Name of the ending time point.
* @param _since Name of the starting time point.
* @param _prec Precision to be used for the elapsed time (microseconds, milliseconds, or seconds).
* @returns Elapsed time as a string or an arithmetic type, depending on the template parameter _R.
*/
template<typename _R>
_R Timer::elapsed(const std::string& _point, const std::string& _since, TimePrecision _prec)
{
    if constexpr (std::is_same_v<_R, std::string>)
    {
        switch (_prec) 
        {
        case TimePrecision::MICROSECONDS:
            return StrParser::colorize(TMUS(this->_timestamps[this->_timestampNames[_since]], this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        case TimePrecision::MILLISECONDS:
            return StrParser::colorize(TMS(this->_timestamps[this->_timestampNames[_since]], this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        case TimePrecision::SECONDS:
            return StrParser::colorize(TS(this->_timestamps[this->_timestampNames[_since]], this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        default:
            return StrParser::colorize(TMUS(this->_timestamps[this->_timestampNames[_since]], this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        }
    }
    else if constexpr (std::is_arithmetic<_R>::value)
    {
        switch (_prec) 
        {
        case TimePrecision::MICROSECONDS:
            return std::chrono::duration_cast<std::chrono::microseconds>(this->_timestamps[this->_timestampNames[_point]] - this->_timestamps[this->_timestampNames[_since]]).count();
            break;
        case TimePrecision::MILLISECONDS:
            return std::chrono::duration_cast<std::chrono::milliseconds>(this->_timestamps[this->_timestampNames[_point]] - this->_timestamps[this->_timestampNames[_since]]).count();
            break;
        case TimePrecision::SECONDS:
            return std::chrono::duration_cast<std::chrono::seconds>(this->_timestamps[this->_timestampNames[_point]] - this->_timestamps[this->_timestampNames[_since]]).count();
            break;
        default:
            return std::chrono::duration_cast<std::chrono::microseconds>(this->_timestamps[this->_timestampNames[_point]] - this->_timestamps[this->_timestampNames[_since]]).count();
            break;
        }
    }
    else 
        throw std::runtime_error("Unknown return type.");
}

template std::string Timer::elapsed(const std::string& _point, const std::string& _since, TimePrecision _prec);
template long Timer::elapsed(const std::string& _point, const std::string& _since, TimePrecision _prec);

// #################################################################################################################################################

template<typename _R>
_R Timer::elapsed(const std::string& _point, TimePrecision _prec)
{
    if constexpr (std::is_same_v<_R, std::string>)
    {
        switch (_prec) 
        {
        case TimePrecision::MICROSECONDS:
            return StrParser::colorize(TMUS(NOW, this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        case TimePrecision::MILLISECONDS:
            return StrParser::colorize(TMS(NOW, this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        case TimePrecision::SECONDS:
            return StrParser::colorize(TS(NOW, this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        default:
            return StrParser::colorize(TMUS(NOW, this->_timestamps[this->_timestampNames[_point]]), "red");
            break;
        }
    }
    else if constexpr (std::is_arithmetic<_R>::value)
    {
        switch (_prec) 
        {
        case TimePrecision::MICROSECONDS:
            return std::chrono::duration_cast<std::chrono::microseconds>(NOW - this->_timestamps[this->_timestampNames[_point]]).count();
            break;
        case TimePrecision::MILLISECONDS:
            return std::chrono::duration_cast<std::chrono::milliseconds>(NOW - this->_timestamps[this->_timestampNames[_point]]).count();
            break;
        case TimePrecision::SECONDS:
            return std::chrono::duration_cast<std::chrono::seconds>(NOW - this->_timestamps[this->_timestampNames[_point]]).count();
            break;
        default:
            return std::chrono::duration_cast<std::chrono::microseconds>(NOW - this->_timestamps[this->_timestampNames[_point]]).count();
            break;
        }
    }
    else 
        throw std::runtime_error("Unknown return type.");
}

template std::string Timer::elapsed(const std::string& _point, TimePrecision _prec);
template long Timer::elapsed(const std::string& _point, TimePrecision _prec);

// #################################################################################################################################################

/*
* @brief pretty prints the time point
* @param _tp specific timepoint
* @returns string time in a given format %Y-%m-%d:%X
*/
std::string prettyTime(std::time_t now)
{
	// take the time
	char buf[42];
#ifdef _WIN32
	std::tm* now_tm		= new tm;
	gmtime_s(now_tm, &now);
#elif defined __linux__ 
	std::tm* now_tm 	= std::localtime(&now);
#else
    std::tm* now_tm 	= std::localtime(&now);
#endif
	std::strftime(buf, 42, "%Y-%m-%d:%X", now_tm);
	// clear memory
#ifdef _WIN32
	delete now_tm;
#endif
	return std::string(buf);
}
#endif // __TIME_H__