#include "../src/Include/time.h"
#include <map>

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

/*
* @brief Get the elapsed time at given indices
* @param _point specific ending timepoint idx
* @param _point specific staring timepoint idx
* @param _prec precision to be used
* @returns string with a timestamp
*/
template<typename _T1, typename _T2>
std::string Timer::elapsed(_T1 _point, _T1 _start, TimePrecision _prec)
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
// size_t
template std::string Timer::elapsed(size_t _point, size_t _start, TimePrecision _prec);
// int
template std::string Timer::elapsed(int _point, int _start, TimePrecision _prec);
// long
template std::string Timer::elapsed(long _point, long _start, TimePrecision _prec);
// long long
template std::string Timer::elapsed(long long _point, long long _start, TimePrecision _prec);

// #################################################################################################################################################

template<typename _T1, typename _T2>
std::string Timer::elapsed(_T1 _point, TimePrecision _prec)
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

// size_t
template std::string Timer::elapsed(size_t _point, TimePrecision _prec);
// int
template std::string Timer::elapsed(int _point, TimePrecision _prec);
// long
template std::string Timer::elapsed(long _point, TimePrecision _prec);
// long long
template std::string Timer::elapsed(long long _point, TimePrecision _prec);

// #################################################################################################################################################

/*
* @brief Get the elapsed time at given names
* @param _point specific ending timepoint name
* @param _point specific staring timepoint name
* @param _prec precision to be used
* @returns string with a timestamp
*/
std::string Timer::elapsed(const std::string& _point, const std::string& _since, TimePrecision _prec)
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

// #################################################################################################################################################

std::string Timer::elapsed(const std::string& _point, TimePrecision _prec)
{
    switch (_prec) 
    {
    case TimePrecision::MICROSECONDS:
        return StrParser::colorize(TMUS(this->_timestamps[this->_timestampNames[_point]], NOW), "red");
        break;
    case TimePrecision::MILLISECONDS:
        return StrParser::colorize(TMS(this->_timestamps[this->_timestampNames[_point]], NOW), "red");
        break;
    case TimePrecision::SECONDS:
        return StrParser::colorize(TS(this->_timestamps[this->_timestampNames[_point]], NOW), "red");
        break;
    default:
        return StrParser::colorize(TMUS(this->_timestamps[this->_timestampNames[_point]], NOW), "red");
        break;
    }
}

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