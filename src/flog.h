#pragma once
#include "Include/files.h"
#include "Include/directories.h"
#include "Include/exceptions.h"

#ifndef FLOGTIME
#define FLOGTIME
#include <time.h>
#include <stdio.h>
#endif

/*******************************
* Contains the possible methods
* for logging the info etc.
*******************************/

// ######################################################## L O G   L E V E L S ########################################################

extern int LASTLVL;

/*
* @brief prints log level based on a given input
* @param _lvl tabulation level
*/
inline void logLvl(unsigned int _lvl) {
	while (_lvl--)
		std::cout << "\t";
	std::cout << "->";
}

/*
* @brief Stores the types of the logger for the program
*/
enum LOG_TYPES 
{
	INFO,
	TIME,
	ERROR,
	TRACE,
	CHOICE,
	FINISH,
	WARNING
};

BEGIN_ENUM(LOG_TYPES)
{
	DECL_ENUM_ELEMENT(INFO),
	DECL_ENUM_ELEMENT(TIME),
	DECL_ENUM_ELEMENT(ERROR),
	DECL_ENUM_ELEMENT(TRACE),
	DECL_ENUM_ELEMENT(CHOICE),
	DECL_ENUM_ELEMENT(FINISH),
	DECL_ENUM_ELEMENT(WARNING)
}
END_ENUM(LOG_TYPES);
#define LOG_INFO(TYP)								"[" + SSTR(getSTR_LOG_TYPES(TYP)) + "]"

// --- create log file if necessary ---
#ifdef LOG_FILE
	#define LOG_DIR									SSTR("LOG") + kPS
	static inline std::string LOG_FILENAME			= "log";
	static clk::time_point LOG_TIME;
#endif

inline void SET_LOG_TIME() {
#ifdef LOG_FILE
	createDir(LOG_DIR);
	LOG_TIME			=		clk::now();
	LOG_FILENAME		=		"." + kPS + "LOG" + kPS + "log_" + prettyTime(LOG_TIME) + ".txt";
	std::ofstream file(LOG_FILENAME);
#endif
};

// ##########################################################################################################################################

/*
* @brief Prints the log for the message using a specific log type
* @param _msg message to be printed
* @param _typ type of the message (one of INFO,	TIME, ERROR, TRACE, CHOICE, FINISH, WARNING)
* @param _lvl tabulation level
*/
template <typename _T>
inline void LOGINFO(const _T& _msg, LOG_TYPES _typ, unsigned int _lvl) 
{
#ifdef FLOGTIME
	// take the time
	std::time_t now		= std::time(0);
	char buf[42];
#ifdef _WIN32
	std::tm* now_tm		= new tm;
	gmtime_s(now_tm, &now);
#elif defined __linux__ 
	std::tm* now_tm 	= std::localtime(&now);
#endif
	std::strftime(buf, 42, "%Y-%m-%d:%X", now_tm);
	std::cout << "[" << buf << "]";
	// clear memory
#ifdef _WIN32
	delete now_tm;
#endif

#endif // FLOGTIME
	
	std::cout << "[" << getSTR_LOG_TYPES(_typ) << "]";
	logLvl(_lvl);
	std::cout << _msg << std::endl;
#ifdef LOG_FILE
	std::ofstream _file;
	openFile(_file, LOG_FILENAME, std::ios::app);
	// check level of log
	if (_lvl > 0)
		_file << "->";
	while (_lvl--)
		_file << "\t";
	// write to the file
	_file << "[" << getSTR_LOG_TYPES(_typ) << "]" << _msg << std::endl;
	_file.close();
#endif
}

// ##########################################################################################################################################

/*
* @brief prints log info based on a given input message and type
* @param _msg message to be printed
* @param _typ type of the message (one of INFO,	TIME, ERROR, TRACE, CHOICE, FINISH, WARNING)
* @param _lvl tabulation level
*/
template<>
inline void LOGINFO(const std::string& _msg, LOG_TYPES _typ, unsigned int _lvl)
{
#ifdef FLOGTIME
	// take the time
	std::time_t now		= std::time(0);
	char buf[42];
#ifdef _WIN32
	std::tm* now_tm		= new tm;
	gmtime_s(now_tm, &now);
#elif defined __linux__ 
	std::tm* now_tm 	= std::localtime(&now);
#endif
	std::strftime(buf, 42, "%Y-%m-%d:%X", now_tm);
	std::cout << "[" << buf << "]";
	// clear memory
#ifdef _WIN32
	delete now_tm;
#endif
#endif // FLOGTIME

	std::cout << "[" << getSTR_LOG_TYPES(_typ) << "]";
	logLvl(_lvl);
	std::cout << _msg << std::endl;

#ifdef LOG_FILE
	std::ofstream _file;
	openFile(_file, LOG_FILENAME, std::ios::app);
	// check level of log
	if (_lvl > 0)
		_file << "->";
	while (_lvl--)
		_file << "\t";
	// write to the file
	_file << "[" << getSTR_LOG_TYPES(_typ) << "]" << _msg << std::endl;
	_file.close();
#endif
}

// ##########################################################################################################################################

/*
* @brief prints log info based on a given input message and type - GLOBAL
* @param _msg message to be printed
* @param _typ type of the message (one of INFO,	TIME, ERROR, TRACE, CHOICE, FINISH, WARNING)
* @param _lvl tabulation level
*/
template <typename _T>
inline void LOGINFOG(const _T& _msg, LOG_TYPES _typ, unsigned int _lvl) 
{
	LOGINFO(_msg, _typ, _lvl + LASTLVL);
}

// ##########################################################################################################################################

/*
* @brief Log the global title at a specific level
* @param _msg message to be printed
* @param _typ type of the message (one of INFO,	TIME, ERROR, TRACE, CHOICE, FINISH, WARNING)
* @param _lvl tabulation level
* @param _desiredSize width of the log columns
* @param fill filling the empty space with that character
*/
inline void LOGINFO(LOG_TYPES _typ, 
					const std::string& _msg,
					unsigned int _desiredSize,
					char fill			= '#',
					unsigned int _lvl	= 0)
{
	auto _tailLen	= _msg.size();
	auto _lvlLen	= 2 + _lvl * 3 * 2;

	// check the length
	if (_tailLen + _lvlLen >= _desiredSize)
	{
		LOGINFO(_msg, _typ, _lvl);
		return;
	}

	// check the size of the fill
	auto fillSize	= _desiredSize - _tailLen;
	fillSize		= fillSize + (!(_tailLen == 0) ? 0 : 2);
	fillSize		= fillSize - (!(_tailLen % 2 == 0) ? 1 : 0);

	std::string out	= "";
	
	// append first
	for (int i = 0; i < fillSize; ++i)
		out			= out + fill;

			// append text
	if(!(_tailLen == 0))
		out			= out + " " + _msg + " ";

	// append last
	for (int i = 0; i < fillSize; ++i)
		out			= out + fill;

	LOGINFO(out, _typ, _lvl);
}

// ##########################################################################################################################################

/*
* @brief Breakline loginfo
* @param _n n lines to break
*/
inline void LOGINFO(unsigned int _n)
{
	std::string _out	=	"";
	for (auto i = 0; i < _n; ++i)
		_out			+=	"\n";
	LOGINFO(LOG_TYPES::TRACE, _out, 0);
}

// ##########################################################################################################################################

/*
* @brief Log timestamp difference from now (in miliseconds and seconds) - for timestamping
* @param _t timestamp to be used for timedifference
* @param funName name of the function or method to be timestamped
* @param _lvl loglevel
*/
inline void LOGINFO(const clk::time_point& _t, const std::string& funName, unsigned int _lvl = 0)
{
	LOGINFO(1);
	LOGINFO(LOG_TYPES::TRACE, "", 30, '%', std::max(int(_lvl) - 2, 0));
	LOGINFO("Function: " + funName + " took:",	LOG_TYPES::TIME, _lvl);
	LOGINFO(STR(t_ms(_t)) + " ms",				LOG_TYPES::TIME, _lvl + 1);
	LOGINFO(STR(t_s(_t)) +	" s",				LOG_TYPES::TIME, _lvl + 1);
	LOGINFO(LOG_TYPES::TRACE, "", 30, '%', std::max(int(_lvl) - 2, 0));
	LOGINFO(1);
}

// ##########################################################################################################################################

/*
* @brief Changes the external level
*/
inline void LOGINFO_CH_LVL(unsigned int _lvl)
{
	LASTLVL = _lvl;
}

// ##########################################################################################################################################
