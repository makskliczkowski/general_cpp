#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <iostream> 
#include <complex>
#include <utility>
#include <cstdint>

template<class T>
using v_3d = std::vector<std::vector<std::vector<T>>>;				// 3d vector
template<class T>
using v_2d = std::vector<std::vector<T>>;							// 2d vector
template<class T>
using v_1d = std::vector<T>;										// 1d vector
template<class T>
using t_3d = std::tuple<T, T, T>;									// 3d tuple
template<class T>
using t_2d = std::pair<T, T>;										// 2d tuple - pair

#define SSTR std::string
#define STR std::to_string
#define STRS(str) str_p(str, 2, true)
#define STRP(str,prec) str_p(str, prec)
#define STRPS(str, prec) str_p(str, prec, true)
using strVec = v_1d<std::string>;


// ############################################################### P R E C I S E   S T R I N G ###############################################################

/**
*@brief Changes a value to a string with a given precision
*@param v Value to be transformed
*@param n Precision default 2
*@return String of a value
*/
template <typename _T>
inline std::string str_p(const _T v, const int n = 2, bool scientific = false) {
	std::ostringstream out;
	out.precision(n);
	if (scientific)
		out << std::scientific;
	else
		out << std::fixed;
	out << v;
	return out.str();
}

template <>
inline std::string str_p(const int v, const int n, bool scientific) {
	std::ostringstream out;
	if (scientific)
		out << std::scientific;
	out << v;
	return out.str();
}

template <>
inline std::string str_p(const std::complex<double> v, const int n, bool scientific) {
	std::ostringstream out;
	out.precision(n);
	if (scientific)
		out << std::scientific;
	else
		out << std::fixed;
	out << "[" << std::real(v) << ", " << std::imag(v) << "]";
	return out.str();
}
template <>
inline std::string str_p(std::string_view v, const int n, bool scientific) {
	return std::string(v);
}
template <>
inline std::string str_p(const char* v, const int n, bool scientific) {
	return SSTR(v);
}
template <>
inline std::string str_p(strVec v, const int n, bool scientific) {
	std::string tmp = "";
	for (auto& i : v)
		tmp += i + " ";
	tmp.pop_back();
	return tmp;
}

namespace StrParser
{
	// ############################################################### C O L O R I Z E ###############################################################

	struct StrColors
	{
		static inline std::string black		= "\033[30m";
		static inline std::string red		= "\033[31m";
		static inline std::string green		= "\033[32m";
		static inline std::string yellow	= "\033[33m";
		static inline std::string blue		= "\033[34m";
		static inline std::string magenta	= "\033[35m";
		static inline std::string cyan		= "\033[36m";
		static inline std::string white		= "\033[37m";
	};

	template <typename _T = std::string>
	inline std::string colorize(const _T& v, std::string_view color [[maybe_unused]] = StrColors::white) 
	{
#ifdef _WIN32
		return color + v + "\033[0m";
#else
		return v;
#endif
	}

	// ###################################################################################################################################

	bool isNumber(std::string_view s);
	bool isAlphanum(std::string_view s);
	bool contanins(std::string_view s, std::string_view sub);

	// ###################################################################################################################################

	strVec split(std::string_view s, char delimiter = '\t');
	strVec split(std::string_view s, std::string_view delimiter = "\t");
	strVec fromPtr(int argc, char** argv, unsigned int offset = 1);

	// ###################################################################################################################################

	bool endsWith(std::string_view _str, std::string_view _suf);
	
};


// ############################################################### V E C T O R I Z E ###############################################################

strVec splitStr(std::string_view s, std::string_view delimiter = "\t");
strVec fromPtr(int argc, char** argv, unsigned int offset = 1);

// ############################################################### S E P A R A T E D   S T R I N G ###############################################################

template <typename Type>
inline void strSepP(std::string& _out, char _sep, uint16_t prec, bool scien, Type arg) {
	_out += str_p(arg, prec, scien);
}

template <typename Type, typename... Types>
inline void strSepP(std::string& _out, char _sep, uint16_t prec, bool scien, Type arg, Types... elements) {
	strSepP(_out, _sep, prec, scien, arg);	
	_out += std::string(1, _sep);
	strSepP(_out, _sep, prec, scien, elements...);
}

template <typename... Types>
inline void strSeparatedP(std::string& out, char sep, uint16_t prec, Types... elements) {
	strSepP(out, sep, prec, false, elements...);
}

template <typename... Types>
inline void strSeparatedS(std::string& out, char sep, Types... elements) {
	strSepP(out, sep, 2, true, elements...);
}

template <typename... Types>
inline void strSeparated(std::string& out, char sep, Types... elements) {
	strSepP(out, sep, 2, false, elements...);
}

/**
* @brief Checks if a given string ends with a specified suffix.
* 
* This function determines whether the string `_str` ends with the suffix `_suf`.
* 
* @param _str The string to be checked.
* @param _suf The suffix to be checked for.
* @return true if `_str` ends with `_suf`, false otherwise.
*/
inline bool endsWith(std::string_view _str, std::string_view _suf)
{
	return _str.size() >= _suf.size() && 0 == _str.compare(_str.size() - _suf.size(), _suf.size(), _suf);
}