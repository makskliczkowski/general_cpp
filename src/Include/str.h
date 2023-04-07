#pragma once
#include <string>
#include <vector>
#include <iostream> 
#ifdef __has_include
#  if __has_include(<format>)
#    include <format>
#    define HAS_FORMAT 1
#	 define strf std::format
#  else
#    define HAS_FORMAT 0
#  endif
#endif

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
#define STRP(str,prec) str_p(str, prec)
typedef v_1d<std::string> strVec;

/*
*@brief Changes a value to a string with a given precision
*@param v Value to be transformed
*@param n Precision default 2
*@return String of a value
*/
template <typename _T>
inline std::string str_p(const _T v, const int n = 2) {
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << v;
	return out.str();
}

strVec splitStr(const std::string& s, std::string delimiter = "\t");
strVec fromPtr(int argc, char** argv, unsigned int offset = 1);