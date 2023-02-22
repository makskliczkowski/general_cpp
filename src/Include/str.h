#pragma once
#include <string>
#include <vector>
#ifdef __has_include
#  if __has_include(<format>)
#    include <format>
#    define HAS_FORMAT 1
#	 define strf std::format
#  else
#    define HAS_FORMAT 0
#  endif
#endif

#define STR std::to_string
#define STRP(str,prec) str_p(str, prec)
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

// ######################################################## STRING RELATED FUNCTIONS ########################################################

/*
* @brief Splits string according to the delimiter
* @param A a string to be split
* @param delimiter A delimiter. Default = '\\t'
* @return Split string
*/
inline v_1d<std::string> split(const std::string& s, std::string delimiter = "\t") {
	unsigned long long pos_start = 0;
	unsigned long long pos_end;
	unsigned long long delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}
	res.push_back(s.substr(pos_start));
	return res;
}

/*
* We want to handle files so let's make the c-way input a string. This way we will parse the command line arguments
* @param argc number of main input arguments
* @param argv main input arguments
* @returns vector of strings with the arguments from command line
*/

// ######################################################## String vector ########################################################

/*
* @brief class containing the vector of string with manipulators
*/
class StringVector
{
	typedef std::vector<std::string> T;
	T vec_;
	public:
		StringVector(T vec) : vec_(vec) {};
	
	template <typename T>
	inline v_1d<std::string> fromPtr(uint argc, T** argv, uint offset = 1) {
		T tmp(argc - offset, "");
		for (auto i = 0; i < argc - offset; i++) 
			tmp[i] = str(argv[i + offset]);
		return tmp;
	};
}; 