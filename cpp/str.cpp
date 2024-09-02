#include "../src/Include/str.h"

// ######################################################## STRING RELATED FUNCTIONS ########################################################

/*
* @brief Splits string according to the delimiter
* @param A a string to be split
* @param delimiter A delimiter. Default = '\\t'
* @return Split string
*/
strVec splitStr(const std::string& s, std::string delimiter)
{
	unsigned long long pos_start	= 0;
	unsigned long long pos_end		= 0;
	unsigned long long delim_len	= delimiter.size();
	std::string token				= "";
	strVec res						= {};

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token		= s.substr(pos_start, pos_end - pos_start);
		pos_start	= pos_end + delim_len;
		res.push_back(token);
	}
	res.push_back(s.substr(pos_start));
	return res;
}

// ############################################################# STRING FROM CMD ############################################################

/*
* We want to handle files so let's make the c-way input a string. This way we will parse the command line arguments
* @param argc number of main input arguments
* @param argv main input arguments
* @returns vector of strings with the arguments from command line
*/
strVec fromPtr(int argc, char** argv, unsigned int offset)
{
	v_1d<std::string> tmp(argc - offset, "");
	for (unsigned int i = 0; i < (unsigned int)argc - offset; i++)
		tmp[i] = argv[i + offset];
	return tmp;
};

// ###################################################################################################################################

/*
* @brief Checks if the string is a alphanumeric
* @param s a string to check
* @returns true if the string is a alphanumeric
*/
bool StrParser::isAlphanum(const std::string &s)
{
	for (auto& c : s)
		if (!std::isalnum(c))
			return false;
	return true;
}

// ###################################################################################################################################

/*
* @brief Checks if the string is a number
* @param s a string to check
* @returns true if the string is a number
*/
bool StrParser::isNumber(const std::string &s)
{
	for (auto& c : s)
		if (!std::isdigit(c))
			return false;
	return true;
}

// ###################################################################################################################################

/*
* @brief Checks if the string contains a substring
* @param s a string to check
* @param sub a substring to check
* @returns true if the string contains a substring
*/
bool StrParser::contanins(const std::string &s, const std::string &sub)
{
	return s.find(sub) != std::string::npos;
}

// ###################################################################################################################################

/*
* @brief Splits string according to the delimiter
* @param s a string to be split
* @param delimiter a delimiter. Default = '\\t'
* @returns split string
*/
strVec StrParser::split(const std::string &s, char delimiter)
{
	unsigned long long pos_start	= 0;
	unsigned long long pos_end		= 0;
	unsigned long long delim_len	= 1;
	std::string token				= "";
	strVec res						= {};

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token		= s.substr(pos_start, pos_end - pos_start);
		pos_start	= pos_end + delim_len;
		res.push_back(token);
	}
	res.push_back(s.substr(pos_start));
	return res;
}

/*
* @brief Splits string according to the delimiter
* @param s a string to be split
* @param delimiter a delimiter. Default = '\\t'
* @returns split string
*/
strVec StrParser::split(const std::string &s, const std::string &delimiter)
{
	unsigned long long pos_start	= 0;
	unsigned long long pos_end		= 0;
	unsigned long long delim_len	= delimiter.size();
	std::string token				= "";
	strVec res						= {};

	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token		= s.substr(pos_start, pos_end - pos_start);
		pos_start	= pos_end + delim_len;
		res.push_back(token);
	}
	res.push_back(s.substr(pos_start));
	return res;
}

// ###################################################################################################################################