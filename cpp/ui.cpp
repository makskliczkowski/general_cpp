#include "../src/UserInterface/ui.h"

/*
* @brief If the commands are given from file, we must treat them the same as arguments
* @param filename the name of the file that contains the command line
* @returns cmd arguments to be parsed
*/
UserInterface::cmdArg UserInterface::parseInputFile(std::string filename) {
	std::ifstream inputFile;
	if (!openFile(inputFile, filename)) this->setDefault();
	else
		if (std::string line = ""; std::getline(inputFile, line))
			return splitStr(line, " ");
	return {};
}

/*
* @brief sets the message sent to user to default
* @param value value to set the option onto
* @param option option parameter
* @param message message to be output if this fails
*/
std::string UserInterface::setDefaultMsg(std::string value, std::string option, std::string message, const cmdMap& map) const
{
	std::string out = "";
	// find the option
	auto it = map.find(option);
	if (it != map.end())
	{
		auto& [val_str, fun] = it->second; // if in table - we take the enum
		out = fun(value);
		if (out != "") {
			stout << message + out << "\n";																				// print warning
			return val_str;
		}
	}
	return option;
}

/*
* @brief Find a given option in a vector of string given from cmd parser
* @param vec vector of strings from cmd
* @param option the option that we seek
* @returnsvalue for given option if exists, if not an empty string
*/
std::string UserInterface::getCmdOption(cmdArg& vec, std::string option) const
{
	if (auto itr = std::find(vec.begin(), vec.end(), option); itr != vec.end() && ++itr != vec.end())
		return *itr;
	return std::string();
}