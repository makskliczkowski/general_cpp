#include "../src/UserInterface/ui.h"


/*
* @brief Find a given option in a vector of string given from cmd parser
* @param vec vector of strings from cmd
* @param option the option that we seek
* @returnsvalue for given option if exists, if not an empty string
*/
std::string user_interface::getCmdOption(const v_1d<std::string>& vec, std::string option) const
{
	if (auto itr = std::find(vec.begin(), vec.end(), option); itr != vec.end() && ++itr != vec.end())
		return *itr;
	return std::string();
}


/*
* @brief If the commands are given from file, we must treat them the same as arguments
* @param filename the name of the file that contains the command line
* @returns
*/
std::vector<std::string> user_interface::parseInputFile(std::string filename) {
	v_1d<std::string> commands;
	std::ifstream inputFile(filename);
	std::string line = "";
	if (!inputFile.is_open()) {
		std::cout << "Cannot open a file " + filename + " that I could parse. Setting all parameters to default. Sorry :c \n";
		this->set_default();
	}
	else {
		if (std::getline(inputFile, line)) {
			// saving lines to out vector if it can be done, then the parser shall treat them normally
			commands = split_str(line, " ");
		}
	}
	return std::vector<std::string>(commands.begin(), commands.end());
}

