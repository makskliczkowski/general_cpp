#pragma once
#ifndef UI_H
#define UI_H

#ifndef COMMON_H
#include "../common.h"
#endif

#define SETOPTION(n, S) this->setOption(n.S, argv, #S)

// -------------------------------------------------------- Make a User interface class --------------------------------------------------------
inline std::string higherThanZero(std::string s)		{ if (stod(s) <= 0) return "must be higher than 0"; else return ""; };
inline std::string defaultReturn(std::string s)			{ return ""; };

class UserInterface {
protected:
	typedef v_1d<std::string> cmdArg;
	typedef std::unordered_map<std::string, std::tuple<std::string, std::function<std::string(std::string)>>> cmdMap;

	std::string main_dir = "." + kPS;																		// main directory - to be saved onto
	int chosen_funtion									= -1;												// chosen function to be used later
	uint thread_number									= 1;	
	
	// ------------- CHOICES and OPTIONS and DEFAULTS
	cmdMap default_params;																					// default parameters

	// ------------ FUNCTIONS ------------

	std::string getCmdOption(cmdArg& vec, std::string option) const;				 						// get the option from cmd input
	std::string setDefaultMsg(std::string v, std::string opt, std::string message, const cmdMap& map) const;// setting value to default and sending a message
	
	template <typename _T>
	void setOption(_T& value, cmdArg& argv, std::string choice);											// set an option


public:
	virtual ~UserInterface() = default;

	// general functions to override
	virtual void exitWithHelp()							= 0;

	// ----------------------- REAL PARSING
	virtual void funChoice()							= 0;												// allows to choose the method without recompilation of the whole code
	virtual void parseModel(int argc, cmdArg& argv)		= 0;												// the function to parse the command line
	virtual cmdArg parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does
	
	// ----------------------- HELPING FUNCIONS
	virtual void setDefault()							= 0;										 		// set default parameters
	
	// ----------------------- NON-VIRTUALS
};

/*
* @brief sets option from a given cmd options
* @param value a value to be set onto
* @param argv arguments to find the corresponding option
* @param choice chosen option
* @param c constraint on the option
*/
template<typename _T>
inline void UserInterface::setOption(_T& value, cmdArg& argv, std::string choice)
{
	if (std::string option = this->getCmdOption(argv, "-" + choice); option != "") {
		option = this->setDefaultMsg(option, choice.substr(1), choice + ":\n", default_params);
		value = static_cast<_T>(stod(option));																// set value to an option
		
	}
}

template<>
inline void UserInterface::setOption<std::string>(std::string& value, cmdArg& argv, std::string choice) {
	if (std::string option = this->getCmdOption(argv, "-" + choice); option != "")
		value = this->setDefaultMsg(option, std::string(choice.substr(1)), std::string(choice + ":\n"), default_params);
}

#endif // !UI_H


