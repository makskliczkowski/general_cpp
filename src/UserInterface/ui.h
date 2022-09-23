#pragma once
#ifndef UI_H
#define UI_H

#include "../common.h"

#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <functional>

// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

class user_interface {
protected:
	int thread_number;																				 		// number of threads
	int boundary_conditions;																		 		// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC,...
	std::string saving_dir;

	std::string getCmdOption(const v_1d<std::string>& vec, std::string option) const;				 		// get the option from cmd input
	
	template <typename T>
	void set_option(T& value, const v_1d<std::string>& argv, std::string choosen_option, bool geq_0 = true);	// set an option

	template <typename T>
	void set_default_msg(T& value, std::string option, std::string message, \
		const std::unordered_map <std::string, std::string>& map) const;									// setting value to default and sending a message
	// std::unique_ptr<LatticeModel> model;															 			// a unique pointer to the model used

public:
	virtual ~user_interface() = default;

	virtual void make_simulation() = 0;

	virtual void exit_with_help() = 0;
	// ----------------------- REAL PARSING
	virtual void parseModel(int argc, const v_1d<std::string>& argv) = 0;									 // the function to parse the command line
	// ----------------------- HELPING FUNCIONS
	virtual void set_default() = 0;																	 		// set default parameters
	// ----------------------- NON-VIRTUALS
	v_1d<std::string> parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does
};

/*
* @brief sets option from a given cmd options
* @param value
* @param argv
* @param choosen_option
* @param geq_0
*/
template<typename T>
inline void user_interface::set_option(T& value, const v_1d<std::string>& argv, std::string choosen_option, bool geq_0)
{
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		value = static_cast<T>(stod(option));													// set value to an option
	if (geq_0 && value <= 0)																	// if the variable shall be bigger equal 0
		this->set_default_msg(value, choosen_option.substr(1), \
			choosen_option + " cannot be negative\n", default_params);
}

// string instance
template<>
inline void user_interface::set_option<std::string>(std::string& value, const v_1d<std::string>& argv, std::string choosen_option, bool geq_0) {
	if (std::string option = this->getCmdOption(argv, choosen_option); option != "")
		value = option;
}

/*
* @brief sets the message sent to user to default
* @param value
* @param option
* @param message
*/
template<typename T>
inline void user_interface::set_default_msg(T& value, std::string option, std::string message, const std::unordered_map <std::string, std::string>& map) const
{
	std::cout << message;																// print warning
	std::string value_str = "";																// we will set this to value
	auto it = map.find(option);
	if (it != map.end()) {
		value_str = it->second;															// if in table - we take the enum
	}
	value = static_cast<T>(stod(value_str));
}



#endif // !UI_H


