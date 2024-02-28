#pragma once
#include <exception>
#include <stdexcept>
#ifndef SIGNATURES_H
#	include "signatures.h"
#endif
#include "str.h"

/*******************************
* Contains the possible methods
* for handling the exceptions.
* REV : 01/12/23 - Maks Kliczkowski
*******************************/
#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

constexpr auto LOG_LVL0 = "";
constexpr auto LOG_LVL1 = "\t->";
constexpr auto LOG_LVL2 = "\t\t->";
constexpr auto LOG_LVL3 = "\t\t\t->";
constexpr auto LOG_LVL4 = "\t\t\t\t->";

// ######################################################## E X C E P T I O N S ########################################################
namespace EXCEPTIONENUM 
{
	enum EXCEPTIONS
	{
		UNDEFINED	= 0,
		RUNTIME		= 1,
		BAD_ALOC	= 2,
		EXCEPTION	= 3,
		FILEE		= 4
	};

	BEGIN_ENUM(EXCEPTIONS)
	{
		DECL_ENUM_ELEMENT(UNDEFINED),
		DECL_ENUM_ELEMENT(RUNTIME),
		DECL_ENUM_ELEMENT(BAD_ALOC),
		DECL_ENUM_ELEMENT(EXCEPTION),
		DECL_ENUM_ELEMENT(FILEE)
	}
	END_ENUM(EXCEPTIONS);
};

/*
* @brief Class that handles the exceptions sent by the software
*/
class ExceptionHandler 
{
public:
	static void printException(const std::string& _what, const std::string& _msg, EXCEPTIONENUM::EXCEPTIONS _ex = EXCEPTIONENUM::EXCEPTIONS::UNDEFINED) {
		auto exIDX = EXCEPTIONENUM::getSTR_EXCEPTIONS(_ex);
		std::cout << LOG_LVL0 << "Exception: " << exIDX << std::endl;
		std::cout << LOG_LVL1 << _what << std::endl;
		std::cout << LOG_LVL2 << _msg << std::endl;
		exit(static_cast<int>(_ex));
	};
	static void handleExceptions(std::exception_ptr _ePtr, const std::string& _msg);
};

#define BEGIN_CATCH_HANDLER				try
#define END_CATCH_HANDLER(message, DO)	catch(...){ ExceptionHandler::handleExceptions(std::current_exception(), message); DO;}

#define IFELSE_EXCEPTION(IF, IFDO, THROW) if(IF) IFDO; else throw std::runtime_error(THROW);
#define IF_EXCEPTION(IF, THROW) if(IF) throw std::runtime_error(THROW);

#endif