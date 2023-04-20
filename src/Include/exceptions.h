#pragma once
#include <exception>
#include <stdexcept>
#include "signatures.h"
#include "str.h"

/*******************************
* Contains the possible methods
* for handling the exceptions.
*******************************/

#define LOG_LVL0		""
#define LOG_LVL1		"\t->"
#define LOG_LVL2		"\t\t->"
#define LOG_LVL3		"\t\t\t->"
#define LOG_LVL4		"\t\t\t\t->"

// ######################################################## E X C E P T I O N S ########################################################
namespace EXCEPTIONENUM {
	enum EXCEPTIONS
	{
		UNDEFINED = 0,
		RUNTIME = -1,
		BAD_ALOC = -2,
		EXCEPTION = -3,
		FILE = -4
	};

	BEGIN_ENUM(EXCEPTIONS)
	{
		DECL_ENUM_ELEMENT(UNDEFINED),
			DECL_ENUM_ELEMENT(RUNTIME),
			DECL_ENUM_ELEMENT(BAD_ALOC),
			DECL_ENUM_ELEMENT(EXCEPTION),
			DECL_ENUM_ELEMENT(FILE)
	}
	END_ENUM(EXCEPTIONS);
};

class ExceptionHandler {
public:
	static void printException(const std::string& _what, const std::string& _msg, EXCEPTIONENUM::EXCEPTIONS _ex = EXCEPTIONENUM::EXCEPTIONS::UNDEFINED) {
		auto exIDX = SSTR(EXCEPTIONENUM::getSTR_EXCEPTIONS(_ex));
		std::cout << LOG_LVL0 << exIDX << std::endl;
		std::cout << LOG_LVL1 << _what << std::endl;
		std::cout << LOG_LVL1 << _msg << std::endl;
		exit(static_cast<int>(_ex));
	};
	static void handleExceptions(std::exception_ptr _ePtr, const std::string& _msg);
};

#define BEGIN_CATCH_HANDLER				try{
#define END_CATCH_HANDLER(message)		}catch(...){ ExceptionHandler::handleExceptions(std::current_exception(), message); }
