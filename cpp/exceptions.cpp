/******************************************************************************
 *
 *  @file cpp/exceptions.cpp
 *  @brief Implementations for custom exception handling.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#include "../src/common/exceptions.h"
#include <iostream>

void ExceptionHandler::printException(const std::string& _what, const std::string& _msg, EXCEPTIONENUM::EXCEPTIONS _ex)
{
	auto exIDX = EXCEPTIONENUM::getSTR_EXCEPTIONS(_ex);
	std::cout << LOG_LVL0 << _msg << std::endl;
	std::cout << LOG_LVL1 << "Exception: " << exIDX << std::endl;
	std::cout << LOG_LVL2 << _what << std::endl;
	std::exit(static_cast<int>(_ex));
}

/*
* @brief handles the most common exceptions
*/
void ExceptionHandler::handleExceptions(std::exception_ptr _ePtr, const std::string& _msg)
{
	try {
		if (_ePtr) std::rethrow_exception(_ePtr);
	}
	catch (const std::runtime_error& err) {
		printException(err.what(), _msg, EXCEPTIONENUM::EXCEPTIONS::RUNTIME);
	}
	catch (const std::bad_alloc& err) {
		printException(err.what(), _msg, EXCEPTIONENUM::EXCEPTIONS::BAD_ALOC);
	}
	catch (const std::exception& err) {
		printException(err.what(), _msg, EXCEPTIONENUM::EXCEPTIONS::EXCEPTION);
	}
	catch (const std::string& err) {
		printException(err, _msg, EXCEPTIONENUM::EXCEPTIONS::RUNTIME);
	}
	catch (const char* err) {
		printException(err, _msg, EXCEPTIONENUM::EXCEPTIONS::RUNTIME);
	}
	catch (...) {
		printException("UNKNOWN EXCEPTION", _msg, EXCEPTIONENUM::EXCEPTIONS::BAD_ALOC);
	};
}