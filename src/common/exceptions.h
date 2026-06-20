/******************************************************************************
 *
 *  @file src/Include/exceptions.h
 *  @brief Custom exception handling classes and catch handlers.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#include <exception>
#include <stdexcept>
#ifndef SIGNATURES_H
#	include "signatures.h"
#endif
#include "str.h"

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

class ExceptionHandler 
{
public:
	static void printException(const std::string& _what, const std::string& _msg, EXCEPTIONENUM::EXCEPTIONS _ex = EXCEPTIONENUM::EXCEPTIONS::UNDEFINED);
	static void handleExceptions(std::exception_ptr _ePtr, const std::string& _msg);
};

#define BEGIN_CATCH_HANDLER				try
#define END_CATCH_HANDLER(message, DO)	catch(...){ ExceptionHandler::handleExceptions(std::current_exception(), message); DO;}

#define IFELSE_EXCEPTION(IF, IFDO, THROW) if(IF) IFDO; else throw std::runtime_error(THROW);
#define IF_EXCEPTION(IF, THROW) if(IF) throw std::runtime_error(THROW);

#endif