#pragma once
/***************************************
* Defines signature pragmas and templates
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef SIGNATURES_H
#define SIGNATURES_H

// ################################################################################################################
#ifndef _PRAGMA_CPP
#define _PRAGMA_CPP

#define RETURNS(...)		-> decltype((__VA_ARGS__))		{ return (__VA_ARGS__); }								// for quickly returning values
#define DOES(...)											{ return (__VA_ARGS__); }								// for single line void functions

#include <utility>
// ################################################################################################################

#define DECL_ENUM_ELEMENT( element )		#element
#define BEGIN_ENUM( ENUM_NAME )				static const char* eSTR##ENUM_NAME []				=
#define END_ENUM( ENUM_NAME )					; inline const char* getSTR_##ENUM_NAME(enum		\
																		ENUM_NAME index)							\
														{ return eSTR##ENUM_NAME [index]; };
#define BEGIN_ENUMC( ENUM_NAME )				static const char* eSTR##ENUM_NAME []				=
#define END_ENUMC( ENUM_NAME )				; inline const char* getSTR_##ENUM_NAME(uint		\
																					index)							\
														{ return eSTR##ENUM_NAME [index]; };
#define BEGIN_ENUM_INLINE(ENUM_NAME)		static const inline char* eSTR##ENUM_NAME []		=
#define END_ENUM_INLINE(ENUM_NAME, CLASS)	; static const char* getSTR_##ENUM_NAME(uint		\
																					index)							\
														{ return CLASS::eSTR##ENUM_NAME [index];	};

// ################################################################################################################

#ifdef _MSC_VER
	#include <intrin.h>
	#include <nmmintrin.h>
	#define __builtin_popcount __popcnt
	#define __builtin_popcountll _mm_popcnt_u64
#endif

// ################################################################################################################

#include <string>
#include <filesystem>

/**
* @brief Returns the canonical form of a given file path.
*
* This function attempts to convert the provided file path to its canonical form,
* which is an absolute path with all symbolic links and relative path components resolved.
* If the conversion fails for any reason, the original file path is returned.
*
* @param file The file path to be converted to its canonical form.
* @return A string representing the canonical form of the file path, or the original file path if an error occurs.
*/
inline std::string canonical_file(const char* file)
{
    try {
        return std::filesystem::canonical(file).string();
    }
    catch (...) {
        return file;
    }
}
#define FILE_LINE "[" << canonical_file(__FILE__) << ":" << __LINE__ << "]"

#if defined(_DEBUG)
    #include <iostream>
    #include <string>

    // Helper macros to convert macros to string
    #define STRINGIFY(x) #x
    #define TOSTRING(x) STRINGIFY(x)

    // Macro to extract class name, file, and line information
    #define CLASS_NAME(type) typeid(type).name()
    #define FUN_SIGNATURE __func__
    
    // Enhanced constructor/destructor call logging
    #define DESTRUCTOR_CALL         \
        ::std::cout << FILE_LINE << " " << FUN_SIGNATURE << " ->\t destructor called" << ::std::endl;
    #define DESTRUCTOR_CALL_T(type) \
        ::std::cout << FILE_LINE << " " << CLASS_NAME(type) << "::" << FUN_SIGNATURE << " ->\t destructor called" << ::std::endl;
    #define CONSTRUCTOR_CALL        \
        ::std::cout << FILE_LINE << " " << FUN_SIGNATURE << " ->\t constructor called" << ::std::endl;
    #define CONSTRUCTOR_CALL_T(type)\
        ::std::cout << FILE_LINE << " " << CLASS_NAME(type) << "::" << FUN_SIGNATURE << " ->\t constructor called" << ::std::endl;
#else
    // Define as empty when not in debug mode
    #define DESTRUCTOR_CALL
    #define DESTRUCTOR_CALL_T(type)
    #define CONSTRUCTOR_CALL
    #define CONSTRUCTOR_CALL_T(type)
#endif

// ################################################################################################################

#endif	// _PRAGMA_CPP
#endif	// SIGNATURES_H