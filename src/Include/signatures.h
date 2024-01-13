#pragma once
/***************************************
* Defines signature pragmas and templates
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef SIGNATURES_H
#define SIGNATURES_H

// ################################################# C P P   V E R S I O N ##################################################
#ifndef _PRAGMA_CPP
#define _PRAGMA_CPP

#define RETURNS(...)		-> decltype((__VA_ARGS__))		{ return (__VA_ARGS__); }								// for quickly returning values
#define DOES(...)											{ return (__VA_ARGS__); }								// for single line void functions

#include <utility>
// ######################################################## E N U M S #######################################################

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

// ######################################################## C O U N T E R S ########################################################

#ifdef _MSC_VER
	#include <intrin.h>
	#include <nmmintrin.h>
	#define __builtin_popcount __popcnt
	#define __builtin_popcountll _mm_popcnt_u64
#endif

// ######################################################## C A L L S ########################################################
#if defined(_DEBUG)
#	define FUN_SIGNATURE		__func__
#	define DESTRUCTOR_CALL		std::cout << FUN_SIGNATURE << "->\t destructor called" << std::endl << std::endl;
#	define CONSTRUCTOR_CALL		std::cout << FUN_SIGNATURE << "->\t constructor called" << std::endl << std::endl;
#else 
#	define DESTRUCTOR_CALL 
#	define CONSTRUCTOR_CALL
#endif
#endif

#endif