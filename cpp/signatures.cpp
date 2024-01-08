#include "../src/Include/signatures.h"

//! check compiler version, only C++17 or newer currently valid for this library
//! older versions are not suppeortd
#if !defined(_MSVC_LANG)
	#if (__cplusplus >= 202002L)
		#define HAS_CXX20
	#elif (__cplusplus >= 201703L)
		#define HAS_CXX17
	#else
		#error "--> at least C++17 compiler required; older versions are not supported"
	#endif
#else 
	#if (_MSVC_LANG >= 202002L)
		#define HAS_CXX20 true
	#elif (_MSVC_LANG >= 201703L)
		#define HAS_CXX17 true
	#else
		#error "--> at least C++17 compiler required; older versions are not supported"
	#endif
#endif

// WHAT IF 2020?
#ifdef HAS_CXX20
	#pragma message ("--> Compiling with c++20 compiler")
	#define callable_type std::invocable
	#ifdef __has_include
		#if __has_include(<format>)
			//#include <format>
			//#define HAS_FORMAT
		#endif
	#endif
	#elif defined HAS_CXX17
		#pragma message ("--> Compiling with c++17 compiler")
		#define callable_type typename
#endif

// ######################################################## F U N C T I O N S ########################################################

/*
* Define function signatures to use in debug scenarios
*/
#if !defined(FUN_SIGNATURE)
	#if defined (__GNUG__)
		#define FUN_SIGNATURE  __PRETTY_FUNCTION__
		#pragma message ("--> Using GNU compiler")
	#elif defined (_MSC_VER)
		#define FUN_SIGNATURE  __FUNCSIG__ 
		#pragma message ("--> Using MSVS compiler")
	#elif defined(__INTEL_COMPILER)
		#define FUN_SIGNATURE  __FUNCTION__
		#pragma message ("--> Using INTEL compiler")
	#else 
		#define FUN_SIGNATURE  __func__
		#pragma message ("--> Using OTHER compiler")
	#endif
#endif