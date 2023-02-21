#pragma once
#ifndef COMMON_H
#define COMMON_H

// ########################################################				RANDOM				########################################################
#include "random.h"

// ########################################################				 ARMA				########################################################

#ifndef ALG_H
	#include "LinearAlgebra/alg.h"
#endif // !ALG_H

// ########################################################				 OTHER				########################################################

// --- SUPPRESS WARNINGS ---

#ifndef STR_H
	#include "str.h"
#endif
#include "directories.h"
#include "files.h"

#include <cmath>
#include <omp.h>
#include <thread>
#include <complex>





#define RETURNS(...) -> decltype((__VA_ARGS__)) { return (__VA_ARGS__); }
// --------------------------------------------------------				DEFINITIONS				--------------------------------------------------------

							// standard out conditional

#define DIAG arma::diagmat

#define EYE(X) arma::eye(X,X)
#define ZEROV(X) arma::zeros(X)
#define ZEROM(X) arma::zeros(X,X)
#define SPACE_VEC_D(Lx, Ly, Lz) v_3d<double>(Lx, v_2d<double>(Ly, v_1d<double>(Lz, 0)))
#define SPACE_VEC(Lx, Ly, Lz) v_3d<int>(Lx, v_2d<int>(Ly, v_1d<int>(Lz, 0)))

/// using types
using cpx = std::complex<double>;
using uint = unsigned int;
using ul = unsigned long;
using ull = unsigned long long;
using ld = long double;

/// constexpressions
constexpr long double PI = 3.141592653589793238462643383279502884L;			// it is me, pi
constexpr long double TWOPI = 2 * PI;										// it is me, 2pi
constexpr long double PI_half = PI / 2.0;									// it is me, half a pi
constexpr cpx imn = cpx(0, 1);												// complex number
const auto global_seed = std::random_device{}();							// global seed for classes
// --------------------------------------------------------				ALGORITHMS FOR MC				--------------------------------------------------------

/*
@brief Here we will state all the already implemented definitions that will help us building the user interfrace
*/
namespace impDef
{
	/*
	/// Different Monte Carlo algorithms that can be provided inside the classes (for simplicity in enum form)
	*/
	enum class algMC {
		metropolis,
		heat_bath,
		self_learning
	};
	/*
	/// Types of implemented lattice types
	*/
	enum lattice_types {
		square,
		hexagonal
		//triangle,
		//hexagonal
	};

	enum ham_types {
		ising = 0,
		heisenberg = 1,
		heisenberg_dots = 2,
		kitaev_heisenberg = 3
	};
}

// --------------------------------------------------------				COMMON UTILITIES				 --------------------------------------------------------
using namespace arma;
using vecMat = v_1d<arma::mat>;


//? ------------------------------------------------------------------------------ VALUE EQUALS ------------------------------------------------------------------------------

#define EQP(value, equals, prec) valueEqualsPrecision(value, equals, prec)
#define EQ(value, equals) valueEqualsPrecision(value, equals)
/*
* @brief Checks if value is equal to some param up to given tolerance
*/
template <typename _T1, typename _T2, typename _T3>
inline bool valueEqualsPrecision(_T1 value, _T2 equals, _T3 tolerance) {
	return std::abs(value - equals) <= tolerance;
}

template <typename _T1, typename _T2>
inline bool valueEqualsPrecision(_T1 value, _T2 equals) {
	return std::abs(value - equals) == 0;
}

#define STR std::to_string
#define STRP(str,prec) str_p(str, prec)
/*
*@brief Changes a value to a string with a given precision
*@param v Value to be transformed
*@param n Precision default 2
*@returns String of a value
*/
template <typename _T>
inline std::string str_p(const _T v, const int n = 2) {
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << v;
	return out.str();
}

#define VEQ(name) valueEquals(#name,(name),2)
#define VEQP(name,prec) valueEquals(#name,(name),prec)
/*
* @brief Given the char* name it prints its value in a format "name=val"
* @param name name of the variable
* @param value value of the variable
* @returns "name=val" string
*/
template <typename T>
inline std::string valueEquals(const char name[], T value, int prec = 2) {
	return std::string(name) + "=" + str_p(value, prec);
}

inline std::string valueEquals(const char name[], std::string value, int prec) {
	return std::string(name) + "=" + value;
}

/*
* @brief pretty prints the complex number in angular form
* @param val complex value
* @n precision
*/
inline std::string print_cpx(cpx val, int n = 2) {
	double phase = std::arg(val) / PI;
	while (phase < 0)
		phase += 2;
	auto absolute = "+" + str_p(std::abs(val), n);
	std::string phase_str = "";
	if (valueEqualsPrec(phase, 0.0, 1e-3) || valueEqualsPrec(phase, 2.0, 1e-3))
		phase_str = "";
	else if (valueEqualsPrec(phase, 1.0, 1e-3)) {
		absolute = "-" + str_p(std::abs(val), n);;
		phase_str = "";
	}
	else {
		phase_str = "*exp(" + str_p(phase, n) + "*pi*i)";
	}


	return absolute + phase_str;
}



// -------------------------------------------------------------------------------------------- PRINT SEPARATED -----------------------------------------------------------------------------

/*
* printing the separated number of variables using the variadic functions initializer
*@param output output stream
*@param elements initializer list of the elements to be printed
*@param separator to be used @n default "\\t"
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*/
template <typename T>
inline void printSeparated(std::ostream& output, char separtator = '\t', std::initializer_list<T> elements = {}, arma::u16 width = 8, bool endline = true) {
	for (auto elem : elements) {
		output.width(width); output << elem << std::string(1, separtator);
	}
	if (endline) output << std::endl;
}


/*
* printing the separated number of variables using the variadic functions initializer -> ONE TYPE FUNCTION FOR RECURSION
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param elements at the very end we give any type of variable to the function
*/
template <typename Type>
inline void printSep(std::ostream& output, char separator, arma::u16 width, Type arg) {
	output.width(width); output << arg << std::string(1, separator);
}
/*
* printing the separated number of variables using the variadic functions initializer -> ONE TYPE FUNCTION FOR RECURSION - PRECISION!
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param elements at the very end we give any type of variable to the function
*@param prec precision for the output
*/
template <typename Type>
inline void printSepP(std::ostream& output, char separator, arma::u16 width, u16 prec, Type arg) {
	output.width(width); output << str_p(arg, prec) << std::string(1, separator);
}

/*
* printing the separated number of variables using the variadic functions initializer
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param arg first element of the argument list
*@param elements at the very end we give any type of variable to the function
*/
template <typename Type, typename... Types>
inline void printSep(std::ostream& output, char separator, arma::u16 width, Type arg, Types... elements) {
	printSep(output, separator, width, arg);
	printSep(output, separator, width, elements...);
}

/*
* printing the separated number of variables using the variadic functions initializer PRECISION
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param arg first element of the argument list
*@param elements at the very end we give any type of variable to the function
*/
template <typename Type, typename... Types>
inline void printSepP(std::ostream& output, char separator, arma::u16 width, u16 prec, Type arg, Types... elements) {
	printSepP(output, separator, width, prec, arg);
	printSepP(output, separator, width, prec, elements...);
}

/*
* printing the separated number of variables using the variadic functions initializer - LAST CALL
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param elements at the very end we give any type of variable to the function
*/
template <typename... Types>
inline void printSeparated(std::ostream& output, char separator, arma::u16 width, bool endline, Types... elements) {
	printSep(output, separator, width, elements...);
	if (endline) output << std::endl;
}

/*
*@brief printing the separated number of variables using the variadic functions initializer - LAST CALL PRECISION
*@param output output stream
*@param separator to be used
*@param width of one element column for printing
*@param endline shall we add endline at the end?
*@param elements at the very end we give any type of variable to the function
*/
template <typename... Types>
inline void printSeparatedP(std::ostream& output, char separator, arma::u16 width, bool endline, u16 prec, Types... elements) {
	printSepP(output, separator, width, prec, elements...);
	if (endline) output << std::endl;
}

// ######################################################## STREAM OVERLOADED ########################################################

/*
*@brief Overwritten standard stream redirection operator for 2D vectors separated by commas
*@param out outstream to be used
*@param v 1D vector
*/
template <typename T>
std::ostream& operator<< (std::ostream& out, const v_1d<T>& v) {
	if (!v.empty()) {
		for (int i = 0; i < v.size(); i++)
			out << v[i] << ",";
		out << "\b"; // use two ANSI backspace characters '\b' to overwrite final ", "
	}
	return out;
}

/*
* @brief Overwritten standard stream redirection operator for 2D vectors
* @param out outstream to be used
* @param v 2D vector
*/
template <typename T>
std::ostream& operator<< (std::ostream& out, const v_2d<T>& v) {
	if (!v.empty())
		for (auto it : v)
			out << "\t" << it << EL;
	return out;
}

std::ostream& operator<< (std::ostream& out, const cpx v)
{
	auto phase = std::arg(v) / PI;
	while (phase < 0)
		phase += 2.0;
	std::string absolute = "+" + STRP(std::abs(v),2);
	std::string phase_str = "";
	if (EQP(phase, 0.0, 1e-3) || EQP(phase, 2.0, 1e-3))
		phase_str = "";
	else if (EQP(phase, 1.0, 1e-3)) {
		absolute = "-" + STRP(std::abs(v),2);
		phase_str = "";
	}
	else
		phase_str = "*exp(" + STRP(phase,2) + "*pi*i)";
	out << absolute + phase_str;
	return out;
}

//! ----------------------------------------------------------------------------- HELPERS -----------------------------------------------------------------------------

/*
* @brief check the sign of a value
* @param val value to be checked
* @returns sign of a variable
*/
template <typename T>
inline int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

/*
* @brief Defines an euclidean modulo denoting also the negative sign
* @param a left side of modulo
* @param b right side of modulo
* @returns euclidean a%b
* @link https://en.wikipedia.org/wiki/Modulo_operation
*/
inline int myModuloEuclidean(int a, int b)
{
	int m = a % b;
	if (m < 0) m = (b < 0) ? m - b : m + b;
	return m;
}


// ----------------------------------------------------------------------------- VECTORS HANDLING -----------------------------------------------------------------------------




template <typename T, typename T2>
inline void print_vector_1d(T& file, const v_1d<T2>& v) {
	for (auto i = 0; i < v.size(); i++)
		printSeparatedP(file, '\t', 8, true, 5, i, v[i]);
}

template <typename T, typename T2>
inline void print_vector_2d(T& file, const v_2d<T2>& v) {
	for (auto i = 0; i < v.size(); i++)
		print_vector_1d(file, v[i]);
}

template <typename T, typename T2>
inline void print_vector_3d(T& file, const v_3d<T2>& v) {
	//for (auto i = 0; i < v.size(); i++)
	//	print_vector_2d(file, v[i]);
	for (auto i = 0; i < v.size(); i++)
		for (auto j = 0; j < v[i].size(); j++)
			for (auto k = 0; k < v[i][j].size(); k++)
				printSeparatedP(file, '\t', 8, true, 5, i, j, k, v[i][j][k]);
}

template <typename T, typename T2>
inline void print_mat(T& file, const Mat<T2>& m) {
	//for (auto i = 0; i < v.size(); i++)
	//	print_vector_2d(file, v[i]);
	for (auto i = 0; i < m.n_rows; i++)
		for (auto j = 0; j < m.n_cols; j++)
			printSeparatedP(file, '\t', 8, true, 5, i, j, m(i, j));
}



template <typename T, typename T2>
inline void print_vector_1d(T& file, const Col<T2>& v) {
	for (auto i = 0; i < v.size(); i++)
		printSeparatedP(file, '\t', 8, true, 5, i, v(i));
}

/*
* @brief take real part of a complex vector
*/
//v_1d<double> realv(const v_1d<cpx>& v) {
//	v_1d<double> tmp(v.size(), 0);
//	for (auto i = 0; i < v.size(); i++)
//		tmp[i] = real(v[i]);
//	return tmp;
//}
//
//
///*
//* @brief take imaginary part of a complex vector
//*/
//v_1d<double> imagv(const v_1d<cpx>& v) {
//	v_1d<double> tmp(v.size(), 0);
//	for (auto i = 0; i < v.size(); i++)
//		tmp[i] = imag(v[i]);
//	return tmp;
//}

/*
*
*/
template <typename T>
T stddev(const v_1d<T>& v)
{
	T sum = std::accumulate(v.begin(), v.end(), cpx(0.0));
	T mean = sum / cpx(v.size());

	std::vector<T> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(), [mean](T x) { return x - mean; });
	T sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), cpx(0.0));
	T stdev = std::sqrt(sq_sum / cpx(v.size()));
	return stdev;
}


/// <summary>
/// Creates a random vector of custom length using the random library and the merson-twister (?) engine
/// </summary>
/// <param name="N"> length of the generated random vector </param>
/// <returns> returns the custom-length random vector </returns>
inline vec create_random_vec(u64 N, randomGen& gen, double h = 1.0) {
	vec random_vec(N, fill::zeros);
	// create random vector from middle to always append new disorder at lattice endpoint
	for (u64 j = 0; j <= N / 2.; j++) {
		u64 idx = N / (long)2 - j;
		random_vec(idx) = gen.randomReal_uni(-h, h);
		idx += 2 * j;
		if (idx < N) random_vec(idx) = gen.randomReal_uni(-h, h);
	}
	return random_vec;
}

/// <summary>
/// Creates a random vector of custom length using the random library and the merson-twister (?) engine
/// </summary>
/// <param name="N"> length of the generated random vector </param>
/// <returns> returns the custom-length random vector </returns>
inline std::vector<double> create_random_vec_std(u64 N, randomGen& gen, double h = 1.0) {
	std::vector<double> random_vec(N, 0);
	for (u64 j = 0; j < N; j++) {
		random_vec[j] = gen.randomReal_uni(-h, h);
	}
	return random_vec;
}

#endif // !COMMON_H
