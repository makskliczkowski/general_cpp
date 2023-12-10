#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "../xoshiro_pp.h"
#include <random>
#include <ctime>
#include <numeric>
// #include <ranges>

// --- RANGES ---
#ifdef __has_include
#	if __has_include(<ranges>)
#		include <ranges>
#   	define have_ranges 1
namespace rng = std::ranges;
#	elif __has_include(<experimental/ranges>)
#		include <experimental/ranges>
#    	define have_ranges 1
#    	define experimental_ranges
namespace rng = std::experimental::ranges;
#	else
#		define have_ranges 0
#	endif
#endif

// -------------------------------------------------------- RANDOM NUMBER CLASS --------------------------------------------------------

/*
* @brief Random number generator class based on Xorshiro256
* @link 
*/
class randomGen {
private:
	XoshiroCpp::Xoshiro256PlusPlus engine;
	std::uint64_t seed_ = 0;
public:
	explicit randomGen(std::uint64_t seed = std::random_device{}()) 
	{
		this->newSeed(seed);
		this->seed_ = seed;
		srand((unsigned int)seed);
	}

	// #################### S E E D   I N I T I A L I Z E R ##################

	/*
	* @brief Creates random seed based on 64bit unsigned integer
	* @parma n 64bit unsigned integer
	* @returns new seed for Xavier256 initializer
	*/
	static auto seedInit(uint64_t n)		->uint64_t
	{
		uint64_t z = (n += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}

	/*
	* @brief initialize seed
	*/
	auto newSeed(std::uint64_t seed)		-> void				{ this->engine = XoshiroCpp::Xoshiro256PlusPlus(randomGen::seedInit(seed)); };
	auto seed()								-> std::uint64_t	{ return this->seed_; }


	// --------------------- WRAPPERS ON RANDOM FUNCTIONS ---------------------

	/*
	* @brief Creates a Xavier random variable
	* @link https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
	*/
	template <typename T>
	auto xavier(T in, T out, float xav = 6) -> T				{ return std::uniform_real_distribution<T>(-1., +1.)(engine) * std::sqrt(xav / (in + out)); };

	template <typename T>
	auto kaiming(T in)						-> T				{ return std::uniform_real_distribution<T>(-1., +1.)(engine) * std::sqrt(6.0 / in); };

	// #################### U N I F O R M ####################

	/*
	* @brief random real uniform distribution : _min <= x < _max
	* @param _min smallest value
	* @param _max highest value
	*/
	template <typename T>
	auto random(T _min = 0, T _max = 1)		-> T				{ return std::uniform_real_distribution<T>(_min, _max)(engine); };

	/*
	* @brief random integer from range _min <= x < _max
	*/
	template <typename T>
	auto randomInt(T _min, T _max)			-> T				{ return static_cast<long long>(_min + static_cast<T>((_max - _min) * this->random<double>())); };

	// #################### N O R M A L ####################

	/*
	* @brief random normal distribution
	*/
	template <typename T>
	auto randomNormal(T _m = 0, T _s = 1)	-> T				{ return std::normal_distribution(_m, _s)(engine); };

	/*
	* @brief Bernoulli distributed random variable
	* @param p probability in the Bernoulli distribution
	*/
	template <typename T>
	auto bernoulli(T p)						-> T				{ return std::bernoulli_distribution(p)(engine); };

	// #################### O T H E R   T Y P E S ####################

	template <class _T>
	std::vector<_T> createRanVecStd(int _size, double _strength, _T _around = 0.0);
	template <class _T>
	COL<_T> createRanVec(int _size, double _strength, _T _around = 0.0);

	CCOL createRanState(uint _gamma);
	std::vector<CCOL> createRanState(uint _gamma, uint _realizations);

	// ####################### M A T R I C E S #######################
	DMAT GOE(uint _x, uint _y) const;
	CMAT CUE(uint _x, uint _y) const;

	// ####################### E L E M E N T S #######################
	template<typename _T>
	std::vector<_T> choice(const std::vector<_T>& _iterable, size_t _num);

};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Choose _num of elements out of some iterable
* @param _iterable iterable to choose from
* @param _num number of elemets
*/
template<typename _T>
inline std::vector<_T> randomGen::choice(const std::vector<_T>& _iterable, size_t _num)
{
	std::vector<_T> _out;
	// std::ranges(_iterable, std::back_inserter(_out), _num, this->engine);
	return _out;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a random vector based on disorder strength
* @param _size size of the vector
* @param _strength strength of the disorder used
* @param _around value around which we create random coefficients
* @returns a random vector from (-_strength + _around) to (_strength + _around)
*/
template <class _T>
inline COL<_T> randomGen::createRanVec(int _size, double _strength, _T _around)
{
	COL<_T> o(_size);
	for (auto i = 0; i < _size; i++)
		o(i) = _around + (this->random<double>() * 2.0 - 1.0) * _strength;
	return o;
}

/*
* @brief Creates a random vector based on disorder strength
* @param _size size of the vector
* @param _strength strength of the disorder used
* @param _around value around which we create random coefficients
* @returns a random vector from (-_strength + _around) to (_strength + _around)
*/
template <class _T>
inline std::vector<_T> randomGen::createRanVecStd(int _size, double _strength, _T _around)
{
	std::vector<_T> o(_size);
	for (auto i = 0; i < _size; i++)
		o[i] = _around + (this->random<double>() * 2.0 - 1.0) * _strength;
	return o;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Create multiple realizations of random states of _gamma coefficients
* @param _gamm number of combinations
* @param _realizations number of realizations of the random state
*/
inline std::vector<CCOL> randomGen::createRanState(uint _gamma, uint _realizations)
{
	std::vector<CCOL> _HM = {};

	// go through gammas
	for (int j = 0; j < _realizations; ++j)
		if (_gamma > 1)
			_HM.push_back(createRanState(_gamma));
		else
			_HM.push_back({ 1.0 });
	return _HM;
}

/*
* @brief Generates random superposition of _gamma states (the states shall not repeat, I guess...)
*/
inline arma::Col<std::complex<double>> randomGen::createRanState(uint _gamma)
{
	if (_gamma <= 1)
		return arma::Col<std::complex<double>>(1, arma::fill::eye);
	return this->CUE(_gamma, _gamma) * (arma::Col<std::complex<double>>(_gamma, arma::fill::eye));
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a GOE matrix...
*/ 
inline arma::Mat<double> randomGen::GOE(uint _x, uint _y) const
{
	arma::Mat<double> A(_x, _y, arma::fill::randn);
	return 0.5 * (A + A.t());
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a CUE matrix...
* A Random matrix distributed with Haar measure...
* ... https://doi.org/10.48550/arXiv.math-ph/0609050 ...
*/
inline arma::Mat<std::complex<double>> randomGen::CUE(uint _x, uint _y) const
{
	arma::Mat<std::complex<double>> A(_x, _y, arma::fill::zeros);
	A.set_real(arma::Mat<double>(_x, _y, arma::fill::randn));
	A.set_imag(arma::Mat<double>(_x, _y, arma::fill::randn));

	arma::Mat<std::complex<double>> Q, R;
	arma::qr(Q, R, A);
	return Q;
	//auto _diag	= R.diag();
	//_diag		= _diag / arma::abs(_diag);
	//return Q * DIAG(_diag) * Q;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#endif // !RANDOM_H
