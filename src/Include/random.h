#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "../xoshiro_pp.h"
#include <random>
#include <ctime>
#include <numeric>

// -------------------------------------------------------- RANDOM NUMBER CLASS --------------------------------------------------------

/*
* @brief Random number generator class based on Xorshiro256
* @link 
*/
class randomGen {
private:
	XoshiroCpp::Xoshiro256PlusPlus engine;
public:
	explicit randomGen(std::uint64_t seed = std::random_device{}()) 
	{
		this->newSeed(seed);
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

	/*
	* @brief creates random vector with a given strength
	*/
	arma::Col<double> createRanVec(int _size, double _strength);
};

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a random vector based on disorder strength
* @param _size size of the vector
* @param _strength strength of the disorder used
* @returns a random vector from -_strength to _strength
*/
inline arma::Col<double> randomGen::createRanVec(int _size, double _strength)
{
	arma::Col<double> o(_size);
	for (auto i = 0; i < _size; i++)
		o(i) = (this->random<double>() * 2.0 - 1.0) * _strength;
	return o;
}

#endif // !RANDOM_H
