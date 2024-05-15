#ifndef RANDOM_H
#define RANDOM_H

#include "../xoshiro_pp.h"
#include "../lin_alg.h"
#include <random>
#include <ctime>
#include <numeric>
#include <type_traits>

// --- RANGES ---
#ifdef __has_include
#	if __has_include(<ranges>)
#		include <ranges>
#   	define HAS_RANGES 1
namespace rng = std::ranges;
#	elif __has_include(<experimental/ranges>)
#		include <experimental/ranges>
#    	define have_ranges 1
#    	define experimental_ranges
namespace rng = std::experimental::ranges;
#	else
#		define HAS_RANGES 0
#	endif
#endif

// declaration
namespace algebra
{
	template <typename _T, typename _Tin>
	inline auto cast(_Tin x) -> _T;

}

// -------------------------------------------------------- RANDOM NUMBER CLASS --------------------------------------------------------

/*
* @brief Random number generator class based on Xorshiro256
* @link 
*/
class randomGen 
{
private:
	XoshiroCpp::Xoshiro256PlusPlus engine;
	std::uint64_t seed_ = 0;
public:

	explicit randomGen(std::uint64_t seed = std::random_device{}());

	// #################### S E E D   I N I T I A L I Z E R ##################

	static auto seedInit(uint64_t n) -> uint64_t;

	// -----------------------------------------------------------------------

	auto newSeed(std::uint64_t seed)			-> void									{ this->engine = XoshiroCpp::Xoshiro256PlusPlus(randomGen::seedInit(seed)); };
	auto seed()									const -> std::uint64_t					{ return this->seed_; }
	auto eng()								const -> XoshiroCpp::Xoshiro256PlusPlus		{ return this->engine; }
	// --------------------- WRAPPERS ON RANDOM FUNCTIONS ---------------------

	template <typename _T, typename _T2 = _T>
	auto random(_T _min = 0, _T2 _max = 1)		-> typename std::common_type<_T, _T2>::type;

	template <typename _T, typename _T2, template <class> typename _V>
	auto random(_T _mn, _T2 _mx, size_t _s)		-> _V<typename std::common_type<_T, _T2>::type>;

	// ---------------------

	template <typename _T, typename _T2 = _T>
	auto randomInt(_T _min, _T2 _max)			-> typename std::common_type<_T, _T2>::type;

	template <typename _T, typename _T2, template <class> typename _V>
	auto randomInt(_T _mn, _T2 _mx, size_t _s)  -> _V<typename std::common_type<_T, _T2>::type>;

	// ---------------------

	template<typename _T, typename _T2 = _T>
	auto randomNormal(_T _m = 0, _T2 _s = 1)	-> typename std::common_type<_T, _T2>::type;

	template <typename _T, typename _T2, template <class> typename _V>
	auto randomNormal(_T _m, _T2 _ss, size_t _s)-> _V<typename std::common_type<_T, _T2>::type>;

	template <typename T>
	auto xavier(T in, T out, float xav = 6)		-> T;

	template <typename T>
	auto kaiming(T in) -> T;

	template <typename T>
	auto bernoulli(T p) -> T;

	// #################### O T H E R   T Y P E S ####################

	template<class _V, typename _T>
	_V rvector(size_t _size, _T _disorder, _T _around);

	template <class _T>
	std::vector<_T> createRanVecStd(int _size, double _strength, _T _around = 0.0);

	template <class _T>
	COL<_T> createRanVec(int _size, double _strength, _T _around = 0.0);

	template<typename _T>
	arma::Col<_T> createRanState(uint _gamma);

	template<typename _T>
	std::vector<arma::Col<_T>> createRanState(uint _gamma, uint _realizations);

	// ####################### M A T R I C E S #######################
	
	template <typename _T>
	arma::Mat<_T> GOE(uint _x, uint _y);
	template <typename _T>
	arma::Mat<_T> GOE(uint _x) { return GOE<_T>(_x, _x); };
	template <typename _T>
	arma::Mat<_T> GUE(uint _x, uint _y);
	template <typename _T>
	arma::Mat<_T> GUE(uint _x) { return GUE<_T>(_x, _x); };
	template <typename _T>
	arma::Mat<_T> CUE(uint _x, uint _y);
	template <typename _T>
	arma::Mat<_T> CUE(uint _x) { return CUE<_T>(_x, _x); };

	// ####################### E L E M E N T S #######################

	template<typename _T, typename _A = std::allocator<_T>>
	std::vector<_T, _A> choice(const std::vector<_T, _A>& _iterable, size_t _num);

	template<class _T>
	_T choice(_T begin, _T end, size_t _num);

};

// ######################################################################################################################

// ######################################################################################################################

inline randomGen::randomGen(std::uint64_t seed)
{
	this->newSeed(seed);
	this->seed_ = seed;
	srand((unsigned int)seed);
}

// ----------------------------------------------------------------------------------------------------------------------

/*
* @brief Creates random seed based on 64bit unsigned integer
* @param n 64bit unsigned integer
* @returns new seed for Xavier256 initializer
*/
inline uint64_t randomGen::seedInit(uint64_t n)
{
	uint64_t z	= (n += 0x9e3779b97f4a7c15);
	z			= (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z			= (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a uniform random variable
* @param _min minimum value
* @param _max maximum value
* @returns a uniform random variable
*/
template <typename _T, typename _T2>
inline typename std::common_type<_T, _T2>::type randomGen::random(_T _min, _T2 _max)
{
	using result_type = typename std::common_type<_T, _T2>::type;
	return std::uniform_real_distribution<result_type>(_min, _max)(this->engine);
}

template <typename _T, typename _T2, template <class> typename _V>
inline _V<typename std::common_type<_T, _T2>::type> randomGen::random(_T _mn, _T2 _mx, size_t _s)
{
	using result_type = typename std::common_type<_T, _T2>::type;
	_V<result_type> _out(_s);
	// generate random numbers
	std::generate(_out.begin(), _out.end(), [&]() { return std::uniform_real_distribution<result_type>(_mn, _mx - 1)(this->engine); });
	return _out;
}

// ---------------------------------------------------------------------------------------------------------------------

/*
* @brief Creates a normal random variable
* @param _m mean
* @param _s standard deviation
* @returns a normal random variable
*/
template <typename _T, typename _T2>
inline typename std::common_type<_T, _T2>::type randomGen::randomNormal(_T _m, _T2 _s)
{
	using result_type = typename std::common_type<_T, _T2>::type;
	return std::normal_distribution<result_type>(_m, _s)(this->engine);
}

template<typename _T, typename _T2, template <class> typename _V>
inline _V<typename std::common_type<_T, _T2>::type> randomGen::randomNormal(_T _m, _T2 _ss, size_t _s)
{
	using result_type = typename std::common_type<_T, _T2>::type;
	_V<result_type> _out(_s);

	// generate random numbers
	std::generate(_out.begin(), _out.end(), [&]() { return std::normal_distribution<result_type>(_m, _ss)(this->engine); });
	return _out;
}

// ---------------------------------------------------------------------------------------------------------------------

/*
* @brief Creates an integer random variable
* @param _min minimum value
* @param _max maximum value
* @returns an integer random variable
*/
template<typename _T, typename _T2>
inline typename std::common_type<_T, _T2>::type randomGen::randomInt(_T _min, _T2 _max)
{
	using result_type = typename std::common_type<_T, _T2>::type;
	return std::uniform_int_distribution<result_type>(_min, _max - 1)(this->engine);
}

template <typename _T, typename _T2, template <class> typename _V>
inline _V<typename std::common_type<_T, _T2>::type> randomGen::randomInt(_T _min, _T2 _max, size_t _size)
{
	using result_type = typename std::common_type<_T, _T2>::type;
	_V<result_type> _out(_size);
	// generate random numbers
	std::generate(_out.begin(), _out.end(), [&]() { return std::uniform_int_distribution<result_type>(_min, _max - 1)(this->engine); });
	return _out;
}

// ---------------------------------------------------------------------------------------------------------------------

/*
* @brief Creates a Xavier random variable
* @param in number of inputs
* @param out number of outputs
* @param xav factor
* @link https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
*/
template<typename T>
inline auto randomGen::xavier(T in, T out, float xav) -> T
{
	return std::uniform_real_distribution<T>(-1., +1.)(this->engine) * std::sqrt(xav / (in + out));
}

// ---------------------------------------------------------------------------------------------------------------------

template<typename T>
inline auto randomGen::kaiming(T in) -> T
{
	return std::uniform_real_distribution<T>(-1., +1.)(engine) * std::sqrt(6.0 / in);
}

// ---------------------------------------------------------------------------------------------------------------------

/*
* @brief Creates a bernoulli random variable
* @param p probability of success
* @returns a bernoulli random variable
*/
template<typename T>
inline T randomGen::bernoulli(T p)
{
	return std::bernoulli_distribution(p)(this->engine);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Choose _num of elements out of some iterable
* @param _iterable iterable to choose from
* @param _num number of elements
* @returns vector of choices
*/
template<typename _T, typename _A>
inline std::vector<_T, _A> randomGen::choice(const std::vector<_T, _A>& _iterable, size_t _num)
{
	std::vector<_T, _A> _out;
#if HAS_RANGES == 1
	rng::sample(_iterable, std::back_inserter(_out), _num, this->engine);
#else
	_out = _iterable;
	this->choice(_out.begin(), _out.end(), _num);
#endif // DEBUG
	return _out;
}

// ----------------------------------------------------------------------------------------------------------------------

/*
* @brief Uses the Fisher�Yates shuffle to obtain the random choice out of a container.
* @url https://stackoverflow.com/questions/9345087/choose-m-elements-randomly-from-a-vector-containing-n-elements
* @param begin begining of the container
* @param end end of the container
* @param _num number of elements
* @returns iterator to the element
*/
template<class _T>
inline _T randomGen::choice(_T begin, _T end, size_t _num)
{
	size_t left = std::distance(begin, end);
	while (_num--)
	{
		_T r = begin;
		std::advance(r, rand() % left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}
	return begin;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<class _V, typename _T>
inline _V randomGen::rvector(size_t _size, _T _disorder, _T _around)
{
	_V o(_size);
	for (auto i = 0; i < _size; ++i)
		o[i] = _around + (this->random<double>() * 2.0 - 1.0) * _disorder;
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
template<typename _T>
inline std::vector<arma::Col<_T>> randomGen::createRanState(uint _gamma, uint _realizations)
{
	std::vector<arma::Col<_T>> _HM = {};

	// go through gammas
	for (int j = 0; j < _realizations; ++j)
		if (_gamma > 1)
			_HM.push_back(createRanState<_T>(_gamma));
		else
			_HM.push_back({ 1.0 });
	return _HM;
}

// ------------------------------------------------------------------------------------------------------------------

/*
* @brief Generates random superposition of _gamma states (the states shall not repeat, I guess...)
*/
template<typename _T>
inline arma::Col<_T> randomGen::createRanState(uint _gamma)
{
	if (_gamma <= 1)
		return arma::Col<_T>(1, arma::fill::eye);
	arma::Col<_T> _state = this->CUE<_T>(_gamma, _gamma) * (arma::Col<_T>(_gamma, arma::fill::eye));
	return _state;
}

template<>
inline arma::Col<double> randomGen::createRanState(uint _gamma)
{
	if (_gamma <= 1)
		return arma::Col<double>(1, arma::fill::eye);
	arma::Col<double> _state = this->GOE<double>(_gamma, _gamma) * (arma::Col<double>(_gamma, arma::fill::eye));
	return _state;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a GOE matrix...
*/ 
template <typename _T>
inline arma::Mat<_T> randomGen::GOE(uint _x, uint _y)
{
	arma::Mat<_T> A(_x, _y, arma::fill::zeros);
	
	for(uint i = 0; i < _x; ++i)
		for(uint j = i; j < _y; ++j)
			A(i, j) = algebra::cast<_T>(this->randomNormal(0.0, 1.0));

	return std::sqrt(0.5) * (A + A.t());
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a GUE matrix...
*/
template<typename _T>
inline arma::Mat<_T> randomGen::GUE(uint _x, uint _y)
{
	arma::Mat<_T> A(_x, _y, arma::fill::zeros);
	for(uint i = 0; i < _x; ++i)
		for(uint j = i; j < _y; ++j)
			A(i, j) = this->randomNormal(0.0, 0.5) + I * this->randomNormal(0.0, 0.5);

	return std::sqrt(0.5) * (A + A.t());
}

template<>
inline arma::Mat<double> randomGen::GUE(uint _x, uint _y)
{
	return this->GOE<double>(_x, _y);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Creates a CUE matrix...
* A Random matrix distributed with Haar measure...
* ... https://doi.org/10.48550/arXiv.math-ph/0609050 ...
*/
template <typename _T>
inline arma::Mat<_T> randomGen::CUE(uint _x, uint _y)
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

template <>
inline arma::Mat<double> randomGen::CUE(uint _x, uint _y)
{
	arma::Mat<std::complex<double>> A(_x, _y, arma::fill::zeros);
	A.set_real(arma::Mat<double>(_x, _y, arma::fill::randn));
	A.set_imag(arma::Mat<double>(_x, _y, arma::fill::randn));

	arma::Mat<std::complex<double>> Q, R;
	arma::qr(Q, R, A);
	return arma::real(Q);
	//auto _diag	= R.diag();
	//_diag		= _diag / arma::abs(_diag);
	//return Q * DIAG(_diag) * Q;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#endif // !RANDOM_H
