#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <functional>
#include <algorithm>
#include <tuple>

#include "../Dynamic/dynamic_bitset.hpp"
#include "linalg/generalized_matrix.h"
#include "maths.h"

// #################################################### G E N E R A L ####################################################

namespace Containers
{
	/*
	* @brief Zip containers toghether. 
	* The decay_t performs the type conversions equivalent to the ones performed when passing function arguments by value. Formally:
	* - If T is "array of U" or reference to it, the member typedef type is U*. 
	* - Otherwise, if T is a function type F or reference to one, the member typedef type is std::add_pointer<F>::type. 
	* - Otherwise, the member typedef type is std::remove_cv<std::remove_reference<T>::type>::type. 
	* - The behavior of a program that adds specializations for std::decay is undefined. \n* 
	* decltype is useful when declaring types that are difficult or impossible to declare using standard notation, like lambda-related types or types that depend on template parameters. 
	* @param containers containers to zip
	*/
	template<typename... Container>
	auto zip(Container&... containers) noexcept 
	{
		// perform a decay type on iterators
		using tuple_type			= std::tuple<std::decay_t<decltype(*std::begin(containers))>...>;
		std::size_t container_size	= std::min({ std::size(containers)... });

		// create a vector for the result
		std::vector<tuple_type> result;
		result.reserve(container_size);

		// create a tuple of iterators
		auto iterators				= std::make_tuple(std::begin(containers)...);
		for (std::size_t i = 0; i < container_size; ++i) 
		{
			std::apply([&result](auto&... it) {	result.emplace_back(*it++...); }, iterators);
		}

		return result;
	}

	template<typename... Ts, typename... Containers>
	void unzip(const std::vector<std::tuple<Ts...>>& zipped, Containers&... containers) {
		// Reserve space in the containers
		((containers.reserve(zipped.size())), ...);

		// Unzip the elements into the containers
		for (const auto& tuple : zipped) 
		{
			std::apply([&](auto&&... args) 
				{
					((containers.push_back(args)), ...);
				}, 
				tuple);
		}
	}

	// ###################################################################################################################

	template<uint _On, typename... Ts>
	void sort(std::vector<std::tuple<Ts...>>& zipped,
			  std::function<bool(typename std::remove_reference_t<typename std::tuple_element<_On, std::tuple<Ts...>>>::type,
								 typename std::remove_reference_t<typename std::tuple_element<_On, std::tuple<Ts...>>>::type)> f)
	{
		std::sort(zipped.begin(), zipped.end(),
			[&](const auto& a, const auto& b)
			{
				//return std::get<_On>(a) < std::get<_On>(b);
				return f(std::get<_On>(a), std::get<_On>(b));
			});
	}

	// ###################################################################################################################
};

// ##################################################### V E C T O R #####################################################

// ################# I S   V E C T O R #################
template<typename T>
struct is_vector : std::false_type {};
template<typename T>
struct is_vector<std::vector<T>> : std::true_type {};

// #####################################################

namespace Vectors
{
	// ########################## C O N V E R S I O N ##########################

	/*
	* @brief Convert vector of one type to another
	* @param _v vector to convert
	* @returns converted vector
	*/
	template <typename _T, typename _T2 = double>
	inline v_1d<_T> convert(const v_1d<_T2>& _v)
	{
		v_1d<_T> _out;
		_out.resize(_v.size());
		std::transform(_v.begin(), _v.end(), _out.begin(), [](const _T& _elem) { return static_cast<_T2>(_elem); });
		return _out;
	}

	template <typename _T>
	inline v_1d<_T> convert(const v_1d<_T>& _v)
	{
		return _v;
	}

	template <typename _T>
	inline v_1d<_T> convert(const v_1d<std::string>& _v)
	{
		v_1d<_T> _out;
		_out.resize(_v.size());
		std::transform(_v.begin(), _v.end(), _out.begin(), [](const std::string& _elem) { return static_cast<_T>(std::stold(_elem)); });
		return _out;
	}

	// -------------------------------------------------------------------------

	template <>
	inline v_1d<size_t> convert(const v_1d<std::string>& _v)
	{
		v_1d<size_t> _out;
		_out.resize(_v.size());
		std::transform(_v.begin(), _v.end(), _out.begin(), [](const std::string& _elem) { return static_cast<size_t>(std::stoull(_elem)); });
		return _out;
	}


	// ######################### S T A T I S T I C A L #########################
	
	/*
	* @brief Calculates the mean of the vector
	* @param _v vector to calculate the mean of
	* @returns mean from vector samples
	*/
	template<typename _T, typename _B = std::allocator<_T>>
	inline _T mean(const std::vector<_T, _B>& _v)
	{
		if (_v.empty())
			return 0;
		return std::reduce(_v.begin(), _v.end()) / _v.size();
	}

	template<typename _T, typename _B = std::allocator<_T>>
	inline arma::Mat<_T> mean(const std::vector<arma::Mat<_T>, _B>& _v)
	{
		if (_v.empty())
			return arma::Mat<_T>(1, 1, arma::fill::zeros);
		arma::Mat<_T> _out = arma::Mat<_T>(_v[0].n_rows, _v[0].n_cols, arma::fill::zeros);
		for (uint i = 0; i < _v.size(); i++)
			_out += _v[i];
		return _out / _v.size();
	}

	// -------------------------------------------------------------------------

	/*
	* @brief Calculates the variance of the vector
	* @param _v vector to calculate the variance of
	* @returns variance from vector samples
	*/
	template<typename _T, typename _B = std::allocator<_T>>
	inline _T var(const std::vector<_T, _B>& _v)
	{
		if (_v.empty())
			return 0;
		_T _mean		= Vectors::mean(_v);
		_T _sqSum		= std::inner_product(_v.begin(), _v.end(), _v.begin(), 0.0);
		return _sqSum / _v.size() - _mean * _mean;
	}

	template<typename _T, typename _B = std::allocator<_T>>
	inline arma::Mat<_T> var(const std::vector<arma::Mat<_T>, _B>& _v)
	{
		if (_v.empty())
			return arma::Mat<_T>(1, 1, arma::fill::zeros);
		arma::Mat<_T> _out = arma::Mat<_T>(_v[0].n_rows, _v[1].n_cols, arma::fill::zeros);
		arma::Mat<_T> _mean= Vectors::mean<_T>(_v);
		for (uint i = 0; i < _v.size(); i++)
			_out += _v[i] * _v[i];
		return _out / _v.size() - _mean * _mean;
	}
	
	// -------------------------------------------------------------------------

	/*
	* @brief Calculates the standard deviation of the vector
	* @param _v vector to calculate the standard deviation of
	* @returns standard deviation from vector samples
	*/
	template<typename _T, typename _B = std::allocator<_T>>
	inline _T std(const std::vector<_T, _B>& _v)
	{
		return std::sqrt(Vectors::var(_v));
	}

	template<typename _T, typename _B = std::allocator<_T>>
	inline arma::Mat<_T> std(const std::vector<arma::Mat<_T>, _B>& _v)
	{
		return arma::sqrt(Vectors::var(_v));
	}

	// -------------------------------------------------------------------------

	/*
	* @brief Create frequency map that allows one to see the repetition of elements in a vector
	* @param _container check this container
	* @param _cut remove number of occurences less (or equal) than this (if bigger than 0, of course)
	*/
	template<typename _T, typename _B = std::allocator<_T>>
	inline std::unordered_map<_T, size_t> freq(const std::vector<_T, _B>& _container, uint _cut = 0)
	{
		std::unordered_map<_T, size_t> _freq;
		// go through elements
		for (auto& _elem : _container)
			++_freq[_elem];

		// check if we cut something
		if (_cut > 0)
		{
			std::erase_if(_freq, [&](const auto& elem)
				{
					auto const& [key, val] = elem;
					return val <= _cut;
				});
		}
		return _freq;
	}

	/*
	* @brief Create frequency map that allows one to see the repetition of elements in a vector
	* @param _container check this container
	* @param _cut remove number of occurences less (or equal) than this (if bigger than 0, of course)
	* @tparam _Trunc truncation value (truncates the number of specific bits to get the degeneracies)
	*/
	template<uint _Trunc, typename _T, typename _B = std::allocator<_T>>
	inline std::unordered_map<_T, size_t> freq(const std::vector<_T, _B>& _container, uint _cut = 0)
	{
		std::unordered_map<_T, size_t> _freq;
		// go through elements
		for (auto& _elem : _container)
			++_freq[Math::trunc<_T, _Trunc>(_elem)];

		// check if we cut something
		if (_cut > 0)
		{
			std::erase_if(_freq, [&](const auto& elem)
				{
					auto const& [key, val] = elem;
					return val <= _cut;
				});
		}
		return _freq;
	}

	// -------------------------------------------------------------------------

	/*
	* @brief Creates a set of combinations created from a given vector elements.
	* @param _iterable vector to create combinations from
	* @param _num number of combinations
	* @returns set of combinations in a form of a vector of vectors
	*/
	template<typename _T, typename _A = std::allocator<_T>>
	inline std::vector<std::vector<_T, _A>> combinations(const std::vector<_T, _A>& _iterable, size_t _num)
	{
		std::vector<std::vector<_T, _A>> _combinations;
		// get size of the iterable
		size_t N = _iterable.size();
		std::vector<bool> bitmask(_num, true);		// Initialize bitmask 
		bitmask.resize(N, 0);						// N-K trailing 0's

		// Helper function to generate combinations
		auto generateCombination = [&]() 
			{
				std::vector<_T, _A> _inner;
				for (size_t i = 0; i < N; ++i) 
				{
					if (bitmask[i]) 
						_inner.push_back(_iterable[i]);
				}
				_combinations.push_back(std::move(_inner));
			};

		// Generate combinations using bitmask
		do {
			generateCombination();
		} while (std::prev_permutation(bitmask.begin(), bitmask.end()));

		return _combinations;
	}

	// ###################### F R O M   A R M A D I L L O ######################

	/*
	* @brief Transform container type to std::vector of the same subtype
	* @param _in container with a given type
	* @returns std::vector of a given type
	*/
	template<template <class _Tin> class _T, class _Tin>
	inline std::vector<_Tin> colToVec(const _T<_Tin>& _in)
	{
		std::vector<_Tin> t_(_in.size());
		auto it = 0;
		for (const auto& _inner : _in)
			t_[it++] = _inner;
		return t_;
	}

	template<template <class _Tin> class _T, class _Tin>
	inline void colToVec(const _T<_Tin>& _in, std::vector<_Tin>& _out)
	{
		_out = std::vector<_Tin>(_in.size());
		auto it = 0;
		for (const auto& _inner : _in)
			_out[it++] = _inner;
	}

	// ###################### I N I T I A L I Z A T I O N ######################

	/*
	* @brief Creates a vector from a to (a + N - 1)
	* @param N size of the vector
	* @param a starting point
	*/
	template<typename _T1, typename = typename std::enable_if<std::is_arithmetic<_T1>::value, _T1>::type>
	inline std::vector<_T1> vecAtoB(_T1 N, _T1 a = 0)
	{
		std::vector<_T1> idxs(N);
		std::iota(idxs.begin(), idxs.end(), a);
		return idxs;
	}

	// ######################### M A T H E M A T I C S #########################

	// ---------- ADD ----------

	/*
	* @brief Add two vectors together. Method is inplace.
	* @param _res first vector
	* @param _toAdd second vector
	*/
	template <class _T, class _T2, class _A, class _A2>
	inline void add(std::vector<_T, _A>& _res, const std::vector<_T, _A2>& _toAdd)
	{
		if (_res.size() != _toAdd.size())
			throw std::runtime_error("Size of vectors mismatch...");
		for (auto i = 0; i < _res.size(); ++i)
			_res[i] += _toAdd[i];
	};

	template <class _T, class _T2, class _A, class _A2>
	inline std::vector<_T, _A> add(const std::vector<_T>& _res, const std::vector<_T2, _A2>& _toAdd)
	{
		std::vector<_T, _A> _out;
		if (_res.size() != _toAdd.size())
			throw std::runtime_error("Size of vectors mismatch...");
		_out.resize(_res.size());
		for (auto i = 0; i < _res.size(); ++i)
			_out[i] = _toAdd[i] + _res[i];
		return _out;
	};

	template <class _T, class _T2, class _A, class _A2>
	inline std::vector<_T> operator+(const std::vector<_T>& _res, const std::vector<_T2, _A2>& _toAdd)
	{
		return add(_res, _toAdd);
	}

	// ------- SUBSTRACT -------

	/*
	* @brief Substract two vectors together. Method is inplace.
	* @param _res first vector
	* @param _toAdd second vector
	*/
	template <class _T>
	inline void sub(std::vector<_T>& _res, const std::vector<_T>& _toAdd)
	{
		if (_res.size() != _toAdd.size())
			throw std::runtime_error("Size of vectors mismatch...");
		for (auto i = 0; i < _res.size(); ++i)
			_res[i] -= _toAdd[i];
	};

	template <class _T>
	inline std::vector<_T> sub(const std::vector<_T>& _res, const std::vector<_T>& _toAdd)
	{
		std::vector<_T> _out;
		if (_res.size() != _toAdd.size())
			throw std::runtime_error("Size of vectors mismatch...");
		_out.resize(_res.size());
		for (auto i = 0; i < _res.size(); ++i)
			_out[i] = _res[i] - _toAdd[i];
		return _out;
	};

	template <class _T, class _T2, class _A, class _A2>
	inline std::vector<_T> operator-(const std::vector<_T>& _res, const std::vector<_T2, _A2>& _toAdd)
	{
		return sub(_res, _toAdd);
	}

	// ------- MULTIPLY --------

	/*
	* @brief Multiply two vectors by some value. Method is inplace.
	* @param _res first vector
	* @param _const value
	*/
	template <class _T>
	inline void mul(std::vector<_T>& _res, _T _const)
	{
		for (auto i = 0; i < _res.size(); ++i)
			_res[i] *= _const;
	};

	template <class _T>
	inline std::vector<_T> mul(const std::vector<_T>& _res, _T _const)
	{
		std::vector<_T> _out;
		_out.resize(_res.size());
		for (auto i = 0; i < _res.size(); ++i)
			_out[i] = _const * _res[i];
		return _out;
	};

	template <class _T, class _T2, class _A, class _A2>
	inline std::vector<_T> operator*(const std::vector<_T>& _res, const std::vector<_T2, _A2>& _toAdd)
	{
		return mul(_res, _toAdd);
	}

	// ############################# S O R T I N G #############################

	template <class VectorIterator, typename Compare>
	inline void bubbleSort(VectorIterator _b, VectorIterator _e, Compare compare, uint* _comparisons = nullptr)
	{
		auto _distance	= std::distance(_b, _e);
		// return already
		if (_distance <= 0)
			return;

		// access each element
		for (auto i = 0; i < _distance; i++)
		{
			// compare elements
			for (auto j = 0; j < _distance - i; j++)
			{
				if (compare(*(_b + i), (*(_b + j))))
				{
					if(_comparisons)
						*_comparisons += 1;
					// swap 'em
					std::swap(*(_b + i), (*(_b + j)));
				}
			}
		}
	};
};

// #############################################################################

/*
* @brief Namespace that provides methods for manipulating with states.
* States are represented with Armadillo columns of doubles.
*/
namespace States
{
	// ###################### T R A N S F O R M A T I O N ######################

	/*
	* @brief Transform vector of indices to full state in Fock real space basis.
	* @param _Ns number of lattice sites
	* @param _state single particle orbital indices
	* @returns an Armadillo vector in the Fock basis
	*/
	template<typename _T>
	inline arma::Col<double> transformIdxToState(uint _Ns, const _T& _state)
	{
		arma::Col<double> _out(_Ns, arma::fill::zeros);
		for (auto& i : _state)
			_out(i) = 1;
		return _out;
	}

	template <typename _T>
	inline arma::Row<double> transformIdxToStateR(uint _Ns, const _T& _state)
	{
		arma::Row<double> _out(_Ns, arma::fill::zeros);
		for (auto& i : _state)
			_out(i) = 1;
		return _out;
	}

	//template<typename _T>
	//inline void transformIdxToState(uint _Ns, const _T& _state, arma::Col<double>& _out)
	//{
	//	_out = arma::Col<double>(_Ns, arma::fill::zeros);
	//	for (auto& i : _state)
	//		_out(i) = 1;
	//}

	// --------------------------------------------------------------------------

	/*
	* @brief Transform vector of indices to full state in Fock real space basis.
	* @param _Ns number of lattice sites
	* @param _state single particle orbital indices
	* @returns an Armadillo vector in the Fock basis
	*/
	template<typename _T>
	inline sul::dynamic_bitset<> transformIdxToBitset(uint _Ns, const _T& _state)
	{
		sul::dynamic_bitset<> _out(_Ns);
		for (auto& i : _state)
			_out[_Ns - i - 1] = true;
		return _out;
	}

	//template<typename _T>
	//inline void transformIdxToBitset(uint _Ns, const _T& _state, sul::dynamic_bitset<>& _out)
	//{
	//	_out = sul::dynamic_bitset<>(_Ns);
	//	for (auto& i : _state)
	//		_out[_Ns - i - 1] = true;
	//	return _out;
	//}

	// --------------------------------------------------------------------------

};

#endif 