/******************************************************************************
 *
 *  @file src/common/containers.h
 *  @brief Common container utilities and type aliases for the general_cpp project.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <unordered_map>
#include <functional>
#include <algorithm>
#include <tuple>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <complex>

// -----------------------------------------------------------------------------
#include "dynamic_bitset.hpp"
#include "../algebra/generalized_matrix.h"
#include "../maths/maths.h"

// -----------------------------------------------------------------------------

namespace Containers
{
	/**
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
	
	/**
	 * @brief Unzip a vector of tuples into separate vectors.
	 * @tparam Ts Types of the elements in the tuples.
	 * @param zipped The vector of tuples to unzip.
	 * @param containers The output vectors to store the unzipped elements.
	 */
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

	/**
	 * @brief Sort a vector of tuples based on a specific element index and a custom comparison function.
	 * @tparam _On The index of the tuple element to sort by.
	 * @tparam Ts Types of the elements in the tuples.
	 * @param zipped The vector of tuples to sort.
	 * @param f The custom comparison function to determine the sorting order.
	 * The comparison function should take two arguments of the type corresponding to the tuple element at index
	 * _On and return a boolean indicating whether the first argument should come before the second in the sorted order.
	 */
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
}; //! namespace Containers

// -----------------------------------------------------------------------------
// Type aliases for common container types
// -----------------------------------------------------------------------------

template<typename T>
struct is_vector : std::false_type {};
template<typename T>
struct is_vector<std::vector<T>> : std::true_type {};

// -----------------------------------------------------------------------------

namespace Vectors
{
	/**
	* @brief Convert vector of one type to another. Namely, it's 
	* a conversion betweeb types that can be constructed from each other, like int to double, or string to double.
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

	/**
	 * @brief Convert a vector of strings to a vector of a specified type by parsing the strings.
	 * @tparam _T The target type to convert the strings to. This type must be constructible from a string, such as int, double, etc.
	 * @param _v The vector of strings to convert.
	 * @return A vector of type _T containing the converted values from the input vector of strings.
	 * @throws std::runtime_error If any string in the input vector cannot be converted to the target type _T, an exception is thrown with an error message.
	 * @note This function uses std::transform to apply the conversion to each element in the input vector. The conversion is
	 * performed using a lambda function that calls std::stod (or similar functions for other types)
	 */
	template <typename _T>
	inline v_1d<_T> convert(const v_1d<std::string>& _v)
	{
		v_1d<_T> _out;
		_out.resize(_v.size());
		try {
	 		std::transform(_v.begin(), _v.end(), _out.begin(), [](const std::string& _elem) { return static_cast<_T>(std::stod(_elem)); });
		} catch (std::exception& e) {
			std::cerr << e.what() << std::endl;
			throw std::runtime_error("WTF");
		}
		return _out;
	}

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
		for (std::size_t i = 0; i < _res.size(); ++i)
			_res[i] += _toAdd[i];
	};

	template <class _T, class _T2, class _A, class _A2>
	inline std::vector<_T, _A> add(const std::vector<_T>& _res, const std::vector<_T2, _A2>& _toAdd)
	{
		std::vector<_T, _A> _out;
		if (_res.size() != _toAdd.size())
			throw std::runtime_error("Size of vectors mismatch...");
		_out.resize(_res.size());
		for (std::size_t i = 0; i < _res.size(); ++i)
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
		for (std::size_t i = 0; i < _res.size(); ++i)
			_res[i] -= _toAdd[i];
	};

	template <class _T>
	inline std::vector<_T> sub(const std::vector<_T>& _res, const std::vector<_T>& _toAdd)
	{
		std::vector<_T> _out;
		if (_res.size() != _toAdd.size())
			throw std::runtime_error("Size of vectors mismatch...");
		_out.resize(_res.size());
		for (std::size_t i = 0; i < _res.size(); ++i)
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
	inline void bubbleSort(VectorIterator _b, VectorIterator _e, Compare compare, std::size_t* _comparisons = nullptr)
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

// #############################################################################

namespace Simulation
{
	//────────────────────────────────────────────────────────────────────────────
    // 1) type‑erasure base
    //────────────────────────────────────────────────────────────────────────────
    class IContainer
    {
    public:
        virtual ~IContainer() = default;
    };

    //────────────────────────────────────────────────────────────────────────────
    // 2) wrapper for any concrete container C
    //-----

    template<typename C>
    class ContainerHolder
        : public IContainer
    {
    public:
        C data;

        // forward any ctor args to C’s ctor
        template<typename... Args>
        ContainerHolder(Args&&... args)
            : data(std::forward<Args>(args)...)
        { }
    };

    //────────────────────────────────────────────────────────────────────────────
    // 3) Main manager: add by name or index, fetch by template
    //────────────────────────────────────────────────────────────────────────────

    class DataContainer
    {
		using uptr_t 							= std::unique_ptr<IContainer>;
        using map_t  							= std::unordered_map<std::string, size_t>;
		using v_uptr_t 							= std::vector<uptr_t>;

		// Armadillo types
		using acol_d 							= arma::Col<double>;
		using amat_d 							= arma::Mat<double>;
		using acol_c 							= arma::Col<std::complex<double>>;
		using amat_c 							= arma::Mat<std::complex<double>>;
		template <typename _T> using acol_t 	= arma::Col<_T>;
		template <typename _T> using amat_t 	= arma::Mat<_T>; 

		using v_amat_d 							= std::vector<amat_d>;
		using v_acol_d 							= std::vector<acol_d>;
		using v_amat_c 							= std::vector<amat_c>;
		using v_acol_c 							= std::vector<acol_c>;
		template <typename _T> using v_amat_t 	= std::vector<amat_t<_T>>;
		template <typename _T> using v_acol_t 	= std::vector<acol_t<_T>>;

    private:
        //! preserves insertion order so index‑based get() works
        v_uptr_t container_list_;
        map_t name_to_index_;

    public:

		// ---------------------------------------------------------------------
		
		using iterator 							= map_t::const_iterator;
		iterator begin() 						const { return name_to_index_.begin(); 	};
		iterator end() 							const { return name_to_index_.end(); 	};

		//! how many containers you have
		size_t size() 							const { return container_list_.size(); }

		// ---------------------------------------------------------------------
		//! FIND
		// ---------------------------------------------------------------------

		/**
		* @brief Finds the iterator to the container with the specified name.
		*
		* Searches for a container by its name in the internal map. If the container is found,
		* returns an iterator to its entry. If the container is not found, throws a std::runtime_error.
		*
		* @param name The name of the container to find.
		* @return map_t::iterator Iterator to the found container.
		* @throws std::runtime_error If no container with the specified name exists.
		*/
		map_t::iterator find(const std::string& name)
		{
			auto it = name_to_index_.find(name);
			if (it == name_to_index_.end())
				throw std::runtime_error("No container named '" + name + "'");
			return it;
		}

		// ---------------------------------------------------------------------
		//! EREASE
		// ---------------------------------------------------------------------

        // erase by name
        void erase(const std::string& name)
        {
			auto it 	= this->find(name);
            size_t idx 	= it->second;

            // remove pointer
            container_list_.erase(container_list_.begin() + idx);

            // erase map entry
            name_to_index_.erase(it);

            // decrement all indices > idx
            for (auto &kv : name_to_index_)
                if (kv.second > idx)  --kv.second;
        }

        // erase by insertion index
        void erase(size_t idx)
        {
            if (idx >= container_list_.size())
                throw std::out_of_range("Index out of range");

            // find the name
            std::string name;
            for (auto &kv : name_to_index_)
                if (kv.second == idx)
                {
                    name = kv.first;
                    break;
                }
            if (name.empty())
                throw std::runtime_error("No name found for index " + std::to_string(idx));

            erase(name);
        }

		// ---------------------------------------------------------------------
		//! ADD
		// ---------------------------------------------------------------------

        //! add a new container of type C, constructed with Args...
        template<typename C, typename... Args>
        void add(const std::string& name, Args&&... args)
        {
			// check if name is already in use
            if (name_to_index_.count(name))
                throw std::runtime_error("Container '" + name + "' already exists");

            size_t idx                             = container_list_.size();
            name_to_index_[name]                   = idx;
            container_list_.emplace_back(
                std::make_unique<ContainerHolder<C>>(std::forward<Args>(args)...)
            );
        }

		// scalar vector
		template <typename _T = double, typename = typename std::enable_if<std::is_floating_point<_T>::value>::type>
		void add_vec_floating(const std::string& name, size_t _size)
		{
			this->add<std::vector<_T>>(name, _size);
		}

		// integer vector
		template <typename _T = int, typename = typename std::enable_if<std::is_integral<_T>::value>::type>
		void add_vec_integer(const std::string& name, size_t _size)
		{
			this->add<std::vector<_T>>(name, _size);
		}

		// complex column (arma::Col<std::complex<double>>)
		void add_col_complex(const std::string& name, size_t _size, std::complex<double> val = std::complex<double>(0.0, 0.0))
		{
			this->add<arma::Col<std::complex<double>>>(name, _size, arma::fill::value(val));
		}

		// complex matrix (arma::Mat<std::complex<double>>)
		void add_mat_complex(const std::string& name, size_t _rows, size_t _cols, std::complex<double> val = std::complex<double>(0.0, 0.0))
		{
			this->add<arma::Mat<std::complex<double>>>(name, _rows, _cols, arma::fill::value(val));
		}

		// add vector of complex matrices
		void add_vec_mat_complex(const std::string& name, size_t _rows, size_t _cols, size_t _size, std::complex<double> val = std::complex<double>(0.0, 0.0))
		{
			std::vector<arma::Mat<std::complex<double>>> vec(_size, arma::Mat<std::complex<double>>(_rows, _cols, arma::fill::value(val)));
			this->add<std::vector<arma::Mat<std::complex<double>>>>(name, std::move(vec));
		}

		// add vector of complex columns
		void add_vec_col_complex(const std::string& name, size_t _size, size_t _rows, std::complex<double> val = std::complex<double>(0.0, 0.0))
		{
			std::vector<arma::Col<std::complex<double>>> vec(_size, arma::Col<std::complex<double>>(_rows, arma::fill::value(val)));
			this->add<std::vector<arma::Col<std::complex<double>>>>(name, std::move(vec));
		}

		// add double column
		void add_col_double(const std::string& name, size_t _size, double val = 0.0)
		{
			this->add<arma::Col<double>>(name, _size, arma::fill::value(val));
		}

		// add double matrix
		void add_mat_double(const std::string& name, size_t _rows, size_t _cols, double val = 0.0)
		{
			this->add<arma::Mat<double>>(name, _rows, _cols, arma::fill::value(val));
		}

		// add vector of double matrices
		void add_vec_mat_double(const std::string& name, size_t _rows, size_t _cols, size_t _size, double val = 0.0)
		{
			std::vector<arma::Mat<double>> vec(_size, arma::Mat<double>(_rows, _cols, arma::fill::value(val)));
			this->add<std::vector<arma::Mat<double>>>(name, std::move(vec));
		}

		// add vector of double columns
		void add_vec_col_double(const std::string& name, size_t _size, size_t _rows, double val = 0.0)
		{
			std::vector<arma::Col<double>> vec(_size, arma::Col<double>(_rows, arma::fill::value(val)));
			this->add<std::vector<arma::Col<double>>>(name, std::move(vec));
		}

		template<typename _T = double>
		void add_col(const std::string& name, size_t _size, _T val = _T(0))
		{
			this->add<arma::Col<_T>>(name, _size, arma::fill::value(val));
		}

		template<typename _T = double>
		void add_mat(const std::string& name, size_t _rows, size_t _cols, _T val = _T(0))
		{
			this->add<arma::Mat<_T>>(name, _rows, _cols, arma::fill::value(val));
		}

		template<typename _T = double>
		void add_vec_mat(const std::string& name, size_t _rows, size_t _cols, size_t _size, _T val = _T(0))
		{
			std::vector<arma::Mat<_T>> vec(_size, arma::Mat<_T>(_rows, _cols, arma::fill::value(val)));
			this->add<std::vector<arma::Mat<_T>>>(name, std::move(vec));
		}

		// ---------------------------------------------------------------------
		//! GET
		// ---------------------------------------------------------------------

        //! get by name

		/**
		* @brief Retrieves a reference to a container of type C by its name.
		*
		* This function searches for a container with the specified name, checks if it is of the expected type C,
		* and returns a reference to the contained data. If the container is not found or the type does not match,
		* a std::runtime_error is thrown.
		*
		* @tparam C The expected type of the container to retrieve.
		* @param name The name of the container to retrieve.
		* @return C& Reference to the container data of type C.
		* @throws std::runtime_error If the container is not found or if there is a type mismatch.
		*/
        template<typename C>
        C& get(const std::string& name)
        {
            auto it 	= this->find(name);
            auto ptr 	= container_list_[it->second].get();
            auto ph  	= dynamic_cast<ContainerHolder<C>*>(ptr);
            if (!ph)
                throw std::runtime_error("Type mismatch for container '" + name + "'");
            return ph->data;
        }

        //! get by insertion index
        template<typename C>
        C& get(size_t idx)
        {
            if (idx >= container_list_.size())
                throw std::out_of_range("Container index out of range");

            auto ptr = container_list_[idx].get();
            auto ph  = dynamic_cast<ContainerHolder<C>*>(ptr);
            if (!ph)
                throw std::runtime_error("Type mismatch for container index " + std::to_string(idx));
            return ph->data;
        }

		// ---------------------------------------------------------------------


    };

	//────────────────────────────────────────────────────────────────────────────

};


#endif 