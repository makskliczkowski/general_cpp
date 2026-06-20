/******************************************************************************
 *
 *  @file hdf5_manager.h
 *  @brief HDF5 collection management: save / load / list named matrices.
 *
 *  @project general_cpp
 *
 *  general_cpp already provides single-dataset save/load through
 *  lin_alg.h (saveAlgebraic / loadAlgebraic). This adds the collection
 *  layer of the Python reference (common/hdf5man.py): writing and reading a
 *  whole map<name, matrix> to one file, listing the datasets in a file,
 *  appending new datasets and overwriting individual fields.
 *
 *  Requires the Armadillo + HDF5 backend (GENUTILS_HAS_ARMADILLO and the
 *  HDF5 feature). Without them the header is empty.
 *
 ******************************************************************************/

#pragma once
#ifndef GENUTILS_HDF5_MANAGER_H
#define GENUTILS_HDF5_MANAGER_H

#include "../algebra/backend_config.h"

#if defined(GENUTILS_HAS_ARMADILLO) && defined(ARMA_USE_HDF5)

#include <hdf5.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace HDF5
{

/**
 * Management of named matrix datasets inside a single HDF5 file. All methods
 * are static; the type parameter is the matrix element type.
 */
template<typename _T = double>
class HDF5Manager
{
public:
	using matrix_type = arma::Mat<_T>;
	using collection  = std::map<std::string, matrix_type>;

	// ############################ E X I S T E N C E ############################

	/**
	 * @brief List every top-level dataset name in an HDF5 file.
	 * @param _path file path
	 * @returns dataset names (empty if the file cannot be opened)
	 */
	[[nodiscard]] static std::vector<std::string> keys(const std::string& _path)
	{
		std::vector<std::string> _names;
		const hid_t _file = H5Fopen(_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		if (_file < 0)
			return _names;

		H5G_info_t _info;
		if (H5Gget_info(_file, &_info) >= 0)
		{
			_names.reserve(static_cast<std::size_t>(_info.nlinks));
			for (hsize_t _i = 0; _i < _info.nlinks; ++_i)
			{
				const ssize_t _len = H5Lget_name_by_idx(
					_file, ".", H5_INDEX_NAME, H5_ITER_INC, _i, nullptr, 0, H5P_DEFAULT);
				if (_len <= 0)
					continue;
				std::string _name(static_cast<std::size_t>(_len), '\0');
				H5Lget_name_by_idx(_file, ".", H5_INDEX_NAME, H5_ITER_INC, _i,
					_name.data(), static_cast<std::size_t>(_len) + 1, H5P_DEFAULT);
				_names.push_back(std::move(_name));
			}
		}
		H5Fclose(_file);
		return _names;
	}

	/**
	 * @brief Whether a dataset exists in a file.
	 */
	[[nodiscard]] static bool contains(const std::string& _path, const std::string& _key)
	{
		const auto _all = keys(_path);
		for (const auto& _name : _all)
			if (_name == _key)
				return true;
		return false;
	}

	// ############################### S A V E ###################################

	/**
	 * @brief Write a single named matrix, optionally appending to an existing
	 *        file (otherwise the file is replaced).
	 * @returns true on success
	 */
	static bool save(const std::string& _path, const std::string& _key,
		const matrix_type& _data, bool _append = false)
	{
		// hdf5_opts::append and ::none are distinct types, so the two paths
		// cannot share a ternary-built spec.
		if (_append)
			return _data.save(arma::hdf5_name(_path, _key, arma::hdf5_opts::append));
		return _data.save(arma::hdf5_name(_path, _key));
	}

	/**
	 * @brief Write a whole collection to one file. The first dataset replaces
	 *        the file; the rest are appended, so the file holds exactly the
	 *        collection.
	 * @returns true if every dataset was written
	 */
	static bool save(const std::string& _path, const collection& _data)
	{
		bool _ok    = true;
		bool _first = true;
		for (const auto& [_key, _matrix] : _data)
		{
			_ok = save(_path, _key, _matrix, !_first) && _ok;
			_first = false;
		}
		return _ok;
	}

	/**
	 * @brief Overwrite one dataset (or add it), leaving the rest of the file
	 *        intact. Uses hdf5_opts::replace, which - unlike plain append -
	 *        succeeds whether or not the dataset already exists.
	 * @returns true on success
	 */
	static bool update(const std::string& _path, const std::string& _key,
		const matrix_type& _data)
	{
		return _data.save(arma::hdf5_name(_path, _key, arma::hdf5_opts::replace));
	}

	// ############################### L O A D ###################################

	/**
	 * @brief Load a single named matrix.
	 * @throws std::runtime_error if the dataset cannot be read
	 */
	[[nodiscard]] static matrix_type load(const std::string& _path, const std::string& _key)
	{
		matrix_type _out;
		if (!_out.load(arma::hdf5_name(_path, _key)))
			throw std::runtime_error("HDF5Manager: failed to load '" + _key + "' from " + _path);
		return _out;
	}

	/**
	 * @brief Load the named datasets (or every dataset when @p _keys is empty)
	 *        into a collection. Named distinctly from the single-matrix load
	 *        to avoid braced-initializer overload ambiguity.
	 */
	[[nodiscard]] static collection load_collection(const std::string& _path,
		std::vector<std::string> _keys = {})
	{
		if (_keys.empty())
			_keys = keys(_path);
		collection _out;
		for (const auto& _key : _keys)
			_out.emplace(_key, load(_path, _key));
		return _out;
	}
};

} // namespace HDF5

#endif // GENUTILS_HAS_ARMADILLO && ARMA_USE_HDF5

#endif // GENUTILS_HDF5_MANAGER_H
