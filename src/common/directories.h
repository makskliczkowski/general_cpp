/******************************************************************************
 *
 *  @file src/Include/directories.h
 *  @brief Directory creation and separator utilities.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once

#ifndef DIRECTORIES_H
#define DIRECTORIES_H

#include <filesystem>
namespace fs = std::filesystem;

// --- K PATH SEPARATOR : DIRECTORY SEPARATOR ---
inline const std::string kPS = 
#ifdef _WIN32 
    "\\"; 
#else 
    "/"; 
#endif



// ############################################################# DIRECTORIES #############################################################

/*
* @brief Append an os separator to the folder
* @param folder - folder to be appended
* @returns folder appended by the os separator
*/
template <typename _T>
std::string makeDir(const _T& folder)
{
	return STRP(folder, 3) + kPS;
}

/*
* @brief Create a path out of given folders
* @param folder - folder to be appended
* @param all the folders
* @returns path
*/
template <typename _T, typename... _Ty>
std::string makeDir(const _T& folder, const _Ty&... folders)
{
	return makeDir(folder) + makeDir(folders...);
}

/*
* @brief Create a path out of given folders
* @param folder - folder to be appended
* @param all the folders
* @returns path
*/
template <typename... _Ty>
std::string makeDirs(const _Ty&... folders)
{
	return makeDir(folders...);
}

// --------------------------------------------------------------------

/*
* @brief Creates a single directory given a string path
* @param dir the directory
*/
void createDir(const std::string& dir);

/*
* @brief Creates a variadic directory set given a string paths
* @param dir the directory
*/
template <typename... _Ty>
inline void createDirs(const std::string& dir, const _Ty&... dirs) {
	createDir(dir);
	createDirs(dirs...);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Make a path out of given folders and create it 
* @param folder - folder to be appended
* @param all the folders
* @returns path
*/
template <typename... _Ty>
std::string makeDirsC(const _Ty&... folders)
{
	std::string _folder = makeDir(folders...);
	createDir(_folder);
	return _folder;
}

#endif