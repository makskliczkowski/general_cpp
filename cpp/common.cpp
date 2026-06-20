/******************************************************************************
 *
 *  @file cpp/common.cpp
 *  @brief Implementations for common utilities, operators, and Slurm helpers.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#include "../src/common.h"
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <array>

// Global mutex for thread-safe logging, and global variable for logging indentation level.
std::recursive_mutex logMutex;

// Canonical definition of the logging indentation level declared in common.h.
int LASTLVL = 0;

// -----------------------------------------------------------------------------
// Complex number output formatting
// -----------------------------------------------------------------------------

/**
 * @brief Overload the output stream operator for complex numbers to display in a+bi format with phase information.
 * @param out The output stream to write to.
 * @param v The complex number to format and output.
 * @returns The output stream with the formatted complex number.
 */
std::ostream& operator<< (std::ostream& out, const cpx v)
{
	auto prec 	= 1e-4;
	auto phase 	= std::arg(v) / PI;
	while (phase < 0) 
		phase += 2.0;
	
	std::string absolute 	= "+" + STRP(std::abs(v), 2);
	std::string phase_str 	= "";

	if (EQP(phase, 0.0, prec) || EQP(phase, 2.0, prec)) {
		phase_str = "";
	} else if (EQP(phase, 1.0, prec)) {
		absolute 	= "-" + STRP(std::abs(v), 2);
		phase_str 	= "";
	}
	else {
		phase_str = "*exp(" + STRP(phase, 2) + "*pi*i)";
	}
	out << absolute + phase_str;
	return out;
}

// -----------------------------------------------------------------------------
// Progress bar implementation
// -----------------------------------------------------------------------------

/**
 * @brief Update the progress bar with new progress and recalculate the filled portion.
 * @param newProgress The amount of progress to add to the current progress.
 * This function updates the current progress and recalculates the amount of filler to display in the progress bar based on the new progress.
 */
void pBar::update(double newProgress)
{
	currentProgress += newProgress;
	if (currentProgress <= neededProgress)
		amountOfFiller = (int)((currentProgress / neededProgress) * (double)pBarLength);
}

/**
 * @brief Print the progress bar to the console, showing the current progress and percentage.
 */
void pBar::print()
{
	currUpdateVal	%= pBarUpdater.size();
	std::cout		<< "\r";															        // Bring cursor to start of line
	std::cout		<< firstPartOfpBar;												            // Print out first part of pBar
	for (int a = 0; a < amountOfFiller; a++) {													// Print out current progress
		std::cout	<< pBarFiller;																// By filling the output
	}
	std::cout		<< pBarUpdater[currUpdateVal];
	for (int b = 0; b < pBarLength - amountOfFiller; b++) {										// Print out spaces
		std::cout	<< " ";
	}
	std::cout		<< lastPartOfpBar;												            // Print out last part of progress bar
	std::cout		<< " (" << (int)(100 * (currentProgress / neededProgress)) << "%)";	        // This just prints out the percent
	std::cout		<< std::flush;
	std::cout		<< EL;
	currUpdateVal += 1;
}

/**
 * @brief Print the progress bar with an additional message and timing information.
 * @param message The message to display alongside the progress bar.
 */
void pBar::printWithTime(std::string message)
{
	std::lock_guard<std::mutex> _guard(_mutex);
	{
		LOGINFO("TIME: " + TMS(timer) + message, LOG_TYPES::TRACE, 2);
		this->print();
	}
	this->update(percentage);
}

// -----------------------------------------------------------------------------
//! EOF
// -----------------------------------------------------------------------------

