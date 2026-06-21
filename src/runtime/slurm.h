/******************************************************************************
 *
 *  @file src/runtime/slurm.h
 *  @brief Slurm environment check, remaining time, and overtime queries.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once
#ifndef GENUTILS_SLURM_H
#define GENUTILS_SLURM_H

#include <csignal>
#include <string>

namespace Slurm
{
	/**
	 * @brief Checks if the current environment is a SLURM job.
	 */
	bool is_slurm();

	/**
	 * @brief Get the remaining time for the current SLURM job.
	 * Executes `scontrol show job $SLURM_JOB_ID` and returns remaining seconds, or -1.
	 */
	int get_remaining_time();

	/**
	 * @brief Checks if the remaining time is less than the specified threshold.
	 * Queries scontrol.
	 */
	bool is_overtime(int limit_seconds = 1000);

	/**
	 * @brief Checks if the remaining time is less than the limit, using a local stopwatch fallback or SLURM.
	 * If start_time_seconds >= 0 and job_time_seconds >= 0, uses local clock elapsed time.
	 * Otherwise falls back to SLURM.
	 */
	bool is_overtime(int limit_seconds, double start_time_seconds, double job_time_seconds, bool verbose = false);

	/**
	 * @brief Checks if the remaining time is less than the limit, caching the SLURM scontrol query for a given interval.
	 */
	bool is_overtime_cached(int limit_seconds, int poll_interval_seconds = 30);

	/** Install async-signal-safe stop notification for scheduler pre-timeout signals. */
	void install_checkpoint_signal_handlers();
	/** True after SIGUSR1, SIGTERM, or SIGINT. Safe to poll between work units. */
	bool checkpoint_requested() noexcept;
	/** Signal that requested the stop, or zero. */
	int checkpoint_signal() noexcept;
	/** Clear a handled request after the application has saved durable state. */
	void clear_checkpoint_request() noexcept;
	/** Recommended sbatch directive value, e.g. "B:USR1@120". */
	std::string checkpoint_signal_spec(int lead_seconds = 120);
}

#endif // GENUTILS_SLURM_H
