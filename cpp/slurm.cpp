/******************************************************************************
 *
 *  @file cpp/slurm.cpp
 *  @brief Implementations for Slurm runtime utilities.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#include "../src/runtime/slurm.h"
#include "../src/common/flog.h"
#include <cstdlib>
#include <cstdio>
#include <array>
#include <string>
#include <stdexcept>
#include <chrono>
#include <csignal>
#include <string>

namespace {
volatile std::sig_atomic_t checkpoint_signal_value = 0;
extern "C" void record_checkpoint_signal(int signal) { checkpoint_signal_value = signal; }
}

namespace Slurm
{
	void install_checkpoint_signal_handlers()
	{
		std::signal(SIGINT, record_checkpoint_signal);
		std::signal(SIGTERM, record_checkpoint_signal);
#if defined(SIGUSR1)
		std::signal(SIGUSR1, record_checkpoint_signal);
#endif
	}

	bool checkpoint_requested() noexcept { return checkpoint_signal_value != 0; }
	int checkpoint_signal() noexcept { return checkpoint_signal_value; }
	void clear_checkpoint_request() noexcept { checkpoint_signal_value = 0; }

	std::string checkpoint_signal_spec(int lead_seconds)
	{
		if (lead_seconds <= 0) throw std::invalid_argument("checkpoint signal lead time must be positive");
		return "B:USR1@" + std::to_string(lead_seconds);
	}

	bool is_slurm()
	{
		return std::getenv("SLURM_JOB_ID") != nullptr;
	}

	int get_remaining_time()
	{
		if (!is_slurm())
			return -1;

		std::string command     = "scontrol show job $SLURM_JOB_ID";
		std::string output;
		std::array<char, 128> buffer;

		FILE* pipe              = popen(command.c_str(), "r");
		if (!pipe) 
			return -1;

		while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
			output              += buffer.data();
		pclose(pipe);
		
		std::string time_limit_key      = "TimeLimit=";
		std::string run_time_key        = "RunTime=";
		auto time_limit_pos             = output.find(time_limit_key);
		auto run_time_pos               = output.find(run_time_key);        
		
		if (time_limit_pos != std::string::npos && run_time_pos != std::string::npos) 
		{
			auto parse_time = [](const std::string& time_str) -> uint64_t {
				uint64_t days = 0, hours = 0, minutes = 0, seconds = 0;
				if (sscanf(time_str.c_str(), "%lld-%lld:%lld:%lld", &days, &hours, &minutes, &seconds) == 4)
					return days * 86400 + hours * 3600 + minutes * 60 + seconds;
				else if (sscanf(time_str.c_str(), "%lld-%lld:%lld", &days, &hours, &minutes) == 3)
					return days * 86400 + hours * 3600 + minutes * 60;
				else if (sscanf(time_str.c_str(), "%lld:%lld:%lld", &hours, &minutes, &seconds) == 3)
					return hours * 3600 + minutes * 60 + seconds;
				return 0;
			};

			// Extract time limit and run time safely (up to next space or newline)
			auto extract_val = [](const std::string& out_str, std::size_t start_pos, const std::string& key) -> std::string {
				auto val_start = start_pos + key.size();
				auto val_end = out_str.find_first_of(" \n\r", val_start);
				if (val_end == std::string::npos)
					return out_str.substr(val_start);
				return out_str.substr(val_start, val_end - val_start);
			};

			std::string time_limit_str = extract_val(output, time_limit_pos, time_limit_key);
			std::string run_time_str = extract_val(output, run_time_pos, run_time_key);

			uint64_t total_time_limit_seconds   = parse_time(time_limit_str);
			uint64_t total_run_seconds          = parse_time(run_time_str);

			if (total_time_limit_seconds > 0 && total_run_seconds >= 0)
			{
				int remaining_time_seconds  = total_time_limit_seconds - total_run_seconds;
				return remaining_time_seconds > 0 ? remaining_time_seconds : 0;
			}
		}
		return -1;
	}

	bool is_overtime(int limit_seconds)
	{
		if (!is_slurm())
			return false;

		int remaining_time = get_remaining_time();
		if (remaining_time == -1)
			return false;
		
		return remaining_time < limit_seconds;
	}

	bool is_overtime(int limit_seconds, double start_time_seconds, double job_time_seconds, bool verbose)
	{
		if (!is_slurm() && (start_time_seconds < 0 || job_time_seconds < 0))
			return false;

		if (start_time_seconds >= 0 && job_time_seconds >= 0)
		{
			// Local stopwatch path
			auto now = std::chrono::steady_clock::now();
			auto duration = now.time_since_epoch();
			double current_time_seconds = std::chrono::duration<double>(duration).count();
			double elapsed = current_time_seconds - start_time_seconds;
			double remaining = job_time_seconds - elapsed;
			if (verbose)
			{
				LOGINFO("Elapsed=" + std::to_string(elapsed) + "s, remaining=" + std::to_string(remaining) + "s, limit=" + std::to_string(limit_seconds) + "s", LOG_TYPES::INFO, 3);
			}
			if (remaining < limit_seconds)
			{
				LOGINFO("Remaining time " + std::to_string(remaining) + "s is below limit " + std::to_string(limit_seconds) + "s", LOG_TYPES::INFO, 3);
				return true;
			}
			return false;
		}

		// SLURM path
		int remaining = get_remaining_time();
		if (remaining == -1)
			return false;

		if (verbose)
		{
			LOGINFO("Remaining time in SLURM job: " + std::to_string(remaining) + "s", LOG_TYPES::INFO, 3);
		}
		if (remaining < limit_seconds)
		{
			LOGINFO("Remaining time " + std::to_string(remaining) + "s is below limit " + std::to_string(limit_seconds) + "s", LOG_TYPES::INFO, 3);
			return true;
		}
		return false;
	}

	bool is_overtime_cached(int limit_seconds, int poll_interval_seconds)
	{
		static double last_slurm_check = -1e30;
		static int cached_remaining = -1;

		if (!is_slurm())
			return false;

		auto now = std::chrono::steady_clock::now();
		auto duration = now.time_since_epoch();
		double current_time_seconds = std::chrono::duration<double>(duration).count();

		if (cached_remaining == -1 || (current_time_seconds - last_slurm_check) >= poll_interval_seconds)
		{
			cached_remaining = get_remaining_time();
			last_slurm_check = current_time_seconds;
		}

		if (cached_remaining == -1)
			return false;

		return cached_remaining < limit_seconds;
	}
}
