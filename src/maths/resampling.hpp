/******************************************************************************
 *
 *  @file src/maths/resampling.hpp
 *  @brief Memory-efficient binning and jackknife error estimation.
 *
 *  @project general_cpp
 *  @author Maksymilian Kliczkowski
 *
 *  @details This header provides STL-only statistical resampling utilities.
 *  It is independent of any simulation or physics model. Streaming
 *  accumulators retain bin summaries instead of individual measurements.
 *
 ******************************************************************************/

#pragma once

#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <functional>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// --------------------------------------------------------------
// Statistical Resampling
// --------------------------------------------------------------

namespace genutils::statistics
{

	template<typename T>
	concept SampleValue = std::default_initializable<T> && std::copy_constructible<T> &&
		requires(T lhs, T rhs, double scale)
		{
			{ lhs + rhs } -> std::convertible_to<T>;
			{ lhs - rhs } -> std::convertible_to<T>;
			{ lhs / scale } -> std::convertible_to<T>;
		};

	/**
	 * @brief A central estimate with its standard error and sample counts.
	 * @tparam T Estimated value type.
	 */
	template<typename T>
	struct Estimate
	{
		T value {};
		double standard_error = 0.0;
		std::size_t samples    = 0;
		std::size_t bins       = 0;
	};

	/**
	 * @brief A correlated numerator/denominator estimate.
	 * @tparam T Ratio value type.
	 */
	template<typename T>
	struct RatioEstimate : Estimate<T>
	{
		double denominator_magnitude = 0.0;
		bool reliable                = false;
	};

	namespace detail
	{

		template<typename T>
		[[nodiscard]] constexpr double squared_magnitude(const T& value) noexcept
		{
			if constexpr(requires { std::norm(value); })
				return static_cast<double>(std::norm(value));
			else
				return static_cast<double>(value * value);
		}

		template<typename T>
		[[nodiscard]] double magnitude(const T& value) noexcept
		{
			using std::abs;
			return static_cast<double>(abs(value));
		}

		template<typename T>
		[[nodiscard]] double standard_error(const std::vector<T>& values, const T& center)
		{
			if(values.size() < 2)
				return 0.0;

			double sum = 0.0;
			for(const auto& value : values)
				sum += squared_magnitude(value - center);
			return std::sqrt(sum / static_cast<double>(values.size() * (values.size() - 1)));
		}

	} // namespace detail

	/**
	 * @brief Streaming accumulator for ordinary binned means.
	 *
	 * The central value includes every sample. The uncertainty is the standard
	 * error of completed, equal-sized bin means. A final partial bin is retained
	 * in the central value but excluded from the uncertainty calculation.
	 *
	 * @tparam T Sample value type.
	 */
	template<SampleValue T>
	class BinnedAccumulator
	{
	public:
		/**
		 * @brief Construct an accumulator.
		 * @param bin_size Number of measurements in each completed bin.
		 */
		explicit BinnedAccumulator(std::size_t bin_size)
			: bin_size_(bin_size)
		{
			if(bin_size_ == 0)
				throw std::invalid_argument("BinnedAccumulator requires a non-zero bin size.");
		}

		/**
		 * @brief Add one measurement without allocating.
		 * @param sample Measurement value.
		 */
		void add(const T& sample)
		{
			total_ += sample;
			partial_ += sample;
			++samples_;
			++partial_count_;

			if(partial_count_ == bin_size_)
			{
				completed_means_.push_back(partial_ / static_cast<double>(bin_size_));
				partial_       = T {};
				partial_count_ = 0;
			}
		}

		/**
		 * @brief Add all values in a range.
		 * @tparam Range Input range type.
		 * @param samples Measurement range.
		 */
		template<std::ranges::input_range Range>
		requires std::convertible_to<std::ranges::range_reference_t<Range>, T>
		void add(const Range& samples)
		{
			for(const auto& sample : samples)
				add(static_cast<T>(sample));
		}

		/**
		 * @brief Reserve completed-bin storage before a hot loop.
		 * @param count Expected number of completed bins.
		 */
		void reserve_bins(std::size_t count)
		{
			completed_means_.reserve(count);
		}

		/**
		 * @brief Calculate the current mean and binned standard error.
		 * @returns Estimate<T> Current estimate.
		 */
		[[nodiscard]] Estimate<T> estimate() const
		{
			if(samples_ == 0)
				return {};

			const auto value = total_ / static_cast<double>(samples_);
			T bin_center {};
			for(const auto& bin : completed_means_)
				bin_center += bin;
			if(!completed_means_.empty())
				bin_center = bin_center / static_cast<double>(completed_means_.size());

			return {
				.value          = value,
				.standard_error = detail::standard_error(completed_means_, bin_center),
				.samples        = samples_,
				.bins           = completed_means_.size() + (partial_count_ != 0 ? 1U : 0U)
			};
		}

		[[nodiscard]] std::size_t bin_size() const noexcept { return bin_size_; }
		[[nodiscard]] std::size_t samples() const noexcept { return samples_; }
		[[nodiscard]] std::size_t completed_bins() const noexcept { return completed_means_.size(); }
		[[nodiscard]] std::size_t partial_bin_size() const noexcept { return partial_count_; }
		[[nodiscard]] const std::vector<T>& completed_bin_means() const noexcept { return completed_means_; }

	private:
		std::size_t bin_size_ = 1;
		std::size_t samples_ = 0;
		std::size_t partial_count_ = 0;
		T total_ {};
		T partial_ {};
		std::vector<T> completed_means_ {};
	};

	/**
	 * @brief Calculate a binned mean from a range.
	 * @tparam Range Input range type.
	 * @param samples Measurement range.
	 * @param bin_size Number of samples per completed bin.
	 * @returns Estimate of the mean and its binned standard error.
	 */
	template<std::ranges::input_range Range>
	requires SampleValue<std::ranges::range_value_t<Range>>
	[[nodiscard]] auto binned_mean(const Range& samples, std::size_t bin_size)
	{
		using T = std::ranges::range_value_t<Range>;
		BinnedAccumulator<T> accumulator(bin_size);
		accumulator.add(samples);
		return accumulator.estimate();
	}

	/**
	 * @brief Calculate the delete-one-sample jackknife error of a mean.
	 * @tparam Range Input sample range.
	 * @param samples Input samples.
	 * @returns Mean and its jackknife standard error.
	 */
	template<std::ranges::input_range Range>
	requires SampleValue<std::ranges::range_value_t<Range>>
	[[nodiscard]] auto jackknife_mean(const Range& samples)
	{
		const auto result = binned_mean(samples, 1);
		if(result.samples == 0)
			throw std::invalid_argument("jackknife_mean requires at least one sample.");
		return result;
	}

	/**
	 * @brief Generic delete-one-sample jackknife estimate.
	 *
	 * This convenience routine favors a simple interface over memory efficiency:
	 * it materializes each reduced sample set. Use a specialized streaming
	 * accumulator for large production data.
	 *
	 * @tparam Range Forward sample range.
	 * @tparam Estimator Callable accepting const std::vector<sample_type>&.
	 * @param samples Input samples.
	 * @param estimator Statistic evaluated on each replica.
	 * @returns Estimate of the full-sample statistic and jackknife error.
	 */
	template<std::ranges::forward_range Range, typename Estimator>
	[[nodiscard]] auto jackknife(const Range& samples, Estimator estimator)
	{
		using Sample = std::ranges::range_value_t<Range>;
		using Result = std::remove_cvref_t<std::invoke_result_t<Estimator, const std::vector<Sample>&>>;

		std::vector<Sample> values(std::ranges::begin(samples), std::ranges::end(samples));
		if(values.empty())
			throw std::invalid_argument("jackknife requires at least one sample.");

		const auto value = static_cast<Result>(estimator(values));
		if(values.size() == 1)
			return Estimate<Result> { value, 0.0, 1, 1 };

		std::vector<Result> replicas;
		replicas.reserve(values.size());
		std::vector<Sample> reduced;
		reduced.reserve(values.size() - 1);
		for(std::size_t omitted = 0; omitted < values.size(); ++omitted)
		{
			reduced.clear();
			for(std::size_t index = 0; index < values.size(); ++index)
				if(index != omitted)
					reduced.push_back(values[index]);
			replicas.push_back(static_cast<Result>(estimator(reduced)));
		}

		Result replica_center {};
		for(const auto& replica : replicas)
			replica_center += replica;
		replica_center = replica_center / static_cast<double>(replicas.size());

		double variance = 0.0;
		for(const auto& replica : replicas)
			variance += detail::squared_magnitude(replica - replica_center);
		variance *= static_cast<double>(replicas.size() - 1) / static_cast<double>(replicas.size());

		return Estimate<Result> { value, std::sqrt(variance), values.size(), values.size() };
	}

	/**
	 * @brief Streaming correlated ratio estimator with binned jackknife errors.
	 * @tparam Numerator Numerator sample type.
	 * @tparam Denominator Denominator sample type.
	 */
	template<SampleValue Numerator, SampleValue Denominator>
	class RatioBinnedAccumulator
	{
		using Ratio = std::remove_cvref_t<decltype(std::declval<Numerator>() / std::declval<Denominator>())>;

		struct Bin
		{
			Numerator numerator {};
			Denominator denominator {};
		};

	public:
		explicit RatioBinnedAccumulator(
		std::size_t bin_size,
		double denominator_tolerance = 64.0 * std::numeric_limits<double>::epsilon())
			: bin_size_(bin_size), denominator_tolerance_(denominator_tolerance)
		{
			if(bin_size_ == 0)
				throw std::invalid_argument("RatioBinnedAccumulator requires a non-zero bin size.");
			if(denominator_tolerance_ < 0.0)
				throw std::invalid_argument("RatioBinnedAccumulator requires a non-negative tolerance.");
		}

		/**
		 * @brief Add one correlated numerator/denominator measurement pair.
		 */
		void add(const Numerator& numerator, const Denominator& denominator)
		{
			total_numerator_ += numerator;
			total_denominator_ += denominator;
			partial_numerator_ += numerator;
			partial_denominator_ += denominator;
			++samples_;
			++partial_count_;

			if(partial_count_ == bin_size_)
			{
				completed_bins_.push_back({ partial_numerator_, partial_denominator_ });
				partial_numerator_   = Numerator {};
				partial_denominator_ = Denominator {};
				partial_count_       = 0;
			}
		}

		/**
		 * @brief Reserve completed-bin storage before a hot loop.
		 * @param count Expected number of completed bins.
		 */
		void reserve_bins(std::size_t count)
		{
			completed_bins_.reserve(count);
		}

		/**
		 * @brief Calculate the ratio and correlated delete-one-bin error.
		 */
		[[nodiscard]] RatioEstimate<Ratio> estimate() const
		{
			RatioEstimate<Ratio> result;
			result.samples               = samples_;
			result.bins                  = completed_bins_.size() + (partial_count_ != 0 ? 1U : 0U);
			result.denominator_magnitude = detail::magnitude(total_denominator_);

			const auto scale = std::max(1.0, static_cast<double>(samples_));
			if(samples_ == 0 || result.denominator_magnitude <= denominator_tolerance_ * scale)
				return result;

			result.value    = total_numerator_ / total_denominator_;
			result.reliable = true;
			if(completed_bins_.size() < 2)
				return result;

			Numerator completed_numerator {};
			Denominator completed_denominator {};
			for(const auto& bin : completed_bins_)
			{
				completed_numerator += bin.numerator;
				completed_denominator += bin.denominator;
			}

			std::vector<Ratio> replicas;
			replicas.reserve(completed_bins_.size());
			for(const auto& bin : completed_bins_)
			{
				const auto denominator = completed_denominator - bin.denominator;
				const auto replica_scale = std::max(1.0, static_cast<double>((completed_bins_.size() - 1) * bin_size_));
				if(detail::magnitude(denominator) <= denominator_tolerance_ * replica_scale)
				{
					result.reliable = false;
					result.standard_error = std::numeric_limits<double>::infinity();
					return result;
				}
				replicas.push_back((completed_numerator - bin.numerator) / denominator);
			}

			Ratio center {};
			for(const auto& replica : replicas)
				center += replica;
			center = center / static_cast<double>(replicas.size());

			double variance = 0.0;
			for(const auto& replica : replicas)
				variance += detail::squared_magnitude(replica - center);
			variance *= static_cast<double>(replicas.size() - 1) / static_cast<double>(replicas.size());
			result.standard_error = std::sqrt(variance);
			return result;
		}

		[[nodiscard]] std::size_t bin_size() const noexcept { return bin_size_; }
		[[nodiscard]] std::size_t samples() const noexcept { return samples_; }
		[[nodiscard]] std::size_t completed_bins() const noexcept { return completed_bins_.size(); }
		[[nodiscard]] std::size_t partial_bin_size() const noexcept { return partial_count_; }

	private:
		std::size_t bin_size_ = 1;
		double denominator_tolerance_ = 0.0;
		std::size_t samples_ = 0;
		std::size_t partial_count_ = 0;
		Numerator total_numerator_ {};
		Denominator total_denominator_ {};
		Numerator partial_numerator_ {};
		Denominator partial_denominator_ {};
		std::vector<Bin> completed_bins_ {};
	};

	/**
	 * @brief Calculate a correlated binned jackknife ratio from two ranges.
	 * @tparam NumeratorRange Numerator sample range.
	 * @tparam DenominatorRange Denominator sample range.
	 * @param numerators Correlated numerator samples.
	 * @param denominators Correlated denominator samples.
	 * @param bin_size Number of samples per completed bin.
	 * @param denominator_tolerance Relative zero-denominator tolerance.
	 * @returns Ratio estimate and correlated jackknife standard error.
	 */
	template<std::ranges::input_range NumeratorRange, std::ranges::input_range DenominatorRange>
	requires SampleValue<std::ranges::range_value_t<NumeratorRange>> &&
		SampleValue<std::ranges::range_value_t<DenominatorRange>>
	[[nodiscard]] auto jackknife_ratio(
		const NumeratorRange& numerators,
		const DenominatorRange& denominators,
		std::size_t bin_size = 1,
		double denominator_tolerance = 64.0 * std::numeric_limits<double>::epsilon())
	{
		using Numerator   = std::ranges::range_value_t<NumeratorRange>;
		using Denominator = std::ranges::range_value_t<DenominatorRange>;
		RatioBinnedAccumulator<Numerator, Denominator> accumulator(bin_size, denominator_tolerance);

		auto numerator   = std::ranges::begin(numerators);
		const auto numerator_end = std::ranges::end(numerators);
		auto denominator = std::ranges::begin(denominators);
		const auto denominator_end = std::ranges::end(denominators);
		while(numerator != numerator_end && denominator != denominator_end)
		{
			accumulator.add(*numerator, *denominator);
			++numerator;
			++denominator;
		}
		if(numerator != numerator_end || denominator != denominator_end)
			throw std::invalid_argument("jackknife_ratio requires ranges of equal length.");
		return accumulator.estimate();
	}

} // namespace genutils::statistics

// --------------------------------------------------------------------------
//! EOF
// --------------------------------------------------------------------------
