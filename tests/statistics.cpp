#include "maths/resampling.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

	void require(bool condition, const char* message)
	{
		if(!condition)
			throw std::runtime_error(message);
	}

	void require_near(double actual, double expected, double tolerance, const char* message)
	{
		if(std::abs(actual - expected) > tolerance)
			throw std::runtime_error(message);
	}

} // namespace

int main()
{
	try
	{
		using genutils::statistics::BinnedAccumulator;
		using genutils::statistics::RatioBinnedAccumulator;

		const std::vector<double> samples { 1.0, 3.0, 5.0, 7.0 };
		const auto binned = genutils::statistics::binned_mean(samples, 2);
		require_near(binned.value, 4.0, 1.0e-14, "binned central mean");
		require_near(binned.standard_error, 2.0, 1.0e-14, "binned standard error");
		require(binned.samples == 4 && binned.bins == 2, "binned counts");

		BinnedAccumulator<double> partial(2);
		partial.add(std::vector<double> { 1.0, 3.0, 5.0 });
		const auto partial_result = partial.estimate();
		require_near(partial_result.value, 3.0, 1.0e-14, "partial bin retained in mean");
		require(partial_result.bins == 2 && partial.partial_bin_size() == 1, "partial bin counts");

		const auto jackknife = genutils::statistics::jackknife(
			samples,
			[](const std::vector<double>& values)
			{
				double sum = 0.0;
				for(const auto value : values)
					sum += value;
				return sum / static_cast<double>(values.size());
			});
		require_near(jackknife.value, 4.0, 1.0e-14, "jackknife central mean");
		require_near(jackknife.standard_error, std::sqrt(5.0 / 3.0), 1.0e-14, "jackknife mean error");
		const auto direct_mean = genutils::statistics::jackknife_mean(samples);
		require_near(direct_mean.standard_error, jackknife.standard_error, 1.0e-14, "direct jackknife mean");

		using Complex = std::complex<double>;
		RatioBinnedAccumulator<Complex, Complex> ratio(1);
		for(const auto phase : std::vector<double> { 1.0, -1.0, 1.0, 1.0 })
			ratio.add(Complex { 2.0 * phase, 0.0 }, Complex { phase, 0.0 });
		const auto ratio_result = ratio.estimate();
		require(ratio_result.reliable, "correlated ratio reliability");
		require_near(ratio_result.value.real(), 2.0, 1.0e-14, "correlated ratio value");
		require_near(ratio_result.standard_error, 0.0, 1.0e-14, "correlated ratio error");
		const auto direct_ratio = genutils::statistics::jackknife_ratio(
			std::vector<double> { 2.0, -2.0, 2.0, 2.0 },
			std::vector<double> { 1.0, -1.0, 1.0, 1.0 });
		require_near(direct_ratio.value, 2.0, 1.0e-14, "direct correlated ratio");

		bool rejected_mismatch = false;
		try
		{
			static_cast<void>(genutils::statistics::jackknife_ratio(
				std::vector<double> { 1.0 }, std::vector<double> { 1.0, 2.0 }));
		}
		catch(const std::invalid_argument&)
		{
			rejected_mismatch = true;
		}
		require(rejected_mismatch, "ratio range length validation");

		RatioBinnedAccumulator<double, double> cancellation(1);
		cancellation.add(1.0, 1.0);
		cancellation.add(-1.0, -1.0);
		const auto cancelled = cancellation.estimate();
		require(!cancelled.reliable, "zero denominator rejected");

		RatioBinnedAccumulator<double, double> partial_ratio(2);
		partial_ratio.add(2.0, 1.0);
		partial_ratio.add(2.0, 1.0);
		partial_ratio.add(8.0, 2.0);
		const auto partial_ratio_result = partial_ratio.estimate();
		require_near(partial_ratio_result.value, 3.0, 1.0e-14, "partial ratio retained");
		require(partial_ratio_result.samples == 3 && partial_ratio_result.bins == 2, "partial ratio counts");

		std::cout << "GenUtils statistics tests passed\n";
		return 0;
	}
	catch(const std::exception& error)
	{
		std::cerr << "GenUtils statistics test failure: " << error.what() << '\n';
		return 1;
	}
}
