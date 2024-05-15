#pragma once

#ifndef ALG_H
#include "containers.h"
#endif // !ALG_H

// ######################################################## binning ########################################################

/*
* @brief bin the data to calculate the correlation time approximation
* @param seriesData data to be binned
* @param bins the vector to save the average into
* @param binSize the size of a given single bin
*/
template<typename _T>
inline void binning(const _T& seriesData, _T& bins, uint binSize) {
	if (binSize * bins.size() > seriesData.size()) throw "Cannot create bins of insufficient elements";
	for (int i = 0; i < bins.size(); i++)
		bins[i] = arma::mean(seriesData.subvec(binSize * i, binSize * (i + 1) - 1));
}


// ######################################################## statistical meassures ########################################################

/*
* @brief Approximate the correlation error
* @param bins the bin average of the data
* @returns approximation of a statistical correlation error
*/
template<typename _T>
inline _T correlationError(const arma::Col<_T>& bins) {
	return arma::real(sqrt(arma::var(bins) / bins.size()));
}

///*
//* @brief for an std::vector calculates the standard deviation
//*/
//inline double stddev(const v_1d<double>& v)
//{
//	auto mean = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
//	v_1d<double> diff(v.size());
//	std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
//	auto sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//	return std::sqrt(sq_sum / double(v.size()));
//}

namespace StatisticalMeasures
{
	/*
	* @brief Calculate the gaussianity of the samples. The gaussianity is the ratio of the mean
	* of the square of the samples to the square of the mean of the samples.
	* @param _mean the mean of the samples
	* @param _mean2 the mean of the square of the samples
	* @returns the gaussianity of the samples
	*/
	inline double gaussianity(double _mean, double _mean2)
	{
		return _mean2 / _mean / _mean;
	}
	
	/*
	* @brief Calculate the gaussianity of the samples. The gaussianity is the ratio of the mean 
	* of the square of the samples to the square of the mean of the samples.
	* @param _samples the samples to calculate the gaussianity
	* @returns the gaussianity of the samples
	*/
	template<typename _T>
	inline double gaussianity(const arma::Col<_T>& _samples)
	{
		auto _abs	= arma::abs(_samples);
		auto _mean	= arma::mean(_abs);
		auto _mean2 = arma::mean(arma::square(_abs));
		return gaussianity(algebra::cast<double>(_mean), algebra::cast<double>(_mean2));
	}

	// ##########################################################################################################################################
	
	/*
	* @brief Calculate the cumulant of the samples. The cumulant is the ratio of the mean of the square
	* of the samples to the square of the mean of the samples.
	* @param _mean2 the mean of the square of the samples
	* @param _mean4 the mean of the fourth power of the samples
	* @returns the cumulant of the samples
	*/
	inline double binder_cumulant(double _mean2, double _mean4)
	{
		return _mean4 / 3.0l / _mean2 / _mean2;
	}

	/*
	* @brief Calculate the cumulant of the samples. The cumulant is the ratio of the mean of the square
	* of the samples to the square of the mean of the samples.
	* @param _samples the samples to calculate the cumulant
	* @returns the cumulant of the samples
	*/
	template<typename _T>
	inline double binder_cumulant(const arma::Col<_T>& _samples)
	{	
		auto _square	= arma::square(_samples);
		auto _mean4		= arma::mean(arma::square(_square));
		auto _mean2		= arma::mean(_square);
		return binder_cumulant(algebra::cast<double>(_mean2), algebra::cast<double>(_mean4));
	}

	// ##########################################################################################################################################

	/*
	* @brief Calculate the kurtosis of the samples. The kurtosis is the ratio of the mean of the cube
	* of the samples to the cube of the mean of the samples.
	* @param _var the variance of the samples
	* @param _skew the skewness of the samples
	* @param fisher whether to use the Fisher kurtosis or not
	* @returns the kurtosis of the samples
	*/
	inline double kurtosis(double _var, double _skew, bool fisher = true)
	{
		if (fisher)
			return _skew / _var / _var - 3.0;
		return _skew / _var / _var;
	}

	/*
	* @brief Calculate the kurtosis of the samples. The kurtosis is the ratio of the mean of the cube
	* of the samples to the cube of the mean of the samples.
	* @param _samples the samples to calculate the kurtosis
	* @param fisher whether to use the Fisher kurtosis or not
	* @returns the kurtosis of the samples
	*/
	template<typename _T>
	inline double kurtosis(const arma::Col<_T>& _samples, bool fisher = true)
	{
		auto _square_x_minus_mu = arma::square(_samples - arma::mean(_samples));
		auto _mean2				= arma::mean(_square_x_minus_mu);
		auto _mean4				= arma::mean(arma::square(_square_x_minus_mu));
		return kurtosis(algebra::cast<double>(_mean2), algebra::cast<double>(_mean4), fisher);
	}

	// ##########################################################################################################################################
};


// ##########################################################################################################################################

/*
* @brief A histogram class that stores the bin edges and the bin counts
*/
class Histogram
{
	using u64		= unsigned long long;
protected:
	u64 nBins_		= 1;

	std::vector<long double> binEdges_;
	std::vector<u64> binCounts_;

public:

	virtual ~Histogram() = default;
	Histogram()
	{
		binEdges_	=	std::vector<long double>(this->nBins_, 0);
		binCounts_	=	std::vector<u64>(this->nBins_ + 1, 0);
	}
	Histogram(u64 _N)
		: nBins_(_N)
	{
		binEdges_	=	std::vector<long double>(_N, 0);
		binCounts_	=	std::vector<u64>(_N + 1, 0);
	}
	
	// ######## Getters ########

	const std::vector<long double>& edges()	const { return this->binEdges_;											}
	arma::Col<double> edgesCol()			const { return arma::conv_to<arma::Col<double>>::from(this->binEdges_); }

	// -------------------------

	u64 counts(u64 i)						const { return this->binCounts_[i];										}
	
	// -------------------------

	arma::Col<u64> countsCol()
	{
		arma::Col<u64> _out(this->binCounts_.size(), arma::fill::zeros);
		for (u64 i = 0; i < this->binCounts_.size(); i++)
			_out(i) = this->binCounts_[i];
		return _out;
	}

	// ######## Binning ########

	/*
	* @brief Calculate the interquartile range of the data
	* @param _data the data to calculate the interquartile range
	* @returns the interquartile range
	*/
	static double iqr(const arma::Col<double>& _data)
	{
		u64 _nobs	= _data.n_elem;
		u64 _mid	= _nobs / 2;
		double _q1	= arma::median(_data.subvec(0, _mid));
		double _q3	= arma::median(_data.subvec(_mid, _nobs - 1));
		return _q3 - _q1;
	}

	// -------------------------

	/*
	* @brief Calculate the number of bins using the Freedman-Diaconis rule
	* @param _nobs the number of observations
	* @param _iqr the interquartile range
	* @param _max the maximum value
	* @param _min the minimum value
	* @returns the number of bins
	*/
	static u64 freedman_diaconis_rule(u64 _nobs, double _iqr, double _max, double _min = 0)
	{
		double h = (2.0 * _iqr / std::pow(_nobs, 1.0 / 3.0));
		return (u64)std::ceil((_max - _min) / h);
	}

	// ######## Setters ########

	virtual void reset()					
	{ 
		for (u64 i = 0; i < this->nBins_; i++)
		{
			this->binEdges_[i] = 0;
			this->binCounts_[i] = 0;
		}
		this->binCounts_[this->nBins_] = 0;
	}

	virtual void reset(u64 _N)
	{
		nBins_		=   _N;
		binEdges_	=	std::vector<long double>(_N, 0);
		binCounts_	=	std::vector<u64>(_N + 1, 0);
	}
	
	// -------------------------

	/*
	* @brief Create a uniform distribution of the bins
	* @param _max the maximum value
	* @param _min the minimum value
	* @returns the bin edges
	*/
	void uniform(long double _max, long double _min = 0)
	{
		const long double binWidth = std::fabs(_max - _min) / this->nBins_;

		// create the bin edges
		for (u64 i = 0; i < this->nBins_; i++)
			this->binEdges_[i] = _min + i * binWidth;
	}
	
	/*
	* @brief Create a logarithmic distribution of the bins
	* @param _max the maximum value
	* @param _min the minimum value
	* @param base the base of the logarithm
	*/
	void uniformLog(long double _max, long double _min = 1e-5, int base = 10)
	{
		auto _maxin = _max;
		auto _minin = _min;

		if (base == 10)
		{
			_maxin = std::ceil(std::log10(_max));
			_minin = std::ceil(std::log10(_min));
		}
		else if (base == 2)
		{
			_maxin = std::ceil(std::log2(_max));
			_minin = std::ceil(std::log2(_min));
		}
		else
		{
			_maxin = std::ceil(std::log(_max));
			_minin = std::ceil(std::log(_min));
		}
		auto _edges = arma::logspace(_minin, _maxin, this->nBins_ + 1);

		// create the bin edges
		for (u64 i = 0; i < this->nBins_; i++)
			this->binEdges_[i] = _edges(i);
	}

	// ######## Methods ########

	/*
	* @brief Append a value to the histogram and return the bin index. 
	* @param _value the value to append
	* @returns the bin index
	*/
	u64 append(long double _value) 
	{
		if (_value < this->binEdges_.front()) 
		{
			// first bin counts values in [-INF, a0)
			++binCounts_[0];
			return 0;
		}
		else if (_value >= this->binEdges_.back()) 
		{
			// last bin counts values in [aN-1, +INF)
			++binCounts_[this->nBins_];
			return nBins_;
		}

		const auto break_it = std::lower_bound(this->binEdges_.begin(), this->binEdges_.end(), _value);
		const auto _binIdx	= std::distance(this->binEdges_.begin(), break_it) + 1;
		++this->binCounts_[_binIdx];
		return _binIdx;
	}
};

/*
* @brief Additional properties for the histogram class, adding the bin averages
* This class allows one to have a function f(x) averaged over the bins
* The bins are intervals for the x values, and the binAverages are the average of the function over the bin
* and counts obtained from the number of times the function was evaluated in the bin
*/
template< typename _T>
class HistogramAverage : public Histogram
{
	using u64 = unsigned long long;
protected:
	arma::Col<_T> binAverages_;

public:

	~HistogramAverage() = default;
	HistogramAverage()						{ binAverages_ = arma::Col<_T>(this->nBins_ + 1, arma::fill::zeros); }
	HistogramAverage(u64 _N)
		: Histogram(_N)						{ binAverages_ = arma::Col<_T>(_N + 1, arma::fill::zeros); }

	// ######## Getters ########

	const arma::Col<_T>& averages()			const { return this->binAverages_;		}
	_T averages(u64 i)						const { return this->binAverages_[i];	}
	
	// -------------------------

	/*
	* @brief Get the average of the function over the bins. Normalized by the number of counts in the bin 
	* as the values are summed over in the bin.
	*/
	arma::Col<_T> averages_av(bool _isTypical = false) 
	{
		arma::Col<_T> _out = this->binAverages_;

		// normalize by the number of counts in the bin
		for (u64 i = 0; i <= this->nBins_; i++)
		{
			if (this->binCounts_[i] != 0)
			{
				_out(i) /= (long double)this->binCounts_[i];
			}
		}
		// if the average shall hold a typical value, then exponentiate the average as the 
		// average is of the logarithms
		if (_isTypical) 
			return arma::exp(_out);
		return _out;
	}

	// ######## Setters ########

	void reset() override final 
	{ 
		Histogram::reset(); 
		this->binAverages_.zeros();
	};

	// -------------------------

	void reset(u64 _N) override final
	{
		Histogram::reset(_N);
		binAverages_ = arma::Col<_T>(_N + 1, arma::fill::zeros);
	}

	// -------------------------

	void average()							
	{ 
		for (u64 i = 0; i <= this->nBins_; i++) 
			if(this->binCounts_[i] != 0) 
				this->binAverages_(i) /= this->binCounts_[i]; 
	}
	
	// ######## Methods ########

	u64 append(long double _value, const _T _element)
	{
		auto _binIdx = 0;
#ifndef _DEBUG
#pragma omp critical
#endif // !_DEBUG
		{
			// get the index of the bin
			_binIdx = Histogram::append(_value);
			// add the element to the bin average
			this->binAverages_(_binIdx) += _element;
			// return the bin index
		}
		return _binIdx;
	}

	// ##########################################################################################################################################

};