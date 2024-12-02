#include "../../src/Include/random.h"
#include "../../src/flog.h"
#include "../../src/common.h"

// ##########################################################################################################################################

/**
* @brief Logs the configuration details of the MCS training process.
*
* This function constructs a formatted string containing various parameters
* related to the Monte Carlo training process and logs it * with a specified log level and type.
*
* @param _in A string to be prefixed to the log message.
*
* The logged information includes:
* - Monte Carlo Samples: The number of Monte Carlo samples used in the training.
* - Thermalization Steps: The number of thermalization steps performed.
* - Number of blocks (single sample): The number of blocks in a single sample.
* - Size of the single block: The size of each block.
* - Number of flips taken at each step: The number of flips performed at each step.
*/
void MonteCarlo::MCS_train_t::hi(const std::string& _in) const
{
    std::string outstr = "";
    strSeparatedP(outstr, ',', 2,
                VEQV(Monte Carlo Samples, this->MC_sam_),
                VEQV(Thermalization Steps, this->MC_th_),
                VEQV(Number of blocks (single sample), this->nblck_),
                VEQV(Size of the single block, this->bsize_),
                VEQV(Number of flips taken at each step, this->nFlip));
    LOGINFOG(_in + outstr, LOG_TYPES::TRACE, 1);
}

// ##########################################################################################################################################

namespace MonteCarlo 
{

	// #################################################################################################################################

	/**
	@brief Mean calculation for Monte Carlo data.
		Calculates the mean and standard deviation of the given data.
	@param _data Data to be analyzed.
	@param _mean Pointer to store the calculated mean value.
	@param _std (Optional) Pointer to store the calculated standard deviation.
	*/
	template <typename _T, typename COLTYPE>
	void mean(const COLTYPE& _data, _T* _mean, _T* _std) 
	{
		if (!_mean)
			throw std::invalid_argument("Invalid mean pointer for mean calculation.");
		*_mean = algebra::cast<_T>(arma::mean(_data));
		if (_std)
			*_std = algebra::cast<_T>(arma::stddev(_data, 0)); // Use sample standard deviation
	}
    
    // template instantiation
    template void mean(const arma::Col<double>&, double*, double*);
    template void mean(const arma::Col<float>&, float*, float*);
    template void mean(const arma::Col<std::complex<double>>&, std::complex<double>*, std::complex<double>*);
    // arma subview
    template void mean(const arma::subview_col<double>&, double*, double*);
    template void mean(const arma::subview_col<float>&, float*, float*);
    template void mean(const arma::subview_col<std::complex<double>>&, std::complex<double>*, std::complex<double>*);

	// specialization for std::vector
	template <typename _T>
	void mean(const std::vector<_T>& _data, _T* _mean, _T* _std) 
	{
		if (!_mean)
			throw std::invalid_argument("Invalid mean pointer for mean calculation.");

		*_mean = algebra::cast<_T>(std::accumulate(_data.begin(), _data.end(), _T(0.0)) / (double)_data.size());
		if (_std) {
			_T _m = *_mean;
			*_std = algebra::cast<_T>(std::sqrt(std::accumulate(_data.begin(), _data.end(), _T(0.0), [_m](auto _sum, auto _val) { return _sum + (_val - _m) * (_val - _m); }) / double(_data.size() - 1)));
		}
	}

    // template instantiation
    template void mean(const std::vector<double>&, double*, double*);
    template void mean(const std::vector<float>&, float*, float*);
    template void mean(const std::vector<std::complex<double>>&, std::complex<double>*, std::complex<double>*);

	// #################################################################################################################################

	/**
	@brief Block mean calculation for correlated Monte Carlo data.
		Calculates the mean and standard deviation using block averaging.

	@param _data Data to be analyzed.
	@param _blockSize Size of each block.
	@param _mean Pointer to store the calculated mean value.
	@param _std (Optional) Pointer to store the calculated standard deviation.
	@throws std::invalid_argument If block size is larger than the data size.
	*/
	template <typename _T, typename COLTYPE>
	void blockmean(const COLTYPE& _data, size_t _blockSize, _T* _mean, _T* _std) 
	{
		if (!_mean)
			throw std::invalid_argument("Invalid mean pointer for block mean calculation.");

		if (_blockSize == 0 || _data.n_elem < _blockSize) {		// Check for valid block size	
			// LOGINFO("Invalid block size for block mean calculation.", LOG_TYPES::WARNING, 1);
			return MonteCarlo::mean(_data, _mean, _std);
		}

		const size_t _nBlocks = _data.n_elem / _blockSize;		// Calculate the number of blocks

		// Reshape data into blocks and calculate block means
		arma::Mat<_T> reshapedData 	= arma::reshape(_data.head(_nBlocks * _blockSize), _blockSize, _nBlocks);
		arma::Col<_T> blockMeans 	= arma::mean(reshapedData, 0).t();

		// Calculate the overall mean
		*_mean = algebra::cast<_T>(arma::mean(blockMeans));

		// Calculate the standard deviation of block means, if requested
		if (_std)
			*_std = algebra::cast<_T>(arma::stddev(blockMeans, 0)); // Use sample standard deviation
	}

    // template instantiation
    template void blockmean(const arma::Col<double>&, size_t, double*, double*);
    template void blockmean(const arma::Col<float>&, size_t, float*, float*);
    template void blockmean(const arma::Col<std::complex<double>>&, size_t, std::complex<double>*, std::complex<double>*);
    // arma subview
    template void blockmean(const arma::subview_col<double>&, size_t, double*, double*);
    template void blockmean(const arma::subview_col<float>&, size_t, float*, float*);
    template void blockmean(const arma::subview_col<std::complex<double>>&, size_t, std::complex<double>*, std::complex<double>*);

	// specialization for std::vector
	template <typename _T>
	void blockmean(const std::vector<_T>& _data, size_t _blockSize, _T* _mean, _T* _std) 
	{
		if (_blockSize == 0 || _data.size() < _blockSize)
			return MonteCarlo::mean(_data, _mean, _std);

		if (!_mean)
			throw std::invalid_argument("Invalid mean pointer for block mean calculation.");

		const double invBlockSize 	= 1.0 / (double)_blockSize;
		size_t _nBlocks 			= (double)_data.size() * invBlockSize;

		if (_nBlocks == 0) {								// Check for valid block size
			*_mean = 0.0;
			return;
		}									
			

		_T sumBlockMeans 		= 0.0;
		_T sumSquareBlockMeans 	= 0.0;
		for (size_t i = 0; i < _nBlocks; ++i) 				// Reshape data into blocks and calculate block means
		{
			const auto blockBegin 	= _data.begin() + i * _blockSize;
			const auto blockEnd	 	= blockBegin + _blockSize;
			_T blockMean 			= std::accumulate(blockBegin, blockEnd, _T(0.0)) * invBlockSize;
			sumBlockMeans 			+= blockMean;
			sumSquareBlockMeans 	+= blockMean * blockMean;
		}

		*_mean = sumBlockMeans * invBlockSize;				// Calculate the overall mean
		if (_std) {											// Calculate the standard deviation of block means, if requested
			_T _var = (sumSquareBlockMeans * invBlockSize) - (*_mean) * (*_mean);
			*_std 	= std::sqrt(_var);
		}
	}

    // template instantiation
    template void blockmean(const std::vector<double>&, size_t, double*, double*);
    template void blockmean(const std::vector<float>&, size_t, float*, float*);
    template void blockmean(const std::vector<std::complex<double>>&, size_t, std::complex<double>*, std::complex<double>*);

	// #################################################################################################################################


	// #################################################################################################################################

};



// ##########################################################################################################################################

// Parallel tempering class

// ##########################################################################################################################################

namespace MonteCarlo 
{
    // #################################################################################################################################
    // template instantiation
    template class MonteCarloSolver<double, double, arma::Col<double>>;
    template class MonteCarloSolver<float, float, arma::Col<float>>;
    template class MonteCarloSolver<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>;
    // mix
    template class MonteCarloSolver<double, std::complex<double>, arma::Col<std::complex<double>>>;
    template class MonteCarloSolver<std::complex<double>, double, arma::Col<double>>;

    // #################################################################################################################################
    // template instantiation
    template class ParallelTempering<double>;
    template class ParallelTempering<float>;
    template class ParallelTempering<std::complex<double>>;
    // #################################################################################################################################

    template <typename _T, typename _stateType, typename _Config_t>
    double MonteCarloSolver<_T, _stateType, _Config_t>::getRandomVal() const 
    {
        return (this->ran_) ? this->ran_->template random<double>() : 0.0;
    }

    // template instantiation
    template double MonteCarloSolver<double, double, arma::Col<double>>::getRandomVal() const;
    template double MonteCarloSolver<float, float, arma::Col<float>>::getRandomVal() const;
    template double MonteCarloSolver<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::getRandomVal() const;
    // mix
    template double MonteCarloSolver<std::complex<double>, double, arma::Col<double>>::getRandomVal() const;
    template double MonteCarloSolver<double, std::complex<double>, arma::Col<std::complex<double>>>::getRandomVal() const;

    // #################################################################################################################################

    template <typename T, typename U, typename V>
    MonteCarloSolver<T, U, V>::~MonteCarloSolver() {}

    // template instantiation
    template MonteCarloSolver<double, double, arma::Col<double>>::~MonteCarloSolver();
    template MonteCarloSolver<float, float, arma::Col<float>>::~MonteCarloSolver();
    template MonteCarloSolver<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::~MonteCarloSolver();
    // mix
    template MonteCarloSolver<double, std::complex<double>, arma::Col<std::complex<double>>>::~MonteCarloSolver();
    template MonteCarloSolver<std::complex<double>, double, arma::Col<double>>::~MonteCarloSolver();
    
    // #################################################################################################################################
    
    // #################################################################################################################################

    // #################################################################################################################################


    template <typename _T>
    ParallelTempering<_T>::~ParallelTempering()
    {
        if (this->pBar_)
            delete this->pBar_;
        this->pBar_ = nullptr;
    }


    /**
    * @brief Executes a training step for the ParallelTempering algorithm.
    * 
    * This function performs a training step for each Monte Carlo solver in the 
    * ParallelTempering instance and then attempts to swap configurations between 
    * solvers to enhance sampling efficiency.
    * 
    * @tparam _T The data type used by the Monte Carlo solvers.
    * @param i The current iteration index.
    * @param En A container to store energy values.
    * @param meanEn A container to store mean energy values.
    * @param stdEn A container to store standard deviation of energy values.
    * @param _par Training parameters for the Monte Carlo solver.
    * @param quiet If true, suppresses output during training.
    * @param randomStart If true, initializes solvers with random starting points.
    * @param _timer A Timer object to measure the duration of the training step.
    */
    template <typename _T>
    void ParallelTempering<_T>::trainStep(  size_t i,    
                                            MonteCarloSolver<_T>::Container_t& En,
                                            MonteCarloSolver<_T>::Container_t& meanEn, 
                                            MonteCarloSolver<_T>::Container_t& stdEn, 
                                            const MonteCarlo::MCS_train_t& _par, 
                                            const bool quiet, 
                                            const bool randomStart,
                                            Timer& _timer)
    {
        for (size_t j = 0; j < this->nSolvers_; ++j)
            this->MCSs_[j]->trainStep(i, En, meanEn, stdEn, _par, quiet, randomStart, _timer);
        this->swaps();
    }

    // template instantiation
    template void ParallelTempering<double>::trainStep(  size_t i,    
                                                        MonteCarloSolver<double>::Container_t& En,
                                                        MonteCarloSolver<double>::Container_t& meanEn, 
                                                        MonteCarloSolver<double>::Container_t& stdEn, 
                                                        const MonteCarlo::MCS_train_t& _par, 
                                                        const bool quiet, 
                                                        const bool randomStart,
                                                        Timer& _timer);
    template void ParallelTempering<float>::trainStep(   size_t i,
                                                        MonteCarloSolver<float>::Container_t& En,
                                                        MonteCarloSolver<float>::Container_t& meanEn, 
                                                        MonteCarloSolver<float>::Container_t& stdEn, 
                                                        const MonteCarlo::MCS_train_t& _par, 
                                                        const bool quiet, 
                                                        const bool randomStart,
                                                        Timer& _timer);
    template void ParallelTempering<std::complex<double>>::trainStep(  size_t i,
                                                        MonteCarloSolver<std::complex<double>>::Container_t& En,
                                                        MonteCarloSolver<std::complex<double>>::Container_t& meanEn, 
                                                        MonteCarloSolver<std::complex<double>>::Container_t& stdEn, 
                                                        const MonteCarlo::MCS_train_t& _par, 
                                                        const bool quiet, 
                                                        const bool randomStart,
                                                        Timer& _timer);
    // #################################################################################################################################

    /**
    * @brief Swaps the states of two Monte Carlo simulations in the Parallel Tempering algorithm.
    *
    * This function calculates the acceptance probability for swapping the states of two Monte Carlo
    * simulations at indices `i` and `j`. The acceptance probability is based on the difference in 
    * their last loss values and their respective beta values. If a randomly generated value is less 
    * than the calculated acceptance probability, the states are swapped.
    *
    * @tparam _T The data type used for the loss values and probabilities.
    * @param i The index of the first Monte Carlo simulation.
    * @param j The index of the second Monte Carlo simulation.
    */
    template <typename _T>
    void ParallelTempering<_T>::swap(size_t i, size_t j)
    {
        // Calculate the acceptance probability
        const _T _loss_i    = this->MCSs_[i]->getLastLoss();
        const _T _loss_j    = this->MCSs_[j]->getLastLoss();
        const _T _delta     = (_loss_i - _loss_j) * (this->betas_[i] - this->betas_[j]);
        _T _prob            = std::exp(_delta);

        if (this->MCSs_[i]->getRandomVal() < std::abs(_prob))
        {
            // !TODO implement the swap
            auto _config_i = this->MCSs_[i]->getLastConfig();
            auto _config_j = this->MCSs_[j]->getLastConfig();
            this->MCSs_[i]->setConfig(_config_j);
            this->MCSs_[j]->setConfig(_config_i);
        }
    }
    
    // template instantiation
    template void ParallelTempering<double>::swap(size_t i, size_t j);
    template void ParallelTempering<float>::swap(size_t i, size_t j);
    template void ParallelTempering<std::complex<double>>::swap(size_t i, size_t j);

    // #################################################################################################################################
    
    /**
    * @brief Perform swaps between adjacent solvers in parallel tempering.
    *
    * This function iterates through the solvers and performs a swap operation
    * between each pair of adjacent solvers. The swaps are intended to facilitate
    * the parallel tempering process, which is a technique used in optimization
    * and sampling algorithms to improve convergence by allowing solvers to
    * exchange information
    *
    * @note swaps all adjacent solvers in the list i, i+1 for i = 0, 1, ..., nSolvers - 2
    *
    * @tparam _T The type of the solvers.
    */
    template <typename _T>
    void ParallelTempering<_T>::swaps()
    {
        for (size_t i = 0; i < this->nSolvers_ - 1; ++i)
            this->swap(i, i + 1);
    }

    // template instantiation
    template void ParallelTempering<double>::swaps();
    template void ParallelTempering<float>::swaps();
    template void ParallelTempering<std::complex<double>>::swaps();

    // #################################################################################################################################

    template <typename _T>
    void ParallelTempering<_T>::train(const MCS_train_t& _par, bool quiet, clk::time_point _t, uint progPrc)
    {
        // Initialize the energy containers
        {
            this->pBar_ = new pBar(progPrc, _par.MC_sam_);					// set the progress bar		
            _par.hi();														// set the info about training
            for (size_t i = 0; i < this->nSolvers_; ++i)
                this->MCSs_[i]->reset(_par.nblck_);						    // reset the derivatives
        }

        Timer _timer;														// timer for the training
        Container_t meanEn(_par.MC_sam_, arma::fill::zeros);				// here we save the mean energy
        Container_t stdEn(_par.MC_sam_, arma::fill::zeros);					// here we save the standard deviation of the energy
        Container_t En(_par.nblck_, arma::fill::zeros);						// history of energies (for given weights) - here we save the local energies at each block
        
        for (size_t i = 0; i < this->nSolvers_; ++i) 
        {
            this->MCSs_[i]->setRandomState();       	    				// set the random state at the begining and the number of flips
            
        }
            // this->trainStep(i, En, meanEn, stdEn, _par, quiet, randomStart, _timer);
        // this->setRandomState();												// set the random state at the begining and the number of flips
        // this->setRandomFlipNum(_par.nFlip);									// set the random state at the begining and the number of flips

        // Perform the training steps

        // std::vector<std::future<void>> futures;

        // for (size_t i = 0; i < _par.MC_sam_; ++i)
            // this->trainStep(i, En, meanEn, stdEn, _par, quiet, randomStart, _timer);
    }

    // template instantiation
    template void ParallelTempering<double>::train(const MCS_train_t& _par, bool quiet, clk::time_point _t, uint progPrc);
    template void ParallelTempering<float>::train(const MCS_train_t& _par, bool quiet, clk::time_point _t, uint progPrc);
    template void ParallelTempering<std::complex<double>>::train(const MCS_train_t& _par, bool quiet, clk::time_point _t, uint progPrc);

    // #################################################################################################################################
    
};