#include "../../src/Include/random.h"
#include "../../src/flog.h"
#include "../../src/common.h"

#ifdef MC_ENABLE_MPI
#   include <mpi.h>
#endif

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
};

// ##########################################################################################################################################
namespace MonteCarlo 
{
    // #################################################################################################################################

    // template instantiation
    template ParallelTempering<double, double, arma::Col<double>>::ParallelTempering();
    template ParallelTempering<float, float, arma::Col<float>>::ParallelTempering();
    template ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering();

    // #################################################################################################################################

    template <typename _T, class _stateType, class _Config_t>
    ParallelTempering<_T, _stateType, _Config_t>::~ParallelTempering()
    {
        if (this->pBar_)
            delete this->pBar_;
        this->pBar_ = nullptr;
    }

    // template instantiation
    template ParallelTempering<double, double, arma::Col<double>>::~ParallelTempering();
    template ParallelTempering<float, float, arma::Col<float>>::~ParallelTempering();
    template ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::~ParallelTempering();

    // #################################################################################################################################

    template <typename _T, class _stateType, class _Config_t>
    ParallelTempering<_T, _stateType, _Config_t>::ParallelTempering(Solver_p _MCS, const std::vector<double>& _betas, size_t _nSolvers)
        : threadPool_(_nSolvers), nSolvers_(_nSolvers), betas_(_betas), lastLosses_(_nSolvers, 0.0), accepted_(_nSolvers, 0), total_(_nSolvers, 0)
    {
        if (_nSolvers < 1)
            throw std::invalid_argument("The number of solvers must be greater than 0.");
        if (_nSolvers != this->betas_.size() && this->betas_.size() != 0)
            throw std::invalid_argument("The number of solvers must match the number of betas.");
        else if (this->betas_.size() == 0)
        {
            this->betas_ = std::vector<double>(_nSolvers);
            for (size_t i = 0; i < _nSolvers; ++i)
                this->betas_[i] = 1.0 / (double)(i + 1);
        }

        this->MCSs_.push_back(_MCS);
        this->replicate(this->nSolvers_);

        // Initialize the counters
        this->finished_     = std::vector<bool>(this->nSolvers_, false);
        this->total_        = std::vector<u64>(this->nSolvers_, 0);
        this->accepted_     = std::vector<u64>(this->nSolvers_, 0);
        this->losses_       = v_1d<Container_t>(this->nSolvers_);
        this->meanLosses_   = v_1d<Container_t>(this->nSolvers_);
        this->stdLosses_    = v_1d<Container_t>(this->nSolvers_);
    }

    // template instantiation
    template ParallelTempering<double, double, arma::Col<double>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);
    template ParallelTempering<float, float, arma::Col<float>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);
    template ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);

    // #################################################################################################################################

    template <typename _T, class _stateType, class _Config_t>
    void ParallelTempering<_T, _stateType, _Config_t>::replicate(size_t _nSolvers)
    {
        if (_nSolvers < 2)
            throw std::invalid_argument("The number of solvers must be greater than 1.");

        for (size_t i = 1; i < _nSolvers; ++i)
            this->MCSs_.push_back(this->MCSs_[0]->clone());        
    }

    // template instantiation
    template void ParallelTempering<double, double, arma::Col<double>>::replicate(size_t);
    template void ParallelTempering<float, float, arma::Col<float>>::replicate(size_t);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::replicate(size_t);

    // #################################################################################################################################

    template <typename _T, class _stateType, class _Config_t>
    ParallelTempering<_T, _stateType, _Config_t>::ParallelTempering(const std::vector<Solver_p>& _MCSs, const std::vector<double>& _betas)
        : threadPool_(_MCSs.size()), nSolvers_(_MCSs.size()), MCSs_(_MCSs), betas_(_betas), lastLosses_(_MCSs.size(), 0.0), accepted_(_MCSs.size(), 0), total_(_MCSs.size(), 0)
    {
        if (this->nSolvers_ < 1)
            throw std::invalid_argument("The number of solvers must be greater than 0.");
        if (this->nSolvers_ != this->betas_.size() && this->betas_.size() != 0)
            throw std::invalid_argument("The number of solvers must match the number of betas.");
        else if (this->betas_.size() == 0)
        {
            this->betas_ = std::vector<double>(this->nSolvers_);
            for (size_t i = 0; i < this->nSolvers_; ++i)
                this->betas_[i] = 1.0 / (double)(i + 1);
        }

        // Initialize the counters
        this->finished_     = std::vector<bool>(this->nSolvers_, false);
        this->total_        = std::vector<u64>(this->nSolvers_, 0);
        this->accepted_     = std::vector<u64>(this->nSolvers_, 0);
        this->losses_       = v_1d<Container_t>(this->nSolvers_);
        this->meanLosses_   = v_1d<Container_t>(this->nSolvers_);
        this->stdLosses_    = v_1d<Container_t>(this->nSolvers_);
    }

    // template instantiation
    template ParallelTempering<double, double, arma::Col<double>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);
    template ParallelTempering<float, float, arma::Col<float>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);
    template ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);

    // #################################################################################################################################

    /**
    * @brief Executes a training step for the ParallelTempering algorithm.
    * 
    * This function performs a training step for each Monte Carlo solver in the 
    * ParallelTempering instance and then attempts to swap configurations between 
    * solvers to enhance sampling efficiency.
    * 
    * @tparam _T The data type used by the Monte Carlo solvers.
    * @tparam _stateType The state type used by the Monte Carlo solvers.
    * @tparam _Config_t The container type used by the Monte Carlo solvers.
    * @tparam useMPI If true, enables MPI support for parallel tempering. This requires the MPI library to be enabled in the build.
    * @param i The current iteration index.
    * @param _par Training parameters for the Monte Carlo solver.
    * @param quiet If true, suppresses output during training.
    * @param randomStart If true, initializes solvers with random starting points.
    * @param _timer A Timer object to measure the duration of the training step.
    */
    template <typename _T, class _stateType, class _Config_t>
    template <bool useMPI>
    void ParallelTempering<_T, _stateType, _Config_t>::trainStep(size_t i,    
                                            const MonteCarlo::MCS_train_t& _par, 
                                            const bool quiet, 
                                            const bool randomStart,
                                            Timer& _timer)
    {

        if constexpr (useMPI)
        {
#ifndef MC_ENABLE_MPI
            throw std::runtime_error("MPI is not enabled in this build.");
#else
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Determine the range of solvers for this rank
        const size_t local_start = (this->nSolvers_ / size) * rank;
        const size_t local_end = (rank == size - 1) ? this->nSolvers_ : local_start + (this->nSolvers_ / size);

        for (size_t j = local_start; j < local_end; ++j)
        {
            if (!this->finished_[j])
                this->finished_[j] = this->MCSs_[j]->trainStep(i, this->losses_[j], this->meanLosses_[j], 
                                                this->stdLosses_[j], _par, quiet, randomStart, _timer);
        }
        
        // update counters
        for (size_t j = local_start; j < local_end; ++j)
        {
            this->total_[j]         = this->MCSs_[j]->getTotal();
            this->accepted_[j]      = this->MCSs_[j]->getAccepted();
            this->lastLosses_[j]    = this->MCSs_[j]->getLastLoss();
        }
#endif
        }
        else
        {
            for (size_t j = 0; j < this->nSolvers_; ++j)
            {
                if (this->finished_[j])                                     // if the solver has finished, skip it
                    continue;
                this->threadPool_.submit([this, i, j, &_par, quiet, randomStart, &_timer]() 
                    {
                        this->finished_[j] = this->MCSs_[j]->trainStep(i, this->losses_[j], 
                                                    this->meanLosses_[j], this->stdLosses_[j], 
                                                    _par, quiet, randomStart, _timer);
                    });
            }

            // Wait for all tasks to finish
            {
                std::unique_lock lock(this->threadPool_.mutex()); // Access thread pool's queue mutex
                threadPool_.cv().wait(lock, [this]() { return threadPool_.taskQueue().empty(); });
            }
            
            // update counters
            for (size_t j = 0; j < this->nSolvers_; ++j)
            {
                this->total_[j]         = this->MCSs_[j]->getTotal();
                this->accepted_[j]      = this->MCSs_[j]->getAccepted();
                this->lastLosses_[j]    = this->MCSs_[j]->getLastLoss();
            }
        }
    }

    // template instantiation
    // no MPI
    template void ParallelTempering<double>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer& _timer);
    template void ParallelTempering<float>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer& _timer);
    template void ParallelTempering<std::complex<double>>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer& _timer);
    // MPI
    template void ParallelTempering<double>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer& _timer);
    template void ParallelTempering<float>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer& _timer);
    template void ParallelTempering<std::complex<double>>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer& _timer);

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
    template <typename _T, class _stateType, class _Config_t>
    void ParallelTempering<_T, _stateType, _Config_t>::swap(size_t i, size_t j)
    {        
        if (i == j || i >= this->nSolvers_ || j >= this->nSolvers_ || this->finished_[i] || this->finished_[j])
            return;

        // Calculate the acceptance probability
        const _T _loss_i    = this->MCSs_[i]->getLastLoss();
        const _T _loss_j    = this->MCSs_[j]->getLastLoss();
        const _T _delta     = (_loss_i - _loss_j) * (this->betas_[i] - this->betas_[j]);
        const _T _prob      = std::exp(_delta);

        if (this->MCSs_[i]->getRandomVal() < std::abs(_prob))
        {
            std::lock_guard<std::mutex> lock(this->swapMutex_); // use the mutex to protect the swap operation
            this->MCSs_[i]->swapConfig(this->MCSs_[j]);         // swap the configurations
            LOGINFO("Swapped configurations between solvers " + std::to_string(i) + " and " + std::to_string(j) + ".", LOG_TYPES::DEBUG, 3);
        }
    }
    
    // template instantiation
    template void ParallelTempering<double>::swap(size_t i, size_t j);
    template void ParallelTempering<float>::swap(size_t i, size_t j);
    template void ParallelTempering<std::complex<double>>::swap(size_t i, size_t j);

    // #################################################################################################################################
    
    /**
    * @brief Perform swaps between solvers in parallel tempering, skipping finished solvers.
    *
    * This function iterates through the solvers and attempts to perform a swap operation
    * between the current solver and the next available unfinished solver. If no suitable
    * solver is found, the current solver is skipped.
    *
    * @note This ensures swaps are only performed when valid solvers are available, improving
    *       the effectiveness of parallel tempering.
    *
    * @tparam _T The type of the solvers.
    */
    template <typename _T, class _stateType, class _Config_t>
    void ParallelTempering<_T, _stateType, _Config_t>::swaps()
    {
        size_t i = 0;
        while (i < this->nSolvers_ - 1)
        {
            if (this->finished_[i])                                 // Skip finished solvers
            {
                ++i;
                continue;
            }

            size_t j = i + 1;
            
            while (j < this->nSolvers_ && this->finished_[j])       // Find the next unfinished solver
                ++j;

            if (j < this->nSolvers_)                                // If a valid solver is found, perform the swap
            {
                this->swap(i, j);
                i = j;
            }
            else 
                break;                                              // Exit the loop if no valid solver is found
        }
    }

    // template instantiation
    template void ParallelTempering<double>::swaps();
    template void ParallelTempering<float>::swaps();
    template void ParallelTempering<std::complex<double>>::swaps();

    // #################################################################################################################################

    /**
    * @brief Trains a single Markov Chain Monte Carlo (MCMC) simulation without using parallel tempering.
    * This is equivalent to not using the replica exchange algorithm.
    * 
    * This function trains a single MCMC simulation using the parameters provided. It utilizes the 
    * first MCMC simulation in the `MCSs_` vector to perform the training. The training process is
    * executed in a single thread and given by the inherited `train` function.
    * 
    * @tparam _T The type of the data being processed.
    * @tparam _stateType The type representing the state of the MCMC simulation.
    * @tparam _Config_t The type representing the configuration of the MCMC simulation.
    * 
    * @param _par The parameters for the MCMC training.
    * @param quiet A boolean flag indicating whether to suppress output during training.
    * @param ranStart A boolean flag indicating whether to start the training with a random state.
    * @param _t The starting time point for the training.
    * @param progPrc The progress percentage of the training.
    */
    template <typename _T, class _stateType, class _Config_t>
    void ParallelTempering<_T, _stateType, _Config_t>::trainSingle(const MCS_train_t& _par, bool quiet, bool ranStart, clk::time_point _t, uint progPrc)
    {
        this->MCSs_[0]->train(_par, quiet, ranStart, _t, progPrc);
    }

    // template instantiation
    template void ParallelTempering<double, double, arma::Col<double>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<float, float, arma::Col<float>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);

    // #################################################################################################################################

    /**
    * @brief Trains the Parallel Tempering model using Monte Carlo simulations.
    * 
    * @tparam _T The data type used for the training.
    * @tparam _stateType The type representing the state of the system.
    * @tparam _Config_t The configuration type.
    * @tparam useMPI A boolean template parameter indicating whether to use MPI for parallel processing.
    * 
    * @param _par The parameters for the Monte Carlo simulation training.
    * @param quiet A boolean flag indicating whether to suppress output.
    * @param ranStart A boolean flag indicating whether to start with a random state.
    * @param _t The starting time point for the training.
    * @param progPrc The progress percentage for the progress bar.
    * 
    * @throws std::invalid_argument If the number of solvers is less than 1.
    * @throws std::runtime_error If MPI is not enabled in the build but useMPI is true.
    * 
    * This function initializes the training process for the Parallel Tempering model. It supports both single and multiple solvers.
    * If MPI is enabled and useMPI is true, it initializes the MPI environment. The function sets up the progress bar, resets the solvers,
    * and performs the training steps. During each training step, it updates the progress bar and checks if all solvers have finished training.
    * If all solvers are finished, it exits the training loop. If MPI is enabled, it finalizes the MPI environment at the end.
    */
    template <typename _T, class _stateType, class _Config_t>
    template <bool useMPI>
    void ParallelTempering<_T, _stateType, _Config_t>::train(const MCS_train_t& _par, bool quiet, bool ranStart, clk::time_point _t, uint progPrc)
    {
        if (this->nSolvers_ == 1)
        {
            return this->trainSingle(_par, quiet, ranStart, _t, progPrc);
        }

        if (this->nSolvers_ < 1)
            throw std::invalid_argument("The number of solvers must be greater than 0.");

        // Initialize MPI if useMPI is true
        if constexpr (useMPI) 
        {
            #ifdef MC_ENABLE_MPI
                MPI_Init(nullptr, nullptr);                                 // Initialize MPI environment
            #else
                throw std::runtime_error("MPI is not enabled in this build.");
            #endif
        }
        {
            this->pBar_ = new pBar(progPrc, _par.MC_sam_);					// set the progress bar		
            _par.hi();														// set the info about training
            for (size_t i = 0; i < this->nSolvers_; ++i)
                this->MCSs_[i]->reset(_par.nblck_);						    // reset the derivatives
        }

        v_1d<Timer> timers(this->nSolvers_);							    // timer for the training
        this->losses_.resize(this->nSolvers_);							    // losses for each solver
        for (size_t i = 0; i < this->nSolvers_; ++i)    
            this->losses_[i].resize(_par.bsize_);			    	        // losses for each solver
        this->meanLosses_.resize(this->nSolvers_);				            // mean losses for each solver
        for (size_t i = 0; i < this->nSolvers_; ++i)    
            this->meanLosses_[i].resize(_par.MC_sam_);	    		        // mean losses for each solver
        this->stdLosses_.resize(this->nSolvers_);					        // standard deviation of the losses for each solver
        for (size_t i = 0; i < this->nSolvers_; ++i)    
            this->stdLosses_[i].resize(_par.MC_sam_);				        // standard deviation of the losses for each solver
        
        for (size_t i = 0; i < this->nSolvers_; ++i) 
        {
            this->MCSs_[i]->setRandomState();       	    			    // set the random state at the begining and the number of flips
            this->MCSs_[i]->setRandomFlipNum(_par.nFlip);				    // set the random state at the begining and the number of flips
        }
    
        for (size_t i = 0; i < _par.MC_sam_; ++i)                           // go through the training steps
        {
            this->trainStep<useMPI>(i, _par, quiet, ranStart, timers[i]);   // perform the training step

            // inform the user about the progress
            { 
                double _bestLoss = std::numeric_limits<double>::max(), _bestAcc = 0.0;
                size_t _bestIdx = 0, _bestAccIdx = 0;
                std::string _prog;
                for (size_t j = 0; j < this->nSolvers_; ++j)
                {
                    const double _currLoss = algebra::cast<double>(this->lastLosses_[j]);
                    if (_currLoss < _bestLoss)
                    {
                        _bestLoss = _currLoss;
                        _bestIdx = j;
                    }
                    const double _currAcc = (double)this->accepted_[j] / this->total_[j];
                    if (_currAcc > _bestAcc)
                    {
                        _bestAcc = _currAcc;
                        _bestAccIdx = j;
                    }
                }
                _prog = "Iteration " + std::to_string(i) + "/" + std::to_string(_par.MC_sam_) +
                    ", Best Loss: " + std::to_string(_bestLoss) +
                    ", Best Acceptance: " + std::to_string(_bestAcc * 100) + "%" +
                    ", Best Solver Index: " + std::to_string(_bestIdx) +
                    ", Best Acceptance Solver Index: " + std::to_string(_bestAccIdx);
                PROGRESS_UPD_Q(i, (*this->pBar_), _prog, !quiet);           // update the progress bar
            }

            if (std::all_of(this->finished_.begin(), this->finished_.end(), [](bool f) { return f; }))
            {
                LOGINFO("All solvers have finished training.", LOG_TYPES::INFO, 1);
                break;                                                      // Exit the training loop
            }
            this->swaps();                                                  // Perform swaps between solvers                                          
        }

        if constexpr (useMPI) 
        {                                                                    // Finalize MPI if useMPI is true
            #ifdef MC_ENABLE_MPI
                MPI_Finalize();  // Finalize MPI environment
            #else
                throw std::runtime_error("MPI is not enabled in this build.");
            #endif
        }
    }

    // template instantiation   
    // no MPI
    template void ParallelTempering<double, double, arma::Col<double>>::train<false>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<float, float, arma::Col<float>>::train<false>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::train<false>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    // MPI
    template void ParallelTempering<double, double, arma::Col<double>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<float, float, arma::Col<float>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);

    // #################################################################################################################################
    
};

// ##########################################################################################################################################
