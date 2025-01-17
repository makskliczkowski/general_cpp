#include "../../src/Include/random.h"
#include "../../src/flog.h"
#include "../../src/common.h"
#include <limits>

#ifdef MC_ENABLE_MPI
#   include <mpi.h>
#endif

// ##########################################################################################################################################

/**
* @brief Resets the random number generator with a new seed.
* 
* This function deletes the existing random number generator (if any) and 
* creates a new one with the provided seed.
* 
* @tparam _T The type parameter for the MonteCarloSolver.
* @tparam _stateType The type parameter for the state.
* @tparam _CT The type parameter for the configuration.
* @param _seed The seed value to initialize the random number generator.
*/
template <typename _T, typename _stateType, typename _CT>
void MonteCarlo::MonteCarloSolver<_T, _stateType, _CT>::reset_random(size_t _seed)
{
    if(this->ran_) 
        delete this->ran_;
    this->ran_ = new randomGen(_seed);
}
// template instantiation
template void MonteCarlo::MonteCarloSolver<double, arma::Col<double>, arma::Mat<double>>::reset_random(size_t);
template void MonteCarlo::MonteCarloSolver<float, arma::Col<float>, arma::Mat<float>>::reset_random(size_t);
template void MonteCarlo::MonteCarloSolver<std::complex<double>, arma::Col<std::complex<double>>, arma::Mat<std::complex<double>>>::reset_random(size_t);

// ##########################################################################################################################################

// constructors
template <typename _T, typename _stateType, typename _CT>
MonteCarlo::MonteCarloSolver<_T, _stateType, _CT>::MonteCarloSolver(const MonteCarloSolver<_T, _stateType, _CT>& _n)
{
    // copy the random generator - !TODO: should copy the seed???
    size_t seed_ [[maybe_unused]] = _n.ran_->seed();
    this->ran_          = new randomGen();
    // copy the progress bar
    // this->pBar_         = new pBar();
    // copy the information
    this->info_         = _n.info_;
    // copy the Hamiltonian
    this->accepted_     = _n.accepted_;
    this->total_        = _n.total_;
    this->lastLoss_     = _n.lastLoss_;
    this->beta_         = _n.beta_;
    this->info_         = _n.info_;
    this->replica_      = _n.replica_;
}

/**
* @brief Move constructor for MonteCarloSolver.
*
* This constructor initializes a MonteCarloSolver object by moving the resources
* from another MonteCarloSolver object. It transfers ownership of the random 
* generator, progress bar, and other relevant information from the source object 
* to the newly created object.
*
* @tparam _T The type parameter for the solver.
* @tparam _stateType The type parameter for the state.
* @tparam _CT The type parameter for the configuration.
* @param _n The MonteCarloSolver object to move from.
*/
template <typename _T, typename _stateType, typename _CT>
MonteCarlo::MonteCarloSolver<_T, _stateType, _CT>::MonteCarloSolver(MonteCarloSolver<_T, _stateType, _CT>&& _n)
{
    // move the random generator
    this->ran_          = _n.ran_;
    _n.ran_             = nullptr;
    // move the progress bar
    this->pBar_         = _n.pBar_;
    _n.pBar_            = nullptr;
    // move the information
    this->info_         = std::move(_n.info_);
    // move the Hamiltonian
    this->accepted_     = _n.accepted_;
    this->total_        = _n.total_;
    this->lastLoss_     = _n.lastLoss_;
    this->beta_         = _n.beta_;
    this->info_         = std::move(_n.info_);
    this->replica_      = _n.replica_;
}

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

    template <typename _T, typename _stateType, typename _Config_t>
    MonteCarloSolver<_T, _stateType, _Config_t>::MonteCarloSolver()
    {
        this->ran_  = new randomGen();
    }
    // template instantiation
    template MonteCarloSolver<double, double, arma::Col<double>>::MonteCarloSolver();
    template MonteCarloSolver<float, float, arma::Col<float>>::MonteCarloSolver();
    template MonteCarloSolver<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::MonteCarloSolver();
    // mix
    template MonteCarloSolver<double, std::complex<double>, arma::Col<std::complex<double>>>::MonteCarloSolver();
    template MonteCarloSolver<std::complex<double>, double, arma::Col<double>>::MonteCarloSolver();

    // #################################################################################################################################

    template <typename T, typename U, typename V>
    MonteCarloSolver<T, U, V>::~MonteCarloSolver() 
    {
        if (this->pBar_)
            delete this->pBar_;
        this->pBar_ = nullptr;
    }

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

    /**
    * @brief Static function to generate beta values based on the spacing type.
    *
    * @param nBetas The number of beta values to generate.
    * @param minBeta The minimum beta value (1 / max temperature).
    * @param maxBeta The maximum beta value (1 / min temperature).
    * @param spacing The type of spacing to use (linear, geometric, etc.).
    * @return A vector of generated beta values.
    */
    template <typename _T, typename _stateType, typename _Config_t>
    std::vector<double> ParallelTempering<_T, _stateType, _Config_t>::generateBetas(size_t nBetas, BetaSpacing spacing, double minBeta, double maxBeta) 
    {
        std::vector<double> betas;
        
        if (nBetas == 1)
            return { 1.0 };

        if (nBetas < 1 || minBeta <= 0.0 || maxBeta <= minBeta) {
            throw std::invalid_argument("Invalid arguments for beta generation");
        }

        switch (spacing) 
        {
        case BetaSpacing::LINEAR:
            for (size_t i = 0; i < nBetas; ++i) {
                betas.push_back(minBeta + i * (maxBeta - minBeta) / (nBetas - 1));
            }
            break;
        case BetaSpacing::GEOMETRIC: {
            double ratio = std::pow(maxBeta / minBeta, 1.0 / (nBetas - 1));
            for (size_t i = 0; i < nBetas; ++i) {
                betas.push_back(minBeta * std::pow(ratio, i));
            }
            break;
        }
        case BetaSpacing::LOGARITHMIC:
            for (size_t i = 0; i < nBetas; ++i) {
                betas.push_back(minBeta + (maxBeta - minBeta) * std::log(1.0 + i) / std::log(1.0 + nBetas - 1));
            }
            break;
        case BetaSpacing::ADAPTIVE:
            throw std::runtime_error("Adaptive beta generation is not implemented yet");
            break;
        default:
            throw std::invalid_argument("Unknown beta spacing type");
        }

        return betas;
    }

    // #################################################################################################################################

    // template class instantiation
    template class ParallelTempering<double, double, arma::Col<double>>;
    template class ParallelTempering<float, float, arma::Col<float>>;
    template class ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>;
    template class ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>;
    template class ParallelTempering<std::complex<double>, double, arma::Col<double>>;

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
    template ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::~ParallelTempering();
    template ParallelTempering<std::complex<double>, double, arma::Col<double>>::~ParallelTempering();

    // #################################################################################################################################

    /**
    * @brief Constructor for the ParallelTempering class.
    *
    * This constructor initializes the ParallelTempering object with the given parameters.
    *
    * @tparam _T The type of the elements.
    * @tparam _stateType The type of the state.
    * @tparam _Config_t The type of the configuration.
    * @param _MCS A pointer to the solver.
    * @param _betas A vector of beta values.
    * @param _nSolvers The number of solvers.
    *
    * @throws std::invalid_argument If the number of solvers is less than 1.
    * @throws std::invalid_argument If the number of solvers does not match the number of betas.
    *
    * This constructor performs the following steps:
    * - Initializes the thread pool with the number of solvers.
    * - Checks if the number of solvers is valid.
    * - Checks if the number of solvers matches the number of betas.
    * - If no betas are provided, initializes the betas with default values.
    * - Adds the provided solver to the list of solvers.
    * - Replicates the solver for the number of solvers.
    * - Sets the beta value for each solver.
    * - Initializes various counters and containers.
    */
    template <typename _T, class _stateType, class _Config_t>
    ParallelTempering<_T, _stateType, _Config_t>::ParallelTempering(Solver_p _MCS, const std::vector<double>& _betas, size_t _nSolvers)
        : threadPool_(_nSolvers), nSolvers_(_nSolvers), betas_(_betas), lastLosses_(_nSolvers, 0.0), accepted_(_nSolvers, 0), total_(_nSolvers, 0)
    {
        if (_nSolvers < 1)
            throw std::invalid_argument("The number of solvers must be greater than 1.");
        if (_nSolvers != this->betas_.size() && this->betas_.size() != 0)
            throw std::invalid_argument("The number of solvers must match the number of betas.");
        else if (this->betas_.size() == 0)
        {
            this->betas_ = std::vector<double>(_nSolvers);
            for (size_t i = 0; i < _nSolvers; ++i)
            {
                this->betas_[i] = 1.0 / (double)(i + 1);
            }
        }

        this->MCSs_.push_back(_MCS);
        this->replicate(this->nSolvers_);
        // Initialize the counters
        for (size_t i = 0; i < this->nSolvers_; ++i)
        {
            this->MCSs_[i]->setBeta(this->betas_[i]);
        }

        // Initialize the counters
        this->finished_     = std::vector<bool>(this->nSolvers_, false);
        this->errors_       = std::vector<bool>(this->nSolvers_, false);
        this->total_        = std::vector<u64>(this->nSolvers_, 0);
        this->accepted_     = std::vector<u64>(this->nSolvers_, 0);
        this->losses_       = v_1d<Container_t>(this->nSolvers_);
        this->meanLosses_   = v_1d<Container_t>(this->nSolvers_);
        this->stdLosses_    = v_1d<Container_t>(this->nSolvers_);
        this->bestLosses_.clear();
    }

    // template instantiation
    template ParallelTempering<double, double, arma::Col<double>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);
    template ParallelTempering<float, float, arma::Col<float>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);
    template ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);
    template ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);
    template ParallelTempering<std::complex<double>, double, arma::Col<double>>::ParallelTempering(Solver_p, const std::vector<double>&, size_t);

    // #################################################################################################################################

    template <typename _T, class _stateType, class _Config_t>
    void ParallelTempering<_T, _stateType, _Config_t>::replicate(size_t _nSolvers)
    {
        if (_nSolvers < 1)
            throw std::invalid_argument("The number of solvers must be greater than 1.");

        for (size_t i = 1; i < _nSolvers; ++i) {
            auto _MCS = this->MCSs_[0]->clone();
            _MCS->setReplica(i);
            this->MCSs_.push_back(_MCS);        
        }
    }

    // template instantiation
    template void ParallelTempering<double, double, arma::Col<double>>::replicate(size_t);
    template void ParallelTempering<float, float, arma::Col<float>>::replicate(size_t);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::replicate(size_t);
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::replicate(size_t);
    template void ParallelTempering<std::complex<double>, double, arma::Col<double>>::replicate(size_t);

    // #################################################################################################################################

    /**
    * @brief Constructs a ParallelTempering object.
    *
    * This constructor initializes the ParallelTempering object with the given Monte Carlo solvers and beta values.
    * It sets up the thread pool, initializes the solvers and beta values, and prepares various counters and containers
    * for tracking the progress and results of the parallel tempering process.
    *
    * @tparam _T The type of the data used in the solvers.
    * @tparam _stateType The type of the state used in the solvers.
    * @tparam _Config_t The type of the configuration used in the solvers.
    * @param _MCSs A vector of shared pointers to the Monte Carlo solvers.
    * @param _betas A vector of beta values corresponding to each solver. If empty, beta values will be initialized
    *               to 1.0 / (i + 1) for each solver.
    *
    * @throws std::invalid_argument If the number of solvers is less than 1 or if the number of solvers does not match
    *                               the number of beta values provided (unless the beta vector is empty).
    */
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
            {
                this->betas_[i] = 1.0 / (double)(i + 1);
                this->MCSs_[i]->setBeta(this->betas_[i]);
            }
        }

        // Initialize the counters
        this->finished_     = std::vector<bool>(this->nSolvers_, false);
        this->errors_       = std::vector<bool>(this->nSolvers_, false);
        this->total_        = std::vector<u64>(this->nSolvers_, 0);
        this->accepted_     = std::vector<u64>(this->nSolvers_, 0);
        this->losses_       = v_1d<Container_t>(this->nSolvers_);
        this->meanLosses_   = v_1d<Container_t>(this->nSolvers_);
        this->stdLosses_    = v_1d<Container_t>(this->nSolvers_);
        this->bestLosses_.clear();
    }

    // template instantiation
    template ParallelTempering<double, double, arma::Col<double>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);
    template ParallelTempering<float, float, arma::Col<float>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);
    template ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);
    template ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);
    template ParallelTempering<std::complex<double>, double, arma::Col<double>>::ParallelTempering(const std::vector<Solver_p>&, const std::vector<double>&);

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
                                            Timer* _timer)
    {
        const size_t local_start    = 0;
        const size_t local_end      = this->nSolvers_;
        if constexpr (useMPI)
        {
#ifndef MC_ENABLE_MPI
            throw std::runtime_error("MPI is not enabled in this build.");
#else
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Determine the range of solvers for this rank
        local_start = (this->nSolvers_ / size) * rank;
        local_end   = (rank == size - 1) ? this->nSolvers_ : local_start + (this->nSolvers_ / size);
#endif
        }

        // Submit tasks to the thread pool
        for (size_t j = local_start; j < local_end; ++j) 
        {
            if (this->finished_[j]) 
                continue;

            this->threadPool_.submit([this, i, j, &_par, quiet, randomStart, &_timer]() {
                try 
                {                    
                    this->finished_[j] = this->MCSs_[j]->trainStep(
                        i, this->losses_[j], this->meanLosses_[j], this->stdLosses_[j], 
                        _par, quiet, randomStart, j == 0 ? _timer : nullptr);
                } 
                catch (const std::exception& e) 
                {
                    LOGINFO("Error in training step for solver " + std::to_string(j) + ": " + e.what(), LOG_TYPES::ERROR, 4);
                    this->total_[j]     = 0;
                    this->accepted_[j]  = 0;
                    this->lastLosses_[j]= std::numeric_limits<_T>::max();
                    this->finished_[j]  = true; // only whenever the whole solver is done
                    this->errors_[j]    = true; // only when the solver has an error
                }
            });
        }
        // Wait for all tasks to complete
        this->threadPool_.waitAll(); 

        // update counters
        for (size_t j = local_start; j < local_end; ++j)
        {
            if (this->finished_[j] || this->errors_[j])
                continue;
            this->total_[j]         = this->MCSs_[j]->getTotal();
            this->accepted_[j]      = this->MCSs_[j]->getAccepted();
            this->lastLosses_[j]    = this->MCSs_[j]->getLastLoss();
        }
    }

    // template instantiation
    // no MPI
    template void ParallelTempering<double>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    template void ParallelTempering<float>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    template void ParallelTempering<std::complex<double>>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::trainStep<false>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    // MPI
    template void ParallelTempering<double>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    template void ParallelTempering<float>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    template void ParallelTempering<std::complex<double>>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::trainStep<true>(size_t i, const MCS_train_t& _par, const bool quiet, const bool randomStart, Timer* _timer);

    // ######################################################################################################################################

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
        if (i == j || i >= this->nSolvers_                  || 
                j >= this->nSolvers_                        || 
                this->finished_[i] || this->finished_[j]    || 
                this->errors_[i] || this->errors_[j])
            return;

        // Calculate the acceptance probability
        const _T _loss_i    = this->MCSs_[i]->getLastLoss();
        const _T _loss_j    = this->MCSs_[j]->getLastLoss();
        const _T _delta     = (_loss_i - _loss_j) * ((-1) * (this->betas_[i] - this->betas_[j]));
        const _T _prob      = std::exp(_delta);
        double _absprob     = algebra::real(_prob);
        if (this->MCSs_[i]->getRandomVal() < _absprob)
        {
            std::lock_guard<std::mutex> lock(this->swapMutex_); // use the mutex to protect the swap operation
            this->MCSs_[i]->setBeta(this->betas_[j]);           // set the beta value for the swapped solver
            this->MCSs_[j]->setBeta(this->betas_[i]);           // set the beta value for the swapped solver
            // swap betas
            LOGINFO(std::format("Swapped solvers {} and {} with p={:.2f}", i, j, _absprob), LOG_TYPES::TRACE, 5);
            std::swap(this->betas_[i], this->betas_[j]);        // swap the beta values
            // swap losses
            std::swap(this->lastLosses_[i], this->lastLosses_[j]); // swap the last losses
            // swap counters
            std::swap(this->total_[i], this->total_[j]);         // swap the total counts
            std::swap(this->accepted_[i], this->accepted_[j]);   // swap the accepted counts
            // swap configurations
            // this->MCSs_[i]->swapConfig(this->MCSs_[j]);         // swap the configurations
        }
    }
    
    // template instantiation
    template void ParallelTempering<double>::swap(size_t i, size_t j);
    template void ParallelTempering<float>::swap(size_t i, size_t j);
    template void ParallelTempering<std::complex<double>>::swap(size_t i, size_t j);
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::swap(size_t i, size_t j);

    // ######################################################################################################################################
    
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
            if (this->finished_[i] || this->errors_[i])             // Skip finished solvers
            {
                ++i;
                continue;
            }

            size_t j = i + 1;
            
            // Find the next unfinished solver
            while (j < this->nSolvers_ && (this->finished_[j] || this->errors_[j]))
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
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::swaps();

    // ######################################################################################################################################

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
        std::tie(this->meanLosses_[0], this->stdLosses_[0]) = this->MCSs_[0]->train(_par, quiet, ranStart, _t, progPrc);
    }

    // template instantiation
    template void ParallelTempering<double, double, arma::Col<double>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<float, float, arma::Col<float>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, double, arma::Col<double>>::trainSingle(const MCS_train_t&, bool, bool, clk::time_point, uint);

    // ######################################################################################################################################

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
        // Initialize the progress bar and training information
        {
            if (this->pBar_)
                delete this->pBar_;
            this->pBar_ = new pBar(progPrc, _par.MC_sam_);					// set the progress bar		
            
            _par.hi();														// set the info about training
            for (size_t i = 0; i < this->nSolvers_; ++i)
                this->MCSs_[i]->reset(_par.nblck_);						    // reset the derivatives
        }

        Timer _timer;														// timer for the training
        
        // resize the losses
        this->losses_.resize(this->nSolvers_);							    // losses for each solver
        for (size_t i = 0; i < this->nSolvers_; ++i)    
            this->losses_[i].resize(_par.nblck_);			    	        // losses for each solver
        
        // resize the means
        this->meanLosses_.resize(this->nSolvers_);				            // mean losses for each solver
        for (size_t i = 0; i < this->nSolvers_; ++i)    
            this->meanLosses_[i].resize(_par.MC_sam_);	    		        // mean losses for each solver
        this->stdLosses_.resize(this->nSolvers_);					        // standard deviation of the losses for each solver
        for (size_t i = 0; i < this->nSolvers_; ++i)    
            this->stdLosses_[i].resize(_par.MC_sam_);				        // standard deviation of the losses for each solver
        
        // resize the best losses
        this->bestLosses_.clear();										    // best losses for each solver
        this->bestLosses_.resize(_par.MC_sam_);					            
        this->bestStdLosses_.clear();									    // best standard deviation of the losses for each solver
        this->bestStdLosses_.resize(_par.MC_sam_);

        // Set the random state and the number of flips
        for (size_t i = 0; i < this->nSolvers_; ++i)
        {
            this->MCSs_[i]->setRandomState();       	    			    // set the random state at the begining and the number of flips
            this->MCSs_[i]->setRandomFlipNum(_par.nFlip);				    // set the random state at the begining and the number of flips
        }
        LOGINFO("", LOG_TYPES::TRACE, 30, '#', 2);
        LOGINFO("Starting the training process.", LOG_TYPES::INFO, 2);		// inform the user about the start of the training
        // Perform the training steps
        for (size_t i = 1; i <= _par.MC_sam_; ++i)                          // go through the training steps
        {
            this->trainStep<useMPI>(i, _par, true, ranStart, &_timer);      // perform the training step

            const bool _progress    = this->pBar_ && (i % pBar_->percentageSteps == 0);

            if (_progress)
                this->swaps();                                              // perform the swaps

            // inform the user about the progress
            { 
                double _bestLoss    = std::numeric_limits<double>::max(), _bestAcc = 0.0, _bestStd = 0.0;
                size_t _bestIdx     = 0;
                size_t _bestAccIdx  = 0;
                for (size_t j = 0; j < this->nSolvers_; ++j)
                {
                    if (this->finished_[j] || this->errors_[j])
                        continue;

                    const double _currLoss = algebra::cast<double>(this->lastLosses_.at(j));

                    if (_currLoss < _bestLoss)
                    {
                        _bestLoss           = _currLoss;
                        _bestIdx            = j;
                        // update the best loss
                        this->bestLoss_     = this->lastLosses_.at(j);
                        this->bestIdx_      = j;
                    }
                    
                    if (this->total_[j] <= 0)
                        continue;

                    const double _currAcc = (double)this->accepted_[j] / this->total_[j];
                    if (_currAcc > _bestAcc)
                    {
                        _bestAcc            = _currAcc;
                        // _bestStd            = algebra::cast<double>(this->lastStdLosses_.at(j));
                        _bestAccIdx         = j;
                        // update the best acceptance
                        this->bestAcc_      = _bestAcc;
                        this->bestAccIdx_   = j;

                        // update the best losses
                        this->bestLosses_.at(i)     = _bestLoss;
                        this->bestStdLosses_.at(i)  = _bestStd;        
                    }

                    if (_progress)
                        LOGINFO(std::format("[{}] For a solver {}[b={:.2e}] the loss is: {:.3e} with acceptance {:.2f}", i, j, betas_[j], _currLoss, _currAcc), LOG_TYPES::TRACE, 4);
                }

                std::string _prog = "Iteration " + std::to_string(i) + "/" + std::to_string(_par.MC_sam_) +
                    ", Best Loss: " + std::to_string(_bestLoss) +
                    ", Best Acceptance: " + std::to_string(_bestAcc * 100) + "%" +
                    ", Best Solver Index: " + std::to_string(_bestIdx) +
                    ", Best Acceptance Solver Index: " + std::to_string(_bestAccIdx);
                PROGRESS_UPD_Q(i, (*this->pBar_), _prog, !quiet);           // update the progress bar
            }

            // ---------------------------------------------------------
            if (_progress)
            {                    
                this->MCSs_[this->bestIdx_]->saveWeights(_par.dir + "Weights" + kPS, "weights.h5");     // save the weights (if it is supported by the solver)
            }

            // ---------------------------------------------------------
            if (std::all_of(this->finished_.begin(), this->finished_.end(), [](bool f) { return f; }))
            {
                LOGINFO("All solvers have finished training.", LOG_TYPES::INFO, 1);
                // clamp the best losses and the best standard deviation of the losses
                this->bestLosses_.resize(i + 1);
                this->bestStdLosses_.resize(i + 1);
                this->MCSs_[this->bestIdx_]->saveWeights(_par.dir + "Weights" + kPS, "weights.h5");     // save the weights (if it is supported by the solver)
                break;                                                      // Exit the training loop
            }

        }

        if constexpr (useMPI) 
        {                                                                   // Finalize MPI if useMPI is true
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
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::train<false>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, double, arma::Col<double>>::train<false>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    // MPI
    template void ParallelTempering<double, double, arma::Col<double>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<float, float, arma::Col<float>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, std::complex<double>, arma::Col<std::complex<double>>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<double, std::complex<double>, arma::Col<std::complex<double>>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);
    template void ParallelTempering<std::complex<double>, double, arma::Col<double>>::train<true>(const MCS_train_t&, bool, bool, clk::time_point, uint);

    // ######################################################################################################################################
    
};

// ##########################################################################################################################################
