#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H
#include "../lin_alg.h"
#include "armadillo"
#include <cstddef>
#include <random>
#include <ctime>
#include <numeric>
#include <type_traits>
#include <vector>
#ifdef USE_MPI
#   define MC_ENABLE_MPI 1
#endif

// #################################################################################################################################

// ------------------------------------------------------- MONTE CARLO SAMPLING --------------------------------------------------------

namespace MonteCarlo 
{
	// #################################################################################################################################

	template <typename _T = double, typename COLTYPE = arma::Col<_T>>
	void mean(const COLTYPE& _data, _T* _mean, _T* _std = nullptr);

    template <typename _T>
    void mean(const std::vector<_T>& _data, _T* _mean, _T* _std = nullptr);
    
	// #################################################################################################################################

	template <typename _T = double, typename COLTYPE = arma::Col<_T>>
	void blockmean(const COLTYPE& _data, size_t _blockSize, _T* _mean, _T* _std = nullptr);

    template <typename _T>
    void blockmean(const std::vector<_T>& _data, size_t _blockSize, _T* _mean, _T* _std = nullptr);

	// #################################################################################################################################
};

// forward declaration
class randomGen;
class pBar;

#define MCS_PUBLIC_TYPES(_T, _ST, _CT) public:                                                  \
                    using MC_t = MonteCarlo::MonteCarloSolver<_T, _stateType, _CT<_stateType>>; \
                    using MC_t_p = std::shared_ptr<MC_t>;                                       \
                    using Container_t = MC_t::Container_t;                                      \
                    using Container_pair_t = MC_t::Container_pair_t;                            \
                    using Config_t = MC_t::Config_t;                                            \
                    using MCS_train_t = MonteCarlo::MCS_train_t;

namespace MonteCarlo
{
    // #################################################################################################################################

    /**
    * @brief Parameters for Monte Carlo Sampling training.
    * 
    * @param _mcs Number of Monte Carlo Steps (outer loops for the training or collecting).
    * @param _mcth Number of Monte Carlo Steps to thermalize (burn-in).
    * @param _nblck Number of blocks for one average step (single iteration step after which the gradient is calculated).
    * @param _bsize Block size for killing correlations (single block size).
    * @param _nFlip Number of flips to set (default is 1).
    * @param _dir Directory for saving weights (default is an empty string).
    */   
    struct MCS_train_t
    {
        MCS_train_t() = default;
        MCS_train_t(uint _mcs, uint _mcth, uint _nblck, uint _bsize, uint _nFlip, const std::string& _dir = "") 
            : MC_sam_(_mcs), MC_th_(_mcth), nblck_(_nblck), bsize_(_bsize), dir(_dir), nFlip(_nFlip) {};

        uint MC_sam_    = 10;                   // number of Monte Carlo Steps (outer loops for the training or collecting)
        uint MC_th_     = 0;                    // number of mcSteps to thermalize (burn-in)
        uint nblck_     = 32;                   // number of such blocks for one average step (single iteration step after which the gradient is calculated)
        uint bsize_     = 4;                    // for killing correlations - (single block size)
        std::string dir = "";                   // saving directory (for the weights) - try to flip this many times (many flips)
        uint nFlip      = 1;                    // number of flips to set (default is 1)

        void hi(const std::string& _in = "Train: ") const;
    };


    // #################################################################################################################################

    /**
    * @class MonteCarloSolver
    * @brief A template class for implementing Monte Carlo Solvers.
    * 
    * @tparam _T The data type for the solver.
    * @tparam _stateType The state type, default is double.
    * @tparam _Config_t The configuration type, default is arma::Col<_stateType>.
    * 
    * This class provides a framework for implementing Monte Carlo Solvers with various configurations and state types.
    * It includes methods for training, setting configurations, and managing the state of the solver.
    * 
    * @note This class is abstract and contains several pure virtual functions that need to be implemented by derived classes.
    * 
    * @var epsilon_ Machine epsilon for the data type _T.
    * @var accepted_ Number of accepted steps.
    * @var total_ Total number of steps.
    * @var lastLoss_ Last loss value, used for stopping criterion and progress bar.
    * @var beta_ Inverse temperature, default is 1.0.
    * @var info_ Information about the solver.
    * @var ran_ Pointer to a random number generator.
    * @var pBar_ Pointer to a progress bar.
    * @note !TODO Add more flexible configuration type for the state.
    */
    template <typename _T, class _stateType	= double, class _Config_t = arma::Col<_stateType>>
    class MonteCarloSolver
    {
    public:
        using Config_t                      =       _Config_t;
        using Container_t                   =       arma::Col<_T>;
        using Container_pair_t              =       std::pair<Container_t, Container_t>;
        using MC_t                          =       MonteCarloSolver<_T, _stateType, _Config_t>;
        using MC_t_p                        =       std::shared_ptr<MC_t>;
    public:
        const _T epsilon_ 					= 		std::numeric_limits<_T>::epsilon();     // machine epsilon
        u64 accepted_                       =       0;                                      // number of accepted steps
        u64 total_                          =       0;                                      // total number of steps
        _T lastLoss_                        =       0.0;                                    // last loss value (used for the stopping criterion and the progress bar)
        double beta_                        =       1.0;                                    // inverse temperature, by default is 1.0 as we are in the energy based models optimization 
    public:                                                     
        virtual ~MonteCarloSolver()         =       0;                                      // virtual destructor

    protected:                                                      
        std::string info_                   =       "Monte Carlo Solver";                   // information about the solver
	    randomGen* ran_                     =       nullptr;                                // consistent quick random number generator
        pBar* pBar_                         =       nullptr;								// for printing out the progress

        // !!! for the future use !!!                                                       
    	virtual void setInfo()			    =		0; 	                                    // set the information about the MCS (e.g., type, number of hidden units, etc.)
        virtual void init()					=       0;                                      // initialize the MCS (e.g., set the random state, etc.)
    public:                                                     
        virtual void setRandomState(bool x = true) = 0;                                     // set the random state of the MCS
        // functions that need to be implemented for other problems
        virtual bool trainStop(size_t i,    const MCS_train_t& _par, 
                                            _T _currLoss, 
                                            _T _currstd = 0.0, 
                                            bool _quiet = false) = 0;                       // check if the training should be stopped
        virtual bool trainStep(size_t i,    Container_t& En,
                                            Container_t& meanEn, 
                                            Container_t& stdEn, 
                                            const MonteCarlo::MCS_train_t& _par, 
                                            const bool quiet, 
                                            const bool randomStart,
                                            Timer& _timer)      = 0;                        // perform a single training step
        virtual Container_pair_t train(     const MCS_train_t& _par, 
                                            bool quiet          = false, 
                                            bool randomStart    = false, 
                                            clk::time_point _t  = NOW, 
                                            uint progPrc        = 25) = 0;                  // train the MCS
    public:
        virtual void setRandomFlipNum(uint _nFlip)                                           = 0; // set the number of flips
    public:
        // getters 
        auto getBeta()						const -> double                                 { return this->beta_;       };
        auto getInfo()						const -> std::string                            { return this->info_;       };
        auto getLastLoss()					const -> _T                                     { return this->lastLoss_;   };
        auto getRandomVal()					const -> double;
        virtual auto getLastConfig()		const -> Config_t                               = 0; // get the last configuration
        auto getTotal()						const -> u64                                    { return this->total_;      };
        auto getAccepted()					const -> u64                                    { return this->accepted_;   };
        auto getRatio()						const -> double                                 { return (double)this->accepted_ / (double)this->total_; };
        // setters
        void setRandomGen(randomGen* _ran)                                                  { this->ran_ = _ran; };
        void setProgressBar(pBar* _pBar)                                                    { this->pBar_ = _pBar; };
        void setBeta(double _beta)                                                          { this->beta_ = _beta; };
        // virtual
        virtual void setConfig(const Config_t& _config)                                     = 0; // set the configuration
        virtual void swapConfig(MC_t_p _other)                                              = 0; // exchange information
        // reset
        virtual void reset(size_t)                                                          = 0; // reset the MCS
        virtual auto clone()                const -> MC_t_p                                 = 0; // clone the MCS
    };

    // #################################################################################################################################

    /**
    * @class ParallelTempering
    * @brief A class that implements the Parallel Tempering algorithm for Monte Carlo simulations.
    * 
    * This class is designed to manage multiple Monte Carlo solvers running in parallel at different 
    * inverse temperatures (betas). It facilitates the swapping of configurations between solvers to 
    * enhance the exploration of the solution space, which is particularly useful in avoiding local 
    * minima in optimization problems.
    * 
    * @tparam _T The data type used by the Monte Carlo solver.
    * @tparam _stateType The state type, default is double.
    * @tparam _Config_t The configuration type, default is arma::Col<_stateType>.
    * 
    * @section Usage
    * - Initialize the class with a single solver and a set of betas, or with multiple solvers and betas.
    * - Use the train() method to start the training process.
    * - The class supports multithreading and can be extended to use MPI for distributed computing.
    * 
    * @section Members
    * - Public:
    *   - Constructors and Destructor
    *   - train() method to start the training process.
    *   - getSolvers() to retrieve the solvers.
    *   - setProgressBar() to set the progress bar.
    *   - getBetas() to retrieve the betas.
    *   - getLosses() to retrieve the losses.
    * 
    * - Protected:
    *   - replicate() to replicate the configurations.
    *   - trainStep() to perform a single training step.
    *   - trainSingle() to train a single solver.
    *   - swap() to swap configurations between solvers.
    *   - swaps() to perform multiple swaps.
    */
    template <typename _T, class _stateType	= double, class _Config_t = arma::Col<_stateType>>
    class ParallelTempering
    {
    public:
        using Solver_p      = std::shared_ptr<MonteCarloSolver<_T>>;
        using Container_t   = MonteCarloSolver<_T>::Container_t;
    private:
        
    private:
        std::mutex swapMutex_;                                                                                                        // Protects swaps in multithreaded context
        Threading::ThreadPool threadPool_;                                                                                            // Thread pool instance

        // other
        pBar* pBar_         = nullptr;                                                                                                // progress bar
    protected:
        size_t nSolvers_;                                                                                                             // number of solvers
        std::vector<Solver_p> MCSs_;                                                                                                  // pointers to the Monte Carlo solvers
        std::vector<double> betas_;                                                                                                   // inverse temperatures

        // !!! for the future use !!!
        std::vector<_T> lastLosses_;                                                                                                  // last losses
        std::vector<u64> accepted_;                                                                                                   // number of accepted steps
        std::vector<u64> total_;                                                                                                      // total number of steps
        std::vector<bool> finished_;                                                                                                  // finished solvers - when the early stopping criterion is met
        v_1d<Container_t> losses_;                                                                                                    // losses for each solver
        v_1d<Container_t> meanLosses_;                                                                                                // mean losses for each solver
        v_1d<Container_t> stdLosses_;                                                                                                 // standard deviation of the losses for each solver
    public:
        ParallelTempering() = default;
        ParallelTempering(Solver_p _MCS, const std::vector<double>& _betas, size_t _nSolvers);
        ParallelTempering(const std::vector<Solver_p>& _MCSs, const std::vector<double>& _betas);
        virtual ~ParallelTempering();
    
    protected:
        void replicate(size_t _nSolvers);                                                                                             // replicate the configurations

        // !!! FOR THE TRAINING !!!
        template <bool useMPI = false>
        void trainStep(size_t i, const MCS_train_t& _par, 
                                const bool quiet, 
                                const bool randomStart,
                                Timer& _timer);                                                                                       // perform a single training step
        void trainSingle(const MCS_train_t& _par, 
                                bool quiet, 
                                bool ranStart, 
                                clk::time_point _t, uint progPrc);                                                                    // train a single solver

        // !!! SWAP CONFIGURATIONS !!!
        virtual void swap(size_t i, size_t j);                                                                                        // swap the configurations
        virtual void swaps();                                                                                                         // perform the swaps
        
    public:
        template <bool useMPI = false>
        void train(const MCS_train_t& _par, bool quiet = false, bool ranStart = false, clk::time_point _t = NOW, uint progPrc = 25);  // train the solvers
        // virtual void collect(const MCS_train_t& _par, bool quiet = false, clk::time_point _t = NOW, uint progPrc = 25) = 0; // collect the data

        v_1d<Solver_p>& getSolvers()                                                        const { return this->MCSs_; };

        // SETTERS
        void setProgressBar(pBar* _pBar)                                                    { this->pBar_ = _pBar; };

        // GETTERS
        auto getBetas()                                                                     const -> std::vector<double>    { return this->betas_; };
        auto getLosses()                                                                    const -> v_1d<Container_t>&     { return this->losses_; };
    };

    // #################################################################################################################################
};

#endif