#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include "../xoshiro_pp.h"
#include "../lin_alg.h"
#include "armadillo"
#include <cstddef>
#include <random>
#include <ctime>
#include <numeric>
#include <type_traits>
#include <vector>

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

namespace MonteCarlo
{
    // #################################################################################################################################

    // Monte Carlo training parameters
    struct MCS_train_t
    {
        MCS_train_t() 	= default;
        MCS_train_t(uint _mcs, uint _mcth, uint _nblck, uint _bsize, uint _nFlip, const std::string& _dir = "") 
            : MC_sam_(_mcs), MC_th_(_mcth), nblck_(_nblck), bsize_(_bsize), dir(_dir), nFlip(_nFlip) {};

        uint MC_sam_	=	10;					// number of Monte Carlo Steps (outer loops for the training or collecting)
        uint MC_th_		=	0;					// number of mcSteps to thermalize (burn-in)
        uint nblck_		= 	32;					// number of such blocks for one average step (single iteration step after which the gradient is calculated)
        uint bsize_		= 	4;					// for killing correlations - (single block size)
        std::string dir	=	"";					// saving directory (for the weights) - try to flip this many times (many flips)
        uint nFlip		= 	1;					// number of flips to set (default is 1)

        void hi(const std::string& _in = "Train: ") const;
    };


    // #################################################################################################################################

    template <typename _T, class _stateType	= double, class _Config_t = arma::Col<_stateType>>
    class MonteCarloSolver
    {
    public:
        using Config_t                      =       _Config_t;
        using Container_t                   =       arma::Col<_T>;
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
        virtual bool trainStop(size_t i, const MCS_train_t& _par, _T _currLoss, _T _currstd = 0.0, bool _quiet = false) = 0;        // check if the training should be stopped
        virtual bool trainStep(size_t i,    Container_t& En,
                                            Container_t& meanEn, 
                                            Container_t& stdEn, 
                                            const MonteCarlo::MCS_train_t& _par, 
                                            const bool quiet, 
                                            const bool randomStart,
                                            Timer& _timer) = 0;                             // perform a single training step
    public:
        // getters 
        auto getBeta()						const -> double                                 { return this->beta_; };
        auto getInfo()						const -> std::string                            { return this->info_; };
        auto getLastLoss()					const -> _T                                     { return this->lastLoss_; };
        auto getRandomVal()					const -> double;
        virtual auto getLastConfig()		const -> Config_t                               = 0; // get the last configuration
        auto getTotal()						const -> u64                                    { return this->total_; };
        auto getAccepted()					const -> u64                                    { return this->accepted_; };
        auto getRatio()						const -> double                                 { return (double)this->accepted_ / (double)this->total_; };
        // setters
        void setRandomGen(randomGen* _ran)                                                  { this->ran_ = _ran; };
        void setProgressBar(pBar* _pBar)                                                    { this->pBar_ = _pBar; };
        void setBeta(double _beta)                                                          { this->beta_ = _beta; };
        virtual void setConfig(const Config_t& _config)                                     = 0; // set the configuration
        // reset
        virtual void reset(size_t)                                                          = 0; // reset the MCS

    };

    // #################################################################################################################################

    template <typename _T>
    class ParallelTempering
    {
        using Solver_p      = std::shared_ptr<MonteCarloSolver<_T>>;
        using Container_t   = MonteCarloSolver<_T>::Container_t;
        std::mutex swapMutex_;                                                              // Protects swaps in multithreaded context
        pBar* pBar_         = nullptr;                                                      // progress bar
    protected:
        size_t nSolvers_;                                                                   // number of solvers
        std::vector<Solver_p> MCSs_;                                                        // pointers to the Monte Carlo solvers
        std::vector<_T> betas_;                                                             // inverse temperatures
        std::vector<_T> lastLosses_;                                                        // last losses
        std::vector<u64> accepted_;                                                         // number of accepted steps
        std::vector<u64> total_;                                                            // total number of steps
    public:
        ParallelTempering() = default;
        ParallelTempering(const std::vector<Solver_p>& _MCSs, const std::vector<_T>& _betas)
            : nSolvers_(_MCSs.size()), MCSs_(_MCSs), betas_(_betas), lastLosses_(_MCSs.size(), 0.0), accepted_(_MCSs.size(), 0), total_(_MCSs.size(), 0) 
        {
            if (_MCSs.size() != _betas.size())
                throw std::invalid_argument("The number of solvers and the number of betas must be the same.");
        };
        virtual ~ParallelTempering();
    
    protected:
        virtual void trainStep(size_t i,    
                                MonteCarloSolver<_T>::Container_t& En,
                                MonteCarloSolver<_T>::Container_t& meanEn, 
                                MonteCarloSolver<_T>::Container_t& stdEn, 
                                const MonteCarlo::MCS_train_t& _par, 
                                const bool quiet, 
                                const bool randomStart,
                                Timer& _timer);                                                                             // perform a single training step
        virtual void swap(size_t i, size_t j);                                                                              // swap the configurations
        virtual void swaps();                                                                                               // perform the swaps

    public:
        // !TODO implement the training and parallel solving 
        virtual void train(const MCS_train_t& _par, bool quiet = false, clk::time_point _t = NOW, uint progPrc = 25);       // train the solvers
        // virtual void collect(const MCS_train_t& _par, bool quiet = false, clk::time_point _t = NOW, uint progPrc = 25) = 0; // collect the data
    };
};

#endif