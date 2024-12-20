#pragma once
#ifndef UI_H
#define UI_H

#ifndef COMMON_H
#include "../common.h"
#endif

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// sets the specific option that is self-explanatory
#define SETOPTION(n, S)							this->setOption(this->n.S##_, argv, SSTR(#S))
#define SETOPTIOND(n, S, DEF)					this->setOption(this->n.S##_, argv, SSTR(#S), DEF)
#define SETOPTIONDIRECT(n, S)					this->setOption(n, S)
#define SETOPTIONV(n, S, v)						this->setOption(this->n.S##_, argv, SSTR(v)	)
#define SETOPTIONVECTOR(n, S)					std::tie(this->n.S##_ra_, this->n.S##_r_) = this->setOption(this->n.S##_, argv, SSTR(#S))
#define SETOPTIONVECTORRESIZE(n, S, D)			std::tie(this->n.S##_ra_, this->n.S##_r_) = this->setOption(this->n.S##_, argv, SSTR(#S), D, true)
#define SETOPTIONVECTORRESIZET(n, S, D, T)		std::tie(this->n.S##_ra_, this->n.S##_r_) = this->setOption<T>(this->n.S##_, argv, SSTR(#S), D, true)
// sets the option with steps etc.
#define SETOPTION_STEP(x, S)					SETOPTION(x, S);							\
												SETOPTION(x, S##0);							\
												SETOPTION(x, S##s);							\
												SETOPTION(x, S##n)
// creates both default variable (_ at front) and used variable (_ at the end) for the UI model
#define UI_PARAM_CREATE_DEFAULT(PAR, TYP, VAL)	static const	TYP _##PAR = VAL; TYP PAR##_ = VAL 
// creates both default variable (_ at front) and used variable (_ at the end) for the UI model - nonstatic doubles etc.
#define UI_PARAM_CREATE_DEFAULTD(PAR, TYP, VAL)	const			TYP _##PAR = VAL; TYP PAR##_ = VAL 
// sets default value for single parameters 
#define UI_PARAM_SET_DEFAULT(PAR)				this->PAR##_	=	this->_##PAR
#define UI_PARAM_SET_DEFAULT_STRUCT(S, PAR)		this->S.PAR##_  =	this->S._##PAR
// creates a default value for the vector type
#define UI_PARAM_CREATE_DEFAULTV(PAR, TYP)		std::vector<TYP> PAR##_; TYP PAR##_r_ = 0.0; TYP PAR##_ra_ = 0.0;

// specifies parameters in the UI that distinguish between step in range, starting point, number of steps and disorder strength
#define UI_PARAM_STEP(TYP, PAR, VAL)			UI_PARAM_CREATE_DEFAULTD(PAR, TYP, VAL);	\
												UI_PARAM_CREATE_DEFAULTD(PAR##0,TYP,0.0);	\
												UI_PARAM_CREATE_DEFAULTD(PAR##s,TYP,0.0);	\
												UI_PARAM_CREATE_DEFAULTD(PAR##n,int,1)		
// allows to set default values for parameters that precise range, disorder strength etc.
#define UI_PARAM_SET_DEFAULT_STEP(PAR)			this->PAR##_	=	this->_##PAR;			\
												this->PAR##0_	=	this->_##PAR##0;		\
												this->PAR##n_	=	this->_##PAR##n;		\
												this->PAR##s_	=	this->_##PAR##s
// adds parameters to the map
#define UI_PARAM_MAP(p, v, f)					{ #p					, std::make_tuple(#v , f)								},	\
												{ SSTR(#p ) + SSTR("0")	, std::make_tuple("0.0", FHANDLE_PARAM_HIGHERV(-1e-15))	},	\
												{ SSTR(#p ) + SSTR("s")	, std::make_tuple("0.0", FHANDLE_PARAM_DEFAULT)			},	\
												{ SSTR(#p ) + SSTR("n")	, std::make_tuple("1.0", FHANDLE_PARAM_HIGHER0)			}
// adds other variables to the map
#define UI_OTHER_MAP(p, v, f)					{ #p					, std::make_tuple(STRP(v, 2), f)						}
#define UI_VECTOR_SEPARATOR						';'
#define UI_VECTOR_RANDOM						'r'
#define UI_RANDOM_SEED							0	// if set to zero, completely random is created, otherwise integer seed is used
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
using HANDLE_FUN_TYPE = std::function<std::string(std::string)>;
// #########################################################
#define HANDLE_PARAM_DEFAULT					SSTR("")
inline	std::string		FHANDLE_PARAM_DEFAULT	(std::string s)	
{ 
	return HANDLE_PARAM_DEFAULT; 
};
// #########################################################
#define HANDLE_PARAM_HIGHERV(val)				SSTR("Value must be higher than : ") + STRP(val, 2)
inline	HANDLE_FUN_TYPE	FHANDLE_PARAM_HIGHERV	(double _low = 0.0)
{
	return [=](std::string s) { if (stod(s) < _low) return HANDLE_PARAM_HIGHERV(_low); return HANDLE_PARAM_DEFAULT; };
};
inline	std::string		FHANDLE_PARAM_HIGHER0	(std::string s)
{
	return FHANDLE_PARAM_HIGHERV()(s);
}
// #########################################################
#define HANDLE_PARAM_BETWEEN(low, high)			SSTR("Value must be between : ") + STRP(low, 2) + SSTR(" and ") + STRP(high, 2)
inline	HANDLE_FUN_TYPE	FHANDLE_PARAM_BETWEEN	(double _low = -1.0, double _high = 1.0) 
{
	return [=](std::string s) { auto v = std::stod(s); if (v > _high || v < _low) return HANDLE_PARAM_BETWEEN(_low, _high); return HANDLE_PARAM_DEFAULT; };
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class UserInterface 
{
protected:
	std::mutex mtx_;																						// mutex for the UI
protected:
	randomGen ran_;																							// random generator
	Timer _timer;
	typedef v_1d<std::string> cmdArg;
	typedef std::unordered_map<std::string, std::tuple<std::string, std::function<std::string(std::string)>>> cmdMap;

	std::string mainDir									= "." + kPS;										// main directory - to be saved onto
	uint threadNum										= 1;	
	uint threadNumIn									= 1;	
	int chosenFun										= -1;												// chosen function to be used later
	bool quiet											= false;
	
	// ------------ CHOICES and OPTIONS and DEFAULTS -----------
	cmdMap defaultParams;																					// default parameters

	// ------------------------ GETTERS ------------------------

	std::string getCmdOption(cmdArg& vec, std::string option) const;				 						// get the option from cmd input

	// ------------------------ SETTERS ------------------------
	virtual void setDefaultMap()						= 0;
	std::string setDefaultMsg(std::string v, 
							  std::string opt, 
							  std::string message, 
							  const cmdMap& map) const;														// setting value to default and sending a message
	template <typename _T>
	bool setOption(_T& value, cmdArg& argv, std::string choice);											
	template <typename _T, typename _Y>
	bool setOption(_T& valueToSet, const _Y& valueSet);
	template <class _Tin>
	std::pair<_Tin, _Tin> setOption(std::vector<_Tin>& value, cmdArg& argv, std::string choice, _Tin _default = 0.0, bool _resize = false);


	// -------------------------- INIT -------------------------
	void init(int argc, char** argv);

public:
	virtual ~UserInterface()							= default;

	// general functions to override
	virtual void exitWithHelp();

	// -------------------------- CHOICE --------------------------
	virtual void funChoice()							= 0;												// allows to choose the method without recompilation of the whole code

	// ----------------------- REAL PARSING -----------------------
	void parseOthers(cmdArg& argv);
	void parseMainDir(cmdArg& argv);

	virtual void parseModel(int argc, cmdArg& argv)		= 0;												// the function to parse the command line
	virtual cmdArg parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does
	
	// --------------------- HELPING FUNCIONS ---------------------
	virtual void setDefault()							= 0;										 		// set default parameters
	
	// ----------------------- NON-VIRTUALS -----------------------
};

// ######################################################################################################################

/*
* @brief Shows default exit with help :)
*/
inline void UserInterface::exitWithHelp()
{
	printf(
		" ------------------------------------------------- General UI parser for C++ ------------------------------------------------ \n"
		"\nThe usage of this CMD parser is straightforward:\n"
		"One uses the syntax -[option] value in order to setup values parameters available in the model.\n"
		"The parser shall skip the whitelines in the commands.\n"
		"To setup the input parameters from a file, one uses:\n"
		"options:\n"
		"-f [string] : -> input file for all of the options : (default none) \n"
		"-dir [string] : -> saving directory : (default 'current directory') \n"
		"Otherwise, the cmd line values are used. The parser allowes for more general features as well. Those include:\n"
		"\n"
		"-q	[0 or 1] : -> quiet mode (no outputs) (default false)\n"
		"\n"
		"-fun [int] : -> function to be used in the calculations. There are predefined functions in the model that allow that:\n"
		"   The options divide each other on different categories according to the first number _ \n"
		"   -1 (default option) : -> shows help \n"
		"	Otherwise, the values shall be specified by a more general class\n"
		"\n"
		"-h	: -> help\n"
		"The input for specific values also may allow parsing vectors that are separated by ';'. If the lenght of the vector mismatches\n"
		"the first value of the ';'-separated string is taken. The options for vector values include:\n"
		"   [double] -- constant value \n"
		"   'r[double]-value;[double]-disorder'	-- uniform random [disorder] around specific [value] \n"
		"   ';' separated -- vector provided by the user \n"
		" ------------------------------------------ Copyright : Maksymilian Kliczkowski, 2023 ------------------------------------------ \n"
	);
}

// ######################################################################################################################

/*
* @brief initialize the UI
* @param argc number of the arguments from the console or the initialize file
* @param argv console arguments or arguments 
*/
inline void UserInterface::init(int argc, char** argv)
{
	// initialize the timer
	_timer			= Timer();
	ran_			= randomGen(UI_RANDOM_SEED ? UI_RANDOM_SEED : std::random_device{}());
	strVec input	= fromPtr(argc, argv, 1);																// change standard input to vec of strings
	if (std::string option = this->getCmdOption(input, "-f"); option != "")
		input		= this->parseInputFile(option);															// parse input from file
	LOGINFO("Parsing input commands:", LOG_TYPES::TRACE, 1);
	LOGINFO(STRP(input, 2), LOG_TYPES::TRACE, 2);
	this->parseModel((int)input.size(), input);
};

// ######################################################################################################################

/*
* @brief sets option from a given cmd options
* @param value a value to be set onto
* @param argv arguments to find the corresponding option
* @param choice chosen option
*/
template<typename _T>
inline bool UserInterface::setOption(_T& value, cmdArg& argv, std::string choice)
{
	std::string option	=	this->getCmdOption(argv, "-" + choice);
	bool setVal			=	option != "";
	option				=	this->setDefaultMsg(option, choice, choice + ":\n", defaultParams);
	if (setVal)	setVal	=	!option.empty();
	if (setVal) value	=	static_cast<_T>(stod(option));
	return setVal;
}

// ######################################################################################################################

template<>
inline bool UserInterface::setOption<std::string>(std::string& value, cmdArg& argv, std::string choice) {
	std::string option	=	this->getCmdOption(argv, "-" + choice);
	bool setVal			=	option.empty();
	if (setVal)
		value = this->setDefaultMsg(option, std::string(choice), std::string(choice + ":\n"), defaultParams);
	else
		value			=	option;
	return !setVal;
}
// ######################################################################################################################

/*
* @brief Sets the option from a specific value given by the user
* @param valueToSet a value to be set onto
* @param valueSet a value to be set from
* @returns whether the operation has been succesful
*/
template<typename _T, typename _Y>
inline bool UserInterface::setOption(_T& valueToSet, const _Y& valueSet)
{
	BEGIN_CATCH_HANDLER
	{
		valueToSet = valueSet;
	}
	END_CATCH_HANDLER("Setting an option failed", return false;);
	return true;
}

// ######################################################################################################################

/*
* @brief Provides the possibility to input values as a vector to the UI input
* @param value vector value
* @param choice option name
* @returns the value of the option randomness
*/
template<class _Tin>
inline std::pair<_Tin, _Tin> UserInterface::setOption(std::vector<_Tin>& value, cmdArg& argv, std::string choice, _Tin _default, bool _resize)
{
	bool setVal			=	false;
	double _rVal        =	0.0;
	double _val			=	0.0;
	std::string option	=	this->getCmdOption(argv, "-" + choice);
	strVec optionVec	=	{};

	if (option != "")
	{
		BEGIN_CATCH_HANDLER
		{
			// check if the vector is of the random type
			if (setVal = option.find(UI_VECTOR_RANDOM) != std::string::npos; setVal)
			{
				optionVec			=	splitStr(option.substr(1), ";");
				_val				=	stod(optionVec[1]);
				_rVal				=	stod(optionVec[2]);
				v_1d<double> _ranV	=	ran_.rvector<v_1d<double>>(value.size(), _rVal, _val);
				for (auto i = 0; i < value.size(); ++i)
					value[i] = static_cast<_Tin>(_ranV[i]);
			}
			// check whether the value containts our special vector separating value
			else if (setVal = option.find(UI_VECTOR_SEPARATOR) != std::string::npos; setVal)
			{
				optionVec	=	splitStr(option, ";");

				// check if one should resize it!
				if (_resize)
					value.resize(optionVec.size());

				if (setVal	=	(optionVec.size() == value.size()); setVal)
					for (auto i = 0; i < value.size(); ++i)
						value[i]	=	static_cast<_Tin>(stod(optionVec[i]));
				else
					for (auto i = 0; i < value.size(); ++i)
						value[i]	=	static_cast<_Tin>(stod(optionVec[0]));
			}
			// if the value is not a vector, we set the value to the same value
			else
			{
				// check if one should resize it!
				if (_resize)
					value.resize(1);
				_val	=	static_cast<_Tin>(stod(option));
				if(setVal = !option.empty(); setVal)
					for (auto i = 0; i < value.size(); ++i)
						value[i]	=	_val;
			}
			return std::make_pair(_val, _rVal);
		}
		END_CATCH_HANDLER("Couldn't set the vector value...", ;);
	}
	std::fill(value.begin(), value.end(), _default);
	return std::make_pair(_val, _rVal);
}

// string specialization
template<>
inline std::pair<std::string, std::string> UserInterface::setOption(std::vector<std::string>& value, cmdArg& argv, std::string choice, std::string _default, bool _resize)
{
	bool setVal			=	false;
	std::string _rVal   =	"";
	std::string _val	=	"";
	std::string option	=	this->getCmdOption(argv, "-" + choice);
	strVec optionVec	=	{};

	if (option != "")
	{
		BEGIN_CATCH_HANDLER
		{
			// check whether the value containts our special vector separating value
			if (setVal = option.find(UI_VECTOR_SEPARATOR) != std::string::npos; setVal)
			{
				optionVec	=	splitStr(option, ";");

				// check if one should resize it!
				if (_resize)
					value.resize(optionVec.size());

				if (setVal	=	(optionVec.size() == value.size()); setVal)
					for (auto i = 0; i < value.size(); ++i)
						value[i]	=	optionVec[i];
				else
					for (auto i = 0; i < value.size(); ++i)
						value[i]	=	optionVec[0];
			}
			// if the value is not a vector, we set the value to the same value
			else
			{
				// check if one should resize it!
				if (_resize)
					value.resize(1);
				_val	=	option;
				if(setVal = !option.empty(); setVal)
					for (auto i = 0; i < value.size(); ++i)
						value[i]	=	_val;
			}
			return std::make_pair(_val, _rVal);
		}
		END_CATCH_HANDLER("Couldn't set the vector value...", ;);
	}
	value = { _default };
	return std::make_pair(_val, _rVal);
}

// ######################################################################################################################

/*
* @brief Parses the input for the directory
* @param argv arguments from CMD
*/
inline void UserInterface::parseMainDir(cmdArg& argv)
{
	bool setDir [[maybe_unused]] = this->setOption(this->mainDir, argv, "dir");
	if (this->mainDir.starts_with("./"))
		this->mainDir = makeDirsC(fs::current_path().string(), "DATA", this->mainDir.substr(2));
	else
		this->mainDir = makeDirsC(this->mainDir);
}

inline void UserInterface::parseOthers(cmdArg & argv)
{
	this->setOption(this->quiet, argv, "q");
	this->setOption(this->threadNum, argv, "th");
	// later function choice
	this->setOption(this->chosenFun, argv, "fun");
}


#endif // !UI_H


