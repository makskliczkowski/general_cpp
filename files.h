#pragma once
#include <string>
#include <ios>
#include <thread>
#include <iomanip>
#include <iostream>
#include <algorithm> 

using clk = std::chrono::system_clock;
#define NOW std::chrono::high_resolution_clock::now()

#define EL std::endl
#define stout std::cout << std::setprecision(8) << std::fixed											// standard out
#define stouts(text, start) stout << text << " -> time : " << tim_s(start) << "s" << EL					// standard out seconds
#define stoutms(text, start) stout << text << " -> time : " << tim_ms(start) << "ms" << EL				// standard out miliseconds
#define stoutmus(text, start) stout << text << " -> time : " << tim_mus(start) << "mus" << EL			// standard out microseconds
#define stoutc(c) if(c) stout <<  std::setprecision(8) << std::fixed		                            // standard out conditional

// --- DEBUG PRINTERS ---
#ifdef DEBUG
    #define stoutd(str) do { stout << str << EL } while(0)
    #define PRT(time_point, cond) do { stoutc(cond) << #cond << " -> time : " << tim_mus(time_point) << "mus" << EL; } while (0);
#else
    #define stoutd(str) do { } while (0)
    #define PRT(time_point, cond) do { } while (0)
#endif

// ########################################################			    FILE AND STREAMS			########################################################

/*
* @brief Opens a file that is specified previously
* @param fileName name of the file to be opened
* @param mode std::ios_base::openmode of file to be opened
*/
template <typename T>
inline void openFile(T& file, std::string fileName, std::ios_base::openmode mode = std::ios::out) {
    try{
        file.open(filename, mode);
        if (!file.is_open())
            throw "couldn't open a file: " + filename + "\n";
    }
    catch(const ifstream::failure& e){
        std::cerr << "Exception opening/reading/closing file\n";
    }
}

// ########################################################				TIME FUNCTIONS				########################################################

#define DURATION(t1, t2) static_cast<long double>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration(NOW - start)).count())

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline long double t_s(clk::time_point start) {
	return DURATION(NOW, start) / 1e6;
}

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline long double t_ms(clk::time_point start) {
	return DURATION(NOW, start) / 1e3;
}

/*
* @brief The duration in seconds from a given time point
* @param point in time from which we calculate the interval
*/
inline long double t_mus(clk::time_point start) {
    return DURATION(NOW, start);
}


