#include "../src/Include/maths.h"


// #################################################################################################################################################

/**
* @brief Defines an euclidean modulo denoting also the negative sign
* @param a left side of modulo
* @param b right side of modulo
* @return euclidean a % b
* @link https://en.wikipedia.org/wiki/Modulo_operation
*/
template <typename _T = int>
_T modEUC(_T a, _T b) requires std::is_integral_v<_T>
{
    _T m = a % b;
    if (m < 0) m = (b < 0) ? m - b : m + b;
    return m;
}
// template specializations 
template int modEUC(int a, int b);
template long modEUC(long a, long b);
template long long modEUC(long long a, long long b);
template unsigned int modEUC(unsigned int a, unsigned int b);
template unsigned long modEUC(unsigned long a, unsigned long b);
template unsigned long long modEUC(unsigned long long a, unsigned long long b);

// #################################################################################################################################################

/**
* @brief Algebraic operations and functions for the library...
*/
namespace algebra
{



    // #################################################################################################################################################

};

// #################################################################################################################################################

namespace Threading
{
    // #################################################################################################################################

    /**
    * @brief Constructs a ThreadPool with a specified number of threads.
    * 
    * This constructor initializes the thread pool by creating a specified number of worker threads.
    * Each worker thread continuously waits for tasks to be added to the task queue. When a task is 
    * available, a worker thread will execute the task. The threads will continue to run and process 
    * tasks until the thread pool is stopped and the task queue is empty.
    * 
    * @param numThreads The number of worker threads to create in the thread pool.
    * @note The number of threads defaults to the number of hardware threads available on the system.
    * @note The thread pool is started immediately after construction.
    * @note The thread pool is stopped and joined when the destructor is called.
    * @note The thread pool is not copyable or movable.
    */
    ThreadPool::ThreadPool(size_t numThreads) 
    {
        for (size_t i = 0; i < numThreads; ++i) 
        {
            workers_.emplace_back([this]() 
            {
                while (true) 
                {
                    ThreadPool::Task task;
                    {
                        std::unique_lock lock(queueMutex_);
                        this->cv_.wait(lock, [this]() { return stop_ || !taskQueue_.empty(); });

                        if (stop_ && taskQueue_.empty())
                            return;

                        task = std::move(taskQueue_.front());
                        this->taskQueue_.pop();
                        ++this->activeTasks_;
                    }
                    task();

                    // Artificial waiting
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));

                    // Notify that the task has been completed
                    {
                        std::lock_guard lock(this->queueMutex_);
                        --this->activeTasks_;
                    }
                    this->waitCv_.notify_one();
                }
            });
        }
    }

    // #################################################################################################################################

    ThreadPool::~ThreadPool() 
    {
        this->shutdown();
    }

    // #################################################################################################################################

    /**
    * @brief Submits a task to the thread pool for execution.
    *
    * This function adds a task to the task queue and notifies one of the worker threads
    * to start processing the task. The task is moved into the queue to avoid unnecessary
    * copying.
    *
    * @param task The task to be executed, encapsulated in a Task object.
    */
    void ThreadPool::submit(Task task) 
    {
        {
            std::lock_guard lock(this->queueMutex_);
            this->taskQueue_.push(std::move(task));
        }
        this->cv_.notify_one();
    }

    // #################################################################################################################################

    /**
    * @brief Shuts down the thread pool.
    *
    * This function stops the thread pool by setting the stop flag to true and 
    * notifying all worker threads. It then joins all worker threads to ensure 
    * they have completed execution before returning.
    */
    void ThreadPool::shutdown() 
    {
        {
            std::lock_guard lock(this->queueMutex_);
            this->stop_ = true;
        }
        this->cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // #################################################################################################################################

    /**
    * @brief Waits for all tasks in the thread pool to complete.
    *
    * This function blocks the calling thread until all tasks in the thread pool
    * have finished executing. It acquires a unique lock on the queue mutex and
    * waits on a condition variable until the task queue is empty and there are
    * no active tasks.
    */
    void ThreadPool::waitAll() 
    {
        std::unique_lock lock(this->queueMutex_);
        waitCv_.wait(lock, [this]() 
        {
            return this->taskQueue_.empty() && this->activeTasks_ == 0;
        });
    }

    // #################################################################################################################################

};

// #####################################################################################################################################