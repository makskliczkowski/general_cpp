#include "../src/flog.h"

/*
* @brief handles the most common exceptions
*/
void ExceptionHandler::handleExceptions(std::exception_ptr _ePtr, const std::string& _msg)
{
	try {
		if (_ePtr) std::rethrow_exception(_ePtr);
	}
	catch (const std::runtime_error& err) {
		printException(err.what(), _msg, EXCEPTIONENUM::EXCEPTIONS::RUNTIME);
	}
	catch (const std::bad_alloc& err) {
		printException(err.what(), _msg, EXCEPTIONENUM::EXCEPTIONS::BAD_ALOC);
	}
	catch (const std::exception& err) {
		printException(err.what(), _msg, EXCEPTIONENUM::EXCEPTIONS::EXCEPTION);
	}
	//catch (const std::ifstream::failure& err) {
		//printException(err.what(), _msg, EXCEPTIONS::FILE);
	//}
	catch (...) {
		printException("UNKNOWN EXCEPTION", _msg, EXCEPTIONENUM::EXCEPTIONS::BAD_ALOC);
	};
}