#include "../../src/lin_alg.h"

// #################################################################################################################################################

// PRECONDITIONERS FOR THE SOLVERS

// #################################################################################################################################################

namespace algebra
{
    namespace Solvers
    {
        namespace Preconditioners
        {
            // #####################################################################################################################################
            
            // define the template specializations 
            // double, false
            template class Preconditioner<double, false>;
            // double, true
            template class Preconditioner<double, true>;
            // complex, false
            template class Preconditioner<std::complex<double>, false>;
            // complex, true
            template class Preconditioner<std::complex<double>, true>;
        };
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////