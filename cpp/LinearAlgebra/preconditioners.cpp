#include "../../src/algebra/lin_alg.h"

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

            // #####################################################################################################################################

            /**
            * @brief Chooses and returns a preconditioner based on the specified type.
            * 
            * @tparam T The data type used in the preconditioner.
            * @tparam _symmetric A boolean indicating whether the preconditioner is symmetric.
            * @param i The type of the preconditioner to be chosen.
            * @return Preconditioner<T, _symmetric>* A pointer to the chosen preconditioner.
            *         Returns nullptr if the specified type is not recognized.
            */
			template <typename T, bool _symmetric>
			inline Preconditioner<T, _symmetric>* choose(Symmetric::PreconditionerType i) {
				switch (i) {
				case Symmetric::PreconditionerType::Identity:
					return new IdentityPreconditioner<T, _symmetric>;
				case Symmetric::PreconditionerType::Jacobi:
					return new JacobiPreconditioner<T, _symmetric>;
				case Symmetric::PreconditionerType::IncompleteCholesky:
					return new IncompleteCholeskyPreconditioner<T, _symmetric>;
				case Symmetric::PreconditionerType::IncompleteLU:
					return new IncompleteLUPreconditioner<T, _symmetric>;
				default:
					return nullptr;
				};
			}
            // template instantiation
            template Preconditioner<double, false>* choose<double, false>(Symmetric::PreconditionerType i);
            template Preconditioner<double, true>* choose<double, true>(Symmetric::PreconditionerType i);
            template Preconditioner<std::complex<double>, false>* choose<std::complex<double>, false>(Symmetric::PreconditionerType i);
            template Preconditioner<std::complex<double>, true>* choose<std::complex<double>, true>(Symmetric::PreconditionerType i);

            // #####################################################################################################################################

            // #####################################################################################################################################

        };
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////