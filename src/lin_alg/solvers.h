#pragma once

	// ########################################################### MATRIX EQUATIONS SOLVERS ############################################################

	// #################################################################################################################################################

	namespace Solvers
	{	
		constexpr double TINY = 1.0e-16;		// a small number

		// #################################################################################################################################################

		template <typename _T1>
		void sym_ortho(_T1 a, _T1 b, _T1& c, _T1& s, _T1& r);

		// #################################################################################################################################################

		namespace Preconditioners {

			/**
			* @brief Preconditioner interface for any method that can be used as a preconditioner for the conjugate gradient method.
			*/
			template<typename T, bool _isPositiveSemidefinite = false>
			class Preconditioner 
			{
			public:
				const bool isPositiveSemidefinite_ 	= _isPositiveSemidefinite;	// is the matrix positive semidefinite
				bool isGram_ 						= false;					// is the matrix a Gram matrix
				double sigma_ 						= 0.0;						// regularization parameter
				int type_ 							= 0;						// type of the preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
			public:
				virtual ~Preconditioner() = default;
				Preconditioner() 
					: isGram_(false)
				{};
				Preconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: isGram_(isGram)
				{
					// this->set(A, isGram, _sigma);
				}
				Preconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: isGram_(true), sigma_(_sigma)
				{
					// this->set(Sp, S, _sigma);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				// Copy constructor
				Preconditioner(const Preconditioner& other)
					: isPositiveSemidefinite_(other.isPositiveSemidefinite_),
					isGram_(other.isGram_),
					sigma_(other.sigma_),
					type_(other.type_) {}

				// Move constructor
				Preconditioner(Preconditioner&& other) noexcept
					: isPositiveSemidefinite_(other.isPositiveSemidefinite_),
					isGram_(std::exchange(other.isGram_, false)),
					sigma_(std::exchange(other.sigma_, 0.0)),
					type_(std::exchange(other.type_, 0)) {}

				// Copy assignment operator
				Preconditioner& operator=(const Preconditioner& other)
				{
					if (this != &other) {
						isGram_ = other.isGram_;
						sigma_ = other.sigma_;
						type_ = other.type_;
					}
					return *this;
				}

				// Move assignment operator
				Preconditioner& operator=(Preconditioner&& other) noexcept
				{
					if (this != &other) {
						isGram_ = std::exchange(other.isGram_, false);
						sigma_ = std::exchange(other.sigma_, 0.0);
						type_ = std::exchange(other.type_, 0);
					}
					return *this;
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
				virtual std::unique_ptr<Preconditioner> clone() const 	= 0;
				virtual std::unique_ptr<Preconditioner> move() 			= 0;
				virtual std::shared_ptr<Preconditioner> shared() 		= 0;
				// -----------------------------------------------------------------------------------------------------------------------------------------

				// set the preconditioner
				void set(bool _isGram, double _sigma = 0.0) { this->isGram_ = _isGram; this->sigma_ = _sigma; }
				virtual void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) = 0;		// set the preconditioner
				virtual void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) = 0;	// set the preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				int type() const { return this->type_; }													// get the type of the preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				// apply the preconditioner
				virtual arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const = 0;			// general matrix preconditioner

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				// operator overloading
				arma::Col<T> operator()(const arma::Col<T>& r, double sigma = 0.0) const { return this->apply(r, sigma); } 
			};

			// #################################################################################################################################################

			/**
			* @brief Identity preconditioner for the conjugate gradient method.
			* The identity preconditioner does not change the input vector.
			* @tparam T The type of the matrix elements.
			*/
			template <typename T, bool _F = false>
			class IdentityPreconditioner : public Preconditioner<T, _F> 
			{
			public:
				IdentityPreconditioner()
					: Preconditioner<T, _F>()
					{
						this->type_ = 0;
					};
				IdentityPreconditioner(const arma::Mat<T>& A, bool _isGram, double _sigma = 0.0)
					: Preconditioner<T, _F>(A, _isGram, _sigma)
				{
					this->type_ = 0;
				}

				IdentityPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _F>(Sp, S, _sigma)
				{
					this->type_ = 0;
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				// set the preconditioner
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _F>::set(isGram, _sigma);
					// do nothing
				}

				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _F>::set(true, _sigma);
					// do nothing
				}

				// apply the preconditioner
				arma::Col<T> apply(const arma::Col<T>& r, double sigma) const override
				{
					return r;
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				std::unique_ptr<Preconditioner<T, _F>> clone() const override
				{
					return std::make_unique<IdentityPreconditioner<T, _F>>(*this);
				}

				std::unique_ptr<Preconditioner<T, _F>> move() override
				{
					return std::make_unique<IdentityPreconditioner<T, _F>>(std::move(*this));
				}

				std::shared_ptr<Preconditioner<T, _F>> shared() override
				{
					return std::make_shared<IdentityPreconditioner<T, _F>>(*this);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
			};	
			
			// #################################################################################################################################################

			/**
			* @brief Jacobi preconditioner for the conjugate gradient method. This preconditioner is used for symmetric positive definite matrices. 
			* The Jacobi preconditioner is a diagonal matrix with the diagonal elements of the original matrix on the diagonal. 
			* The inverse of the diagonal elements is used as the preconditioner.
			* @tparam T The type of the matrix elements.
			*/
			template <typename T, bool _T = true>
			class JacobiPreconditioner : public Preconditioner<T, _T> {
			private:
				arma::Col<T> diaginv_;			// diagonal inverse of the matrix 
				double tolBig_ 		= 1.0e10;	// if the value is bigger than this, then 1/value is small and we create cut-off
				T  bigVal_			= 1e-10;	// treated as zero for 1/value
				double tolSmall_ 	= 1.0e-10; 	// if the value is smaller than this, then 1/value is big and we create cut-off
				T  smallVal_		= 1e10;		// treated as zero for 1/value
			public:

				// -----------------------------------------------------------------------------------------------------------------------------------------
				JacobiPreconditioner() 
					: Preconditioner<T, _T>()
				{
					this->type_ = 1;
				};

				// Copy constructor
				JacobiPreconditioner(const JacobiPreconditioner& other)
					: Preconditioner<T, _T>(other)
				{
					this->diaginv_ 	= other.diaginv_;
					this->tolBig_ 	= other.tolBig_;
					this->bigVal_ 	= other.bigVal_;
					this->tolSmall_ = other.tolSmall_;
					this->smallVal_ = other.smallVal_;
				}

				// Move constructor
				JacobiPreconditioner(JacobiPreconditioner&& other) noexcept
					: Preconditioner<T, _T>(std::move(other))
				{
					this->diaginv_ 	= std::move(other.diaginv_);
					this->tolBig_ 	= std::exchange(other.tolBig_, 1.0e10);
					this->bigVal_ 	= std::exchange(other.bigVal_, 1e-10);
					this->tolSmall_ = std::exchange(other.tolSmall_, 1.0e-10);
					this->smallVal_ = std::exchange(other.smallVal_, 1e10);
				}		

				// -----------------------------------------------------------------------------------------------------------------------------------------				

				// is any matrix A, not necessarily a Gram matrix. Otherwise, use isGram = true and A = S+ * S
				JacobiPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{
					this->type_ = 1;
				}

				JacobiPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _T>(Sp, S, _sigma)
				{
					this->type_ = 1;
				}
				// set the preconditioner
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(isGram, _sigma);

					if (!isGram)
					{
						arma::Col<T> diag 	= arma::diagvec(A);
						if (_sigma > 0.0)
							diag += (T)this->sigma_;
						this->diaginv_ 		= 1.0 / diag;
						this->diaginv_ 		= arma::clamp(this->diaginv_, -1.0e10, 1.0e10);
					}
					else
						this->set(A, A, _sigma); // setting A, as Aplus is not needed
				}

				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(true, _sigma < 0 ? 0 : _sigma);

					this->diaginv_.set_size(S.n_cols);
					for (size_t i = 0; i < diaginv_.n_elem; ++i) 
					{
						const T norm_val 		= arma::norm(S.col(i)) + this->sigma_;
						const double norm_abs 	= std::abs(norm_val);

						this->diaginv_(i) = (norm_abs > tolBig_) 	? bigVal_ 	:
											(norm_abs < tolSmall_) 	? smallVal_ :
											(1.0 / norm_val);
					}
				}

				// apply the preconditioner
				arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const override { return this->diaginv_ % r; }

				// -----------------------------------------------------------------------------------------------------------------------------------------

				std::unique_ptr<Preconditioner<T, _T>> clone() const override
				{
					return std::make_unique<JacobiPreconditioner<T, _T>>(*this);
				}

				std::unique_ptr<Preconditioner<T, _T>> move() override
				{
					return std::make_unique<JacobiPreconditioner<T, _T>>(std::move(*this));
				}

				std::shared_ptr<Preconditioner<T, _T>> shared() override
				{
					return std::make_shared<JacobiPreconditioner<T, _T>>(*this);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
			};

			// #################################################################################################################################################
			
			
			/**
			* @brief Incomplete Cholesky preconditioner for the conjugate gradient method. This preconditioner is used for symmetric positive definite matrices.
			* @tparam T The type of the matrix elements.
			*/
			template <typename T, bool _T = true>
			class IncompleteCholeskyPreconditioner : public Preconditioner<T, _T> {

			private:
				arma::Mat<T> L_;     // lower triangular incomplete Cholesky factor
				bool success_ 	= false;
			public:
				IncompleteCholeskyPreconditioner()
					: Preconditioner<T, _T>()
				{
					this->type_ = 2;
				}

				// Copy constructor
				IncompleteCholeskyPreconditioner(const IncompleteCholeskyPreconditioner& other)
					: Preconditioner<T, _T>(other)
				{
					this->L_ 		= other.L_;
					this->success_ = other.success_;
				}

				// Move constructor
				IncompleteCholeskyPreconditioner(IncompleteCholeskyPreconditioner&& other) noexcept
					: Preconditioner<T, _T>(std::move(other))
				{
					this->L_ 		= std::move(other.L_);
					this->success_ = std::exchange(other.success_, false);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				/**
				* @brief Constructor to initialize the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				*/
				IncompleteCholeskyPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{
					this->type_ = 2;
				}

				IncompleteCholeskyPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _T>(Sp, S, _sigma)
				{
					this->type_ = 2;
				}
				// -----------------------------------------------------------------------------------------------------------------------------------------

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(isGram, _sigma);

					if (!isGram) {
						// Directly calculate incomplete Cholesky factor L
						this->success_ = arma::chol(L_, A + arma::Mat<T>(A.n_cols, A.n_cols, arma::fill::eye)  * this->sigma_, "lower");
						if (!success_) {
							std::cerr << "Incomplete Cholesky decomposition failed.\n";
							L_.reset(); // Clear L_ if decomposition fails
						}
					} else 
						this->set(A.t(), A, _sigma);
				}

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param Sp The matrix to decompose.
				* @param S The matrix to decompose.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(true, _sigma);

					arma::Mat<T> A 	= Sp * S;
					A.diag() 		+= this->sigma_;
					
					if (this->success_ 	= arma::chol(L_, A, "lower"); !this->success_) {
						std::cerr << "Incomplete Cholesky decomposition failed.\n";
						L_.reset(); // Clear L_ if decomposition fails
					}
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				/**
				* @brief Apply the preconditioner to a given vector.
				* @param r The vector to precondition.
				* @param sigma Regularization parameter (default is 0.0).
				* @return The preconditioned vector.
				*/
				arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const override
				{
					if (this->success_) {
						
						arma::Col<T> y;
						
						// Forward solve L*y = r
						try {
							y = arma::solve(arma::trimatl(L_), r);
						} catch (const std::runtime_error& e) {
							std::cerr << "Forward solve failed: " << e.what() << "\n";
							return r; // If forward solve fails, return r as is
						}

						// Backward solve L^T*z = y
						try {
							return arma::solve(arma::trimatu(L_.t()), y);
						} catch (const std::runtime_error& e) {
							std::cerr << "Backward solve failed: " << e.what() << "\n";
							return r; // If backward solve fails, return r as is
						}
					} else
						return r; // If decomposition failed, return r as is
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				std::unique_ptr<Preconditioner<T, _T>> clone() const override
				{
					return std::make_unique<IncompleteCholeskyPreconditioner<T, _T>>(*this);
				}

				std::unique_ptr<Preconditioner<T, _T>> move() override
				{
					return std::make_unique<IncompleteCholeskyPreconditioner<T, _T>>(std::move(*this));
				}

				std::shared_ptr<Preconditioner<T, _T>> shared() override
				{
					return std::make_shared<IncompleteCholeskyPreconditioner<T, _T>>(*this);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
			};

			// #################################################################################################################################################

			/**
			* @brief Binormalization preconditioner for the conjugate gradient method. This preconditioner is used for symmetric positive definite matrices.
			* Scale the matrix with a series of k diagonal matrices D1, D2, ..., Dk -> DAD = D_k ... D_2 D_1 A D_1 D_2 ... D_k
			*/
			template <typename T, bool _T = true>
			class BinormalizationPreconditioner : public Preconditioner<T, _T> {
			private:
				bool success_ 	= false;

			public:
				BinormalizationPreconditioner()
					: Preconditioner<T, _T>()
				{
					this->type_ = 3;
				}				
				
				/**
				* @brief Constructor to initialize the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				*/
				BinormalizationPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{
					this->type_ = 3;
				}
				BinormalizationPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _T>(Sp, S, _sigma)
				{
					this->type_ = 3;
				}
				// -----------------------------------------------------------------------------------------------------------------------------------------

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					// !TODO
				}

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param Sp The matrix to decompose.
				* @param S The matrix to decompose.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the
				* diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					// !TODO
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				std::unique_ptr<Preconditioner<T, _T>> clone() const override
				{
					return std::make_unique<BinormalizationPreconditioner<T, _T>>(*this);
				}

				std::unique_ptr<Preconditioner<T, _T>> move() override
				{
					return std::make_unique<BinormalizationPreconditioner<T, _T>>(std::move(*this));
				}

				std::shared_ptr<Preconditioner<T, _T>> shared() override
				{
					return std::make_shared<BinormalizationPreconditioner<T, _T>>(*this);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
			};

			// #################################################################################################################################################

			template <typename T, bool _T = true>
			class IncompleteLUPreconditioner : public Preconditioner<T, _T> {
			private:
				arma::Mat<T> L_;     // lower triangular incomplete Cholesky factor
				arma::Mat<T> U_;     // upper triangular incomplete Cholesky factor
				arma::Col<arma::uword> P_; // permutation vector
				bool success_ 	= false;
				int type_ 		= 3;	// type of the preconditioner
			public:
				IncompleteLUPreconditioner()
					: Preconditioner<T, _T>()
				{};

				// Copy constructor
				IncompleteLUPreconditioner(const IncompleteLUPreconditioner& other)
					: Preconditioner<T, _T>(other)
				{
					this->L_ 		= other.L_;
					this->U_ 		= other.U_;
					this->P_ 		= other.P_;
					this->success_ = other.success_;
				}

				// Move constructor
				IncompleteLUPreconditioner(IncompleteLUPreconditioner&& other) noexcept
					: Preconditioner<T, _T>(std::move(other))
				{
					this->L_ 		= std::move(other.L_);
					this->U_ 		= std::move(other.U_);
					this->P_ 		= std::move(other.P_);
					this->success_ = std::exchange(other.success_, false);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				/**
				* @brief Constructor to initialize the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				*/
				IncompleteLUPreconditioner(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0)
					: Preconditioner<T, _T>(A, isGram, _sigma)
				{}

				IncompleteLUPreconditioner(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0)
					: Preconditioner<T, _T>(Sp, S, _sigma)
				{}

				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				/**
				* @brief Set the preconditioner with a given matrix.
				* @param A The matrix to decompose.
				* @param isGram Flag indicating if the matrix is a Gram matrix.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& A, bool isGram = true, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(isGram, _sigma);

					// !TODO
				}

				/**
				* @brief Set the preconditioner with a given matrix.
				* @param Sp The matrix to decompose.
				* @param S The matrix to decompose.
				* @param _sigma Regularization parameter (default is 0.0). This is added to the diagonal of the matrix before decomposition.
				*/
				void set(const arma::Mat<T>& Sp, const arma::Mat<T>& S, double _sigma = 0.0) override
				{
					Preconditioner<T, _T>::set(true, _sigma);

					// !TODO
				}

				/**
				* @brief Apply the preconditioner to a given vector.
				* @param r The vector to precondition.
				* @param sigma Regularization parameter (default is 0.0).
				* @return The preconditioned vector.
				*/
				arma::Col<T> apply(const arma::Col<T>& r, double sigma = 0.0) const override
				{
					// !TODO
					return r;
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------

				std::unique_ptr<Preconditioner<T, _T>> clone() const override
				{
					return std::make_unique<IncompleteLUPreconditioner<T, _T>>(*this);
				}

				std::unique_ptr<Preconditioner<T, _T>> move() override
				{
					return std::make_unique<IncompleteLUPreconditioner<T, _T>>(std::move(*this));
				}

				std::shared_ptr<Preconditioner<T, _T>> shared() override
				{
					return std::make_shared<IncompleteLUPreconditioner<T, _T>>(*this);
				}

				// -----------------------------------------------------------------------------------------------------------------------------------------
			};
			

			// #################################################################################################################################################
			namespace Symmetric {
				enum class PreconditionerType {
					Identity,
					Jacobi,
					IncompleteCholesky,
					IncompleteLU
				}; 
			};
			namespace NonSymmetric {
				enum class PreconditionerType {
					Identity
				}; 
			};		

			// -----------------------------------------------------------------------------------------------------------------------------------------

			template <typename T, bool _symmetric = true>
			Preconditioner<T, _symmetric>* choose(Symmetric::PreconditionerType i);

			template <typename T, bool _Sym = true>
			inline Preconditioner<T, _Sym>* choose(int i = 0) { return choose<T, _Sym>(static_cast<Symmetric::PreconditionerType>(i)); };
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

            /**
            * @brief Returns the name of the given preconditioner type.
            *
            * @param i The preconditioner type.
            * @return A string representing the name of the preconditioner type.
            *         Possible return values are:
            *         - "Identity" for Symmetric::PreconditionerType::Identity
            *         - "Jacobi" for Symmetric::PreconditionerType::Jacobi
            *         - "Incomplete Cholesky" for Symmetric::PreconditionerType::IncompleteCholesky
            *         - "Incomplete LU" for Symmetric::PreconditionerType::IncompleteLU
            *         - "None" for any other value
            */
            inline std::string name(Symmetric::PreconditionerType i) {
				switch (i) {
				case Symmetric::PreconditionerType::Identity:
					return "Identity";
				case Symmetric::PreconditionerType::Jacobi:
					return "Jacobi";
				case Symmetric::PreconditionerType::IncompleteCholesky:
					return "Incomplete Cholesky";
				case Symmetric::PreconditionerType::IncompleteLU:
					return "Incomplete LU";
				default:
					return "None";
				};
			}
			inline std::string name(int i = 0) { return name(static_cast<Symmetric::PreconditionerType>(i)); };
			
			// -----------------------------------------------------------------------------------------------------------------------------------------
			
		};

		// #################################################################################################################################################
		template<typename _T>
		using _AX_fun = std::function<arma::Col<_T>(const arma::Col<_T>&, double)>;	// matrix-vector multiplication function
		template<typename _T, bool _T1 = true>
		using Precond = Preconditioners::Preconditioner<_T, _T1>;					// preconditioner type
		// #################################################################################################################################################
		#define _MATFREE_MULT(_T) _AX_fun(_T) _matFreeMul							// matrix-free multiplication function
		#define SOLVE_GENERAL_ARG_TYPES(_T1) 		const arma::Col<_T1>& _F,									\
													arma::Col<_T1>* _x0,										\
													double _eps,												\
													size_t _max_iter,											\
													bool* _converged, 											\
													double _reg												
		#define SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)const arma::Col<_T1>& _F,						\
																arma::Col<_T1>* _x0,							\
																Solvers::Precond<_T1, _T2>* _preconditioner,	\
																double _eps,									\
																size_t _max_iter,								\
																bool* _converged, 								\
																double _reg			
		// with default values			
		#define SOLVE_GENERAL_ARG_TYPESD(_T1) 		const arma::Col<_T1>& _F,									\
													arma::Col<_T1>* _x0 	= nullptr,							\
													double _eps				= 1e-10,							\
													size_t _max_iter		= 100,								\
													bool* _converged		= nullptr, 							\
													double _reg				= -1.0				
		#define SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2) const arma::Col<_T1>& _F,						\
																arma::Col<_T1>* _x0,							\
																Solvers::Precond<_T1, _T2>* _preconditioner = nullptr, \
																double _eps							= 1e-10,	\
																size_t _max_iter					= 100,		\
																bool* _converged					= nullptr,	\
																double _reg							= -1.0
		// with matrix multiplication function
		#define SOLVE_MATMUL_ARG_TYPES(_T1) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPES(_T1)
		#define SOLVE_MATMUL_ARG_TYPES_PRECONDITIONER(_T1, _T2) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)
		#define SOLVE_MATMUL_ARG_TYPESD(_T1) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPESD(_T1)
		#define SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, _T2) Solvers::_AX_fun<_T1> _matrixFreeMultiplication, SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2)

		// #################################################################################################################################################
		namespace General 
		{
			// #################################################################################################################################################
			enum class Type {
				ARMA				= 0,				// Armadillo solver
				PseudoInverse		= 4,				// Pseudo Inverse - minimum norm solution
				Direct				= 5,				// Direct solver - may not be s
				// SYMMETRIC
				ConjugateGradient	= 1,				// Conjugate Gradient Method
				MINRES				= 2,				// Minimum Residual Method
				MINRES_QLP			= 3					// Minimum Residual Method with QLP
			};
			// #############################################################################################################################################
						
			template <typename _T, bool _symmetric = true>
			class Solver 
			{
			protected:
				Type type_ 				= Type::Direct;	// type of the solver
				bool isSymmetric_ 		= _symmetric;	// is the matrix symmetric
				bool converged_ 		= false;		// has the method converged
				bool isGram_			= false;		// is the matrix a Gram matrix
				size_t N_				= 1;			// size of the matrix
				size_t iter_ 			= 0; 			// current iteration - [[maybe_unused]]
				size_t max_iter_ 		= 1000;			// maximum number of iterations
				double eps_ 			= 1e-10;		// convergence criterion
				double reg_ 			= -1.0;			// regularization parameter (if needed)
				Precond<_T, _symmetric>* precond_;		// preconditioner (if exists) - this is used to solve the system M^{-1}Ax = M^{-1}b
				bool isPreconditioned_ 	= false;		// is the matrix preconditioned (reffers to the preconditioner_ field)
				_AX_fun<_T> matVecFun_;					// matrix-vector multiplication function such that Ax = b (if exists)
				arma::Col<_T> x_;						// solution vector

			public:
				// -----------------------------------------------------------------------------------------------------------------------------------------
				virtual ~Solver()		= default;
				Solver() 				= default;
				Solver(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T, _symmetric>* _preconditioner = nullptr);
				// Copy constructor
				Solver(const Solver& other)
					: type_(other.type_), isSymmetric_(other.isSymmetric_), converged_(other.converged_), isGram_(other.isGram_), N_(other.N_), iter_(other.iter_), max_iter_(other.max_iter_), eps_(other.eps_), reg_(other.reg_), precond_(other.precond_), isPreconditioned_(other.isPreconditioned_), matVecFun_(other.matVecFun_), x_(other.x_) {}
				// Move constructor
				Solver(Solver&& other) noexcept
					: type_(std::exchange(other.type_, Type::Direct)), isSymmetric_(std::exchange(other.isSymmetric_, _symmetric)), converged_(std::exchange(other.converged_, false)), isGram_(std::exchange(other.isGram_, false)), N_(std::exchange(other.N_, 1)), iter_(std::exchange(other.iter_, 0)), max_iter_(std::exchange(other.max_iter_, 1000)), eps_(std::exchange(other.eps_, 1e-10)), reg_(std::exchange(other.reg_, -1.0)), precond_(std::exchange(other.precond_, nullptr)), isPreconditioned_(std::exchange(other.isPreconditioned_, false)), matVecFun_(std::exchange(other.matVecFun_, nullptr)), x_(std::move(other.x_)) {}
				// Copy assignment
				Solver& operator=(const Solver& other)
				{
					if (this != &other)
					{
						type_ 			= other.type_;
						isSymmetric_ 	= other.isSymmetric_;
						converged_ 		= other.converged_;
						isGram_ 		= other.isGram_;
						N_ 				= other.N_;
						iter_ 			= other.iter_;
						max_iter_ 		= other.max_iter_;
						eps_ 			= other.eps_;
						reg_ 			= other.reg_;
						precond_ 		= other.precond_;
						isPreconditioned_ = other.isPreconditioned_;
						matVecFun_ 		= other.matVecFun_;
						x_ 				= other.x_;
					}
					return *this;
				}
				// Move assignment
				Solver& operator=(Solver&& other) noexcept
				{
					if (this != &other)
					{
						type_ 			= std::exchange(other.type_, Type::Direct);
						isSymmetric_ 	= std::exchange(other.isSymmetric_, _symmetric);
						converged_ 		= std::exchange(other.converged_, false);
						isGram_ 		= std::exchange(other.isGram_, false);
						N_ 				= std::exchange(other.N_, 1);
						iter_ 			= std::exchange(other.iter_, 0);
						max_iter_ 		= std::exchange(other.max_iter_, 1000);
						eps_ 			= std::exchange(other.eps_, 1e-10);
						reg_ 			= std::exchange(other.reg_, -1.0);
						precond_ 		= std::exchange(other.precond_, nullptr);
						isPreconditioned_ = std::exchange(other.isPreconditioned_, false);
						matVecFun_ 		= std::exchange(other.matVecFun_, nullptr);
						x_ 				= std::move(other.x_);
					}
					return *this;
				}
				// -----------------------------------------------------------------------------------------------------------------------------------------
				
				virtual void init(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(_AX_fun<_T> _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr);
				virtual void init(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr) = 0;

				// -----------------------------------------------------------------------------------------------------------------------------------------
				// getters
				inline const arma::Col<_T>& solution() 					const { return this->x_; };			
				inline arma::Col<_T>&& moveSolution() 					{ return std::move(this->x_); }
				inline _T solution(size_t _i) 							const { return this->x_(_i); }
				inline size_t getN() 									const { return this->N_; }
				inline size_t getIter() 								const { return this->iter_; }
				inline size_t getMaxIter() 								const { return this->max_iter_; }
				inline double getEps() 									const { return this->eps_; }
				inline double getReg() 									const { return this->reg_; }
				inline bool isConverged() 								const { return this->converged_; }
				inline bool isPreconditioned() 							const { return this->isPreconditioned_; }
				inline Precond<_T, _symmetric>* getPreconditioner() 	const { return this->precond_; }
				// -----------------------------------------------------------------------------------------------------------------------------------------
				// setters
				inline void setMaxIter(size_t _max_iter) 				{ this->max_iter_ = _max_iter; }
				inline void setEps(double _eps) 						{ this->eps_ = _eps; }
				inline void setReg(double _reg) 						{ this->reg_ = _reg; }
				inline void setPreconditioner(Precond<_T, _symmetric>* _precond) { this->precond_ = _precond; isPreconditioned_ = (_precond != nullptr); }
				// -----------------------------------------------------------------------------------------------------------------------------------------

				virtual void solve(const arma::Mat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);							// if we want to use a dense matrix
				virtual void solve(const arma::SpMat<_T>& _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);							// if we want to use a sparse matrix
				virtual void solve(const arma::Mat<_T>& _S, const arma::Mat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);	// if we want to use a Fisher matrix
				virtual void solve(const arma::SpMat<_T>& _S, const arma::SpMat<_T>& _Sp, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);	// if we want to use a Fisher matrix
				virtual void solve(_AX_fun<_T> _A, const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr);										// if we want to use a matrix-vector multiplication function
				virtual void solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr) = 0;													// if the matrix multiplication function is set
				// -----------------------------------------------------------------------------------------------------------------------------------------
				virtual std::unique_ptr<Solver<_T, _symmetric>> clone() const 	= 0;
				virtual std::unique_ptr<Solver<_T, _symmetric>> move() 			= 0;
				virtual std::shared_ptr<Solver<_T, _symmetric>> shared() 		= 0;
				// -----------------------------------------------------------------------------------------------------------------------------------------
			};	
			// #############################################################################################################################################
		};
		// #################################################################################################################################################
		namespace General 
		{ 
			// #################################################################################################################################################
			// with a signle matrix
			#define SOLVE_MAT_ARG_TYPES(_T1) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES(_T1)
			#define SOLVE_MAT_ARG_TYPES_PRECONDITIONER(_T1, _T2) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)
			#define SOLVE_MAT_ARG_TYPESD(_T1) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD(_T1)
			#define SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, _T2) const arma::Mat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2)
			// with sparse matrix
			#define SOLVE_SPMAT_ARG_TYPES(_T1) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES(_T1)
			#define SOLVE_SPMAT_ARG_TYPES_PRECONDITIONER(_T1, _T2) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, _T2)
			#define SOLVE_SPMAT_ARG_TYPESD(_T1) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD(_T1)
			#define SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, _T2) const arma::SpMat<_T1>& _A, SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, _T2)
			// #################################################################################################################################################	
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _A, const arma::Col<_T>& _x, const double _reg = 0.0);
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::SpMat<_T>& _A, const arma::Col<_T>& _x, const double _reg = 0.0);
			// #################################################################################################################################################
			#define MAKE_MATRIX_FREE_MULT(_T) auto _f = [&](const arma::Col<_T>& _x, double _reg) -> arma::Col<_T> { return matrixFreeMultiplication<_T>(_A, _x, _reg); };
			// #################################################################################################################################################
			namespace CG 
			{
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MATMUL_ARG_TYPESD(_T1));
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, true));
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_SPMAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg); } 
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return conjugate_gradient<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
			
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class ConjugateGradient_s : virtual public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Col<_T1> r, p, Ap;
					// for preconditioned only
					arma::Col<_T1> z;
					_T1 rs_old;
				public:
					ConjugateGradient_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::ConjugateGradient;
						if (!_symmetric) 
							throw std::invalid_argument("Conjugate Gradient method is only for symmetric matrices.");
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					std::unique_ptr<Solver<_T1, _symmetric>> clone() const override final { return std::make_unique<ConjugateGradient_s<_T1, _symmetric>>(*this); }
					std::unique_ptr<Solver<_T1, _symmetric>> move() override final { return std::make_unique<ConjugateGradient_s<_T1, _symmetric>>(std::move(*this)); }
					std::shared_ptr<Solver<_T1, _symmetric>> shared() override final { return std::make_shared<ConjugateGradient_s<_T1, _symmetric>>(*this); }
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES 
			{
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPESD(_T1));
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, true));
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_SPMAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres(SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				// -------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class MINRES_s : virtual public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Col<_T1> r, pkm1, pk, pkp1, Ap_km1, Ap_k, Ap_kp1;

					_T1 beta0_;

				public:
					MINRES_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::MINRES;
						if(!_symmetric) 
							throw std::invalid_argument("MINRES method is only for symmetric matrices.");
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override final;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					std::unique_ptr<Solver<_T1, _symmetric>> clone() const override final { return std::make_unique<MINRES_s<_T1, _symmetric>>(*this); }
					std::unique_ptr<Solver<_T1, _symmetric>> move() override final { return std::make_unique<MINRES_s<_T1, _symmetric>>(std::move(*this)); }
					std::shared_ptr<Solver<_T1, _symmetric>> shared() override final { return std::make_shared<MINRES_s<_T1, _symmetric>>(*this); }
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
			
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			
			/**
			* % MINRES-QLP: Minimum Residual QLP Method - minimal leng solution to symmetric (possibly singular) Ax = b or min ||Ax - b||
			* ---------
			* !TODO 
			*	- Implement the MINRES-QLP method for general symmetric matrices A (possibly singular)
			*	- Implement the MINRES-QLP method for symmetric positive definite matrices A (not singular)
			*  - Implement convergence criterion return rather than finished iterations
			* !CURRENTLY
			* 	- The method is implemented for Fisher matrices, which are symmetric positive definite matrices constructed as S = \Delta O^* \Delta O
			* 	X = minres_qlp(deltaO, deltaO^+, F, x0, eps, max_iter, converged, reg) solves the system Sx = F or the minimization problem ||Sx - F||_2
			*  	The N_samples x N_params matrix deltaO is the derivative of the observable with respect to the parameters (rows are samples, columns are parameters)
			*  	The N_params x N_params matrix deltaO^+ is the conjugate transpose of deltaO (rows are parameters, columns are samples)
			*      The method allows for specification of the initial guess x0, the convergence criterion eps, the maximum number of iterations max_iter, the regularization parameter reg 
			*      such that the system to solve is (S + reg*I)x = F or the minimization problem ||(S + reg*I)x - F||_2
			* @ see MINRES_QLP::minres_qlp in upper part of this namespace - inside other namespaces.
			* 		Additionally, in the method MAXXNORM and ACONDLIM parameters are specified on Norm of X and Condition number of A, respectively.
			* @note The method shall be possible to solve the complex and real systems.
			* @note in minres_qlp one can also specify the preconditioner for the system to solve such that the system to solve is M^{-1}Sx = M^{-1}F or the minimization problem ||M^{-1}Sx - M^{-1}F||_2
			* !CONVERGENCE CRITERION:
			* 		- -1 	(beta_k = 0) 		F and X are eigenvectors of (A - sigma*I) 
			* 		- 0 	(beta_km1  = 0) 	F = 0, X = 0
			* 		- 1     X solves the system to the required tolerance RELRES = RNORM / (ANORM * XNORM + BNORM) <= RTOL, where R = B - (A - sigma*I)X and RNORM = ||R||_2
			* 		- 2     X solves the system to the required tolerance RELRES = ARNORM / (ANORM * XNORM) <= RTOL,  where AR = (A - sigma*I)R and ARNORM = NORM(AR).
			*      	- 3 	same as 1, but with RTOL = EPS
			*      	- 4 	same as 2, but with RTOL = EPS
			*      	- 5 	X converged to eigenvector of (A - sigma*I) 
			*      	- 6     XNORM exceeded MAXXNORM
			*      	- 7     ACOND exceeded ACONDLIM
			*      	- 8 	MAXITER reached
			* 		- 9 	The sytem appears to be singular or badly scaled
			* @ref Sou-Cheng T. Choi and Michael A. Saunders, ALGORITHM: MINRES-QLP for Singular Symmetric and Hermitian Linear Equations and Least-Squares Problems, to appear in ACM Transactions on Mathematical Software.
			* @credit The code was based on the published algorithm and the MATLAB implementation by Sou-Cheng: https://www.mathworks.com/matlabcentral/fileexchange/42419-minres-qlp and translated to C++ 
			* with some modifications and related changes.
			// ---------
			*/
			namespace MINRES_QLP 
			{
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, true));
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MATMUL_ARG_TYPESD(_T1));
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_SPMAT_ARG_TYPESD(_T1)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _x0, _eps, _max_iter, _converged, _reg); }
				template<typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { MAKE_MATRIX_FREE_MULT(_T1); return minres_qlp<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); }
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class MINRES_QLP_s : public Solver<_T1, _symmetric> 
				{
				protected:
					bool rnormvec_ = false;
					std::vector<_T1> resvec_, Aresvec_;			// residuals
					// Lanczos vectors and scalars
					arma::Col<_T1> z_km2, z_km1, z_k;
					_T1 _beta1, _beta_km1, _beta_k, _phi_k;
					arma::Col<_T1> v;
					// !!!!!!!!! Previous left reflection 
					_T1 _delta_k;
					_T1 _c_k_1, _c_k_2, _c_k_3; 							// is cs in the algorithm, cr2, cr1 - cosines
					_T1 _s_k_1, _s_k_2, _s_k_3; 							// is sn in the algorithm
					_T1 _gamma_k, _gamma_km1, _gamma_km2, _gamma_km3;
					_T1 _gamma_min, _gamma_min_km1, _gamma_min_km2; 		// is gamma, gammal, gammal2, gammal3 
					_T1 _tau_k, _tau_km1, _tau_km2;							// use them as previous values of tau's - is tau, taul, taul2 in the algorithm
					_T1 _eps_k, _eps_k_p1;								
					_T1 _Ax_norm_k;												
					// !!!!!!!!!! Previous right reflection
					_T1 _theta_k, _theta_km1, _theta_km2;					// use them as previous values of theta's, is theta, thetal, thetal2 in the algorithm
					_T1 _eta_k, _eta_km1, _eta_km2;	
					// !!!!!!!!!! 
					_T1 _xnorm_k;											// is xi in the algorithm - norm of the solution vector, is also xnorm, xnorml
					_T1 _xl2norm_k;											// is xil in the algorithm : xl2norm
					_T1 _mu_k, _mu_km1, _mu_km2, _mu_km3, _mu_km4;			// use them as previous values of mu'
					_T1 _relres_km1, _relAres_km1;							// use them as previous values of relative residuals
					_T1 _rnorm, _rnorm_km1, _rnorm_km2;						// use them as previous values of rnorm's
					_T1 _relres, _relAres;									// relative residual with a safety margin for beta_k = 0
					// !!!!!!!!! Regarding the wektor w and the solution vector x
					arma::Col<_T1> _w_k, _w_km1, _w_km2;
					arma::Col<_T1> x_km1;
					_T1 _Anorm, _Anorm_km1; 
					_T1 _Acond, _Acond_km1;									// use them as previous values of A's norm and condition number
					// !!!!!!!!! QLP part
					_T1 _gammaqlp_k, _gammaqlp_km1;
					_T1 _thetaqlp_k;
					_T1 _muqlp_k, _muqlp_km1;
					_T1 _root_km1;
					int _QLP_iter;											// number of QLP iterations
					// !!!!!!!!!
					int flag_ = -2;											// flag for convergence

				public:
					MINRES_QLP_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::MINRES_QLP;
						if(!_symmetric) 
							throw std::invalid_argument("MINRES_QLP_s: The MINRES_QLP method is only for symmetric matrices.");
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					std::unique_ptr<Solver<_T1, _symmetric>> clone() const override final { return std::make_unique<MINRES_QLP_s<_T1, _symmetric>>(*this); }
					std::unique_ptr<Solver<_T1, _symmetric>> move() override final { return std::make_unique<MINRES_QLP_s<_T1, _symmetric>>(std::move(*this)); }
					std::shared_ptr<Solver<_T1, _symmetric>> shared() override final { return std::make_shared<MINRES_QLP_s<_T1, _symmetric>>(*this); }
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace PseudoInverse 
			{
				template<typename _T1, bool _symmetric = true>
				class PseudoInverse_s : public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Mat<_T1> Amat_;
				public:
					PseudoInverse_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::PseudoInverse;
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					std::unique_ptr<Solver<_T1, _symmetric>> clone() const override final { return std::make_unique<PseudoInverse_s<_T1, _symmetric>>(*this); }
					std::unique_ptr<Solver<_T1, _symmetric>> move() override final { return std::make_unique<PseudoInverse_s<_T1, _symmetric>>(std::move(*this)); }
					std::shared_ptr<Solver<_T1, _symmetric>> shared() override final { return std::make_shared<PseudoInverse_s<_T1, _symmetric>>(*this); }
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace Direct 
			{
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class Direct_s : public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Mat<_T1> Amat_;
				public:
					Direct_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::Direct;
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					void solve(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* _precond = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					std::unique_ptr<Solver<_T1, _symmetric>> clone() const override final { return std::make_unique<Direct_s<_T1, _symmetric>>(*this); }
					std::unique_ptr<Solver<_T1, _symmetric>> move() override final { return std::make_unique<Direct_s<_T1, _symmetric>>(std::move(*this)); }
					std::shared_ptr<Solver<_T1, _symmetric>> shared() override final { return std::make_shared<Direct_s<_T1, _symmetric>>(*this); }
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
				// ############################################################################################################################################
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			namespace ARMA 
			{
				template<typename _T1>
				arma::Col<_T1> arma_solve(SOLVE_MAT_ARG_TYPESD(_T1)) { return arma::solve(_A, _F); }
				template<typename _T1>
				arma::Col<_T1> arma_solve(SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, true)) { return arma::solve(_A, _F); }
				// --------------------------------------------------------------------------------------------------------------------------------------------
				template<typename _T1, bool _symmetric = true>
				class ARMA_s : public Solver<_T1, _symmetric> 
				{
				protected:
					arma::Mat<_T1> Amat_;
				public:
					ARMA_s(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr)
						: Solver<_T1, _symmetric>(_N, _eps, _max_iter, _reg, _preconditioner)
					{
						this->type_ = Type::ARMA;
					}
					// ----------------------------------------------------------------------------------------------------------------------------------------
					void init(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;
					void init(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr) override;

					// ----------------------------------------------------------------------------------------------------------------------------------------
					void solve(const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::SpMat<_T1>& _A, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::Mat<_T1>& _S, const arma::Mat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					void solve(const arma::SpMat<_T1>& _S, const arma::SpMat<_T1>& _Sp, const arma::Col<_T1>& _F, arma::Col<_T1>* _x0 = nullptr, Precond<_T1, _symmetric>* = nullptr) override;
					// ----------------------------------------------------------------------------------------------------------------------------------------
					auto clone() const 	-> std::unique_ptr<Solver<_T1, _symmetric>> override final { return std::make_unique<ARMA_s<_T1, _symmetric>>(*this); }
					auto move() 		-> std::unique_ptr<Solver<_T1, _symmetric>> override final { return std::make_unique<ARMA_s<_T1, _symmetric>>(std::move(*this)); }
					auto shared() 		-> std::shared_ptr<Solver<_T1, _symmetric>> override final { return std::make_shared<ARMA_s<_T1, _symmetric>>(*this); }
					// ----------------------------------------------------------------------------------------------------------------------------------------
				};
			};
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			
			// with matrix multiplication function and preconditioner
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MATMUL_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));

			// with Matrix A and preconditioner
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			
			// with sparse matrix and preconditioner
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_SPMAT_ARG_TYPESD_PRECONDITIONER(_T1, _symmetric));

			// with matrix multiplication function
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MATMUL_ARG_TYPESD(_T1));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MATMUL_ARG_TYPESD(_T1));

			// with Matrix A
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_MAT_ARG_TYPESD(_T1));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_MAT_ARG_TYPESD(_T1));

			// with sparse matrix
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_SPMAT_ARG_TYPESD(_T1));
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_SPMAT_ARG_TYPESD(_T1));
			
			// ------------------------------------------------------------------------------------------------------------------------------------------------
			template <typename _T1, bool _symmetric = true>
			Solver<_T1, _symmetric>* choose(Solvers::General::Type _type, size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr);
			template <typename _T1, bool _symmetric = true>
			Solver<_T1, _symmetric>* choose(int _type, size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T1, _symmetric>* _preconditioner = nullptr);
			// -----------------------------------------------------------------------------------------------------------------------------------------
			std::string name(Solvers::General::Type _type);
			std::string name(int _type);

			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace Tests
			{
				template <typename _T1, bool _symmetric = true>
				std::pair<arma::Mat<_T1>, arma::Col<_T1>> solve_test_mat_vec(bool _makeRandom = true);
				
				template <typename _T1, bool _symmetric = true>
				void solve_test(const arma::Mat<_T1>& _A, const arma::Col<_T1>& _b, Solvers::General::Type _type, double _eps, int _max_iter, double _reg, int _preconditionertype = -1);
				template <typename _T1, bool _symmetric = true>
				void solve_test(Solvers::General::Type _type, double _eps, int _max_iter, double _reg, int _preconditionertype = -1, bool _makeRandom = false);
				
				template <typename _T1, bool _symmetric = true>
				void solve_test_multiple(const arma::Mat<_T1>& A_true, const arma::Col<_T1>& b_true, double _eps, int _max_iter, double _reg, int _preconditionertype = -1);
				template <typename _T1, bool _symmetric = true>
				void solve_test_multiple(double _eps, int _max_iter, double _reg, int _preconditionertype = -1, bool _makeRandom = false);
			};
		};
		// #################################################################################################################################################
		
		#define SOLVE_FISHER_MATRIX(_T1) 						const arma::Mat<_T1>& _DeltaO
		#define SOLVE_FISHER_MATRICES(_T1) 						const arma::Mat<_T1>& _DeltaO, const arma::Mat<_T1>& _DeltaOConjT
		#define SOLVE_FISHER_ARG_TYPES(_T1) 					SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPES(_T1)
		#define SOLVE_FISHER_ARG_TYPES_PRECONDITIONER(_T1) 		SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPES_PRECONDITIONER(_T1, true)
		// with default values
		#define SOLVE_FISHER_ARG_TYPESD(_T1) 					SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPESD(_T1)
		#define SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1) 	SOLVE_FISHER_MATRICES(_T1), SOLVE_GENERAL_ARG_TYPESD_PRECONDITIONER(_T1, true)
		#define MAKE_MATRIX_FREE_MULT_FISHER(_T) auto _f = [&](const arma::Col<_T>& _x, double _reg) -> arma::Col<_T> { return FisherMatrix::matrixFreeMultiplication<_T>(_DeltaO, _DeltaOConjT, _x, _reg); };
		namespace FisherMatrix 
		{	
			/**
			* This methods are used whenever the matrix can be 
			* decomposed into the form S = \Delta O^* \Delta O, where \Delta O is 
			* the derivative of the observable with respect to the parameters. 
			* The matrix S is symmetric and positive definite, so the conjugate gradient method can be used.
			* @equation S_{ij} = <\Delta O^*_i \Delta O_j> / N 
			*/


			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _DeltaO, const arma::Col<_T>& _x, const double _reg = 0.0);

			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::Mat<_T>& _DeltaO, const arma::Mat<_T>& _DeltaOConjT, const arma::Col<_T>& x, const double _reg = 0.0);
			
			template <typename _T>
			arma::Col<_T> matrixFreeMultiplication(const arma::SpMat<_T>& _DeltaO, const arma::SpMat<_T>& _DeltaOConjT, const arma::Col<_T>& x, const double _reg = 0.0);
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

			// Conjugate gradient solver for the Fisher matrix inversion
			namespace CG 
			{
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_FISHER_ARG_TYPESD(_T1)) 
				{ 
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::CG::conjugate_gradient<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg); 
				}
				template<typename _T1>
				arma::Col<_T1> conjugate_gradient(SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1)) 
				{ 
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::CG::conjugate_gradient<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg); 
				}
			};
			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES
			{
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_FISHER_ARG_TYPESD(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES::minres<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
				template <typename _T1>
				arma::Col<_T1> minres(SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES::minres<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}
			}
			// -----------------------------------------------------------------------------------------------------------------------------------------
			namespace MINRES_QLP 
			{	
				template <typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_FISHER_ARG_TYPESD(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES_QLP::minres_qlp<_T1>(_f, _F, _x0, _eps, _max_iter, _converged, _reg);
				}
				template <typename _T1>
				arma::Col<_T1> minres_qlp(SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
				{
					MAKE_MATRIX_FREE_MULT_FISHER(_T1);
					return General::MINRES_QLP::minres_qlp<_T1>(_f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
				}
			};

			// #################################################################################################################################################
			
			// -----------------------------------------------------------------------------------------------------------------------------------------

			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
			}

			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_FISHER_ARG_TYPESD_PRECONDITIONER(_T1))
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _preconditioner, _eps, _max_iter, _converged, _reg);
			}

			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(Solvers::General::Type _type, SOLVE_FISHER_ARG_TYPESD(_T1))
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _eps, _max_iter, _converged, _reg);
			}
			
			template <typename _T1, bool _symmetric = true>
			arma::Col<_T1> solve(int _type, SOLVE_FISHER_ARG_TYPESD(_T1)) 
			{
				MAKE_MATRIX_FREE_MULT_FISHER(_T1);
				return General::solve<_T1, true>(_type, _f, _F, _x0, _eps, _max_iter, _converged, _reg);
			}

			// #################################################################################################################################################
		};
		
		// ################################################################### ARNOLDI #####################################################################

		/**
		* @brief Arnoldi method for solving eigenvalue problems. Computes V and H such that :math:`AV_n=V_{n+1}\\underline{H}_n`.  If
        * the Krylov subspace becomes A-invariant then V and H are truncated such
        * that :math:`AV_n = V_n H_n`.

		*/
		template <typename _T, bool _symmetric = true, bool _reorthogonalize = false>
		class Arnoldi : public General::Solver<_T, _symmetric>
		{
		protected:
			bool reorthogonalize_ 		= _reorthogonalize;		// reorthogonalize the vectors
			bool isGram_ 				= false;				// is the matrix a Gram matrix 
			bool invariant_ 			= false;				// is the Krylov subspace A-invariant
			size_t krylovDim_ 			= 0;					// dimension of the Krylov subspace
																// as V_n = M * P_n, where M is the preconditioner and P_n is the original basis
			arma::Mat<_T> V_;									// basis (reorthogonalized or not)
			arma::Mat<_T> P_; 									// basis (preconditioned)
			arma::Mat<_T> H_;									// Hessenberg matrix - or Lanczos matrix (if symmetric)
			arma::Col<_T> p_;									// preconditioned vector - maybe unnecessary
			arma::Col<_T> v_;									// original vector
			arma::Col<_T> Av_; 									// A * v
			arma::Col<_T> MAv_; 								// M * A * v
			double vnorm_ 				= 0.0;					// norm of the original vector

		public:
			void init(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr) override final;
			// -----------------------------------------------------------------------------------------------------------------------------------------
			~Arnoldi() 					{};
			Arnoldi() 					= default;
			Arnoldi(size_t _N, double _eps = 1e-10, size_t _max_iter = 1000, double _reg = -1.0, Precond<_T, _symmetric>* _preconditioner = nullptr);
			// Copy constructor
			Arnoldi(const Arnoldi<_T, _symmetric, _reorthogonalize>& _solver)
				: General::Solver<_T, _symmetric>(_solver), reorthogonalize_(_solver.reorthogonalize_), isGram_(_solver.isGram_), invariant_(_solver.invariant_),
				krylovDim_(_solver.krylovDim_), V_(_solver.V_), P_(_solver.P_), H_(_solver.H_), p_(_solver.p_), 
				v_(_solver.v_), Av_(_solver.Av_), MAv_(_solver.MAv_), vnorm_(_solver.vnorm_) {}
			// Move constructor
			Arnoldi(Arnoldi<_T, _symmetric, _reorthogonalize>&& _solver) noexcept
				: General::Solver<_T, _symmetric>(std::move(_solver)), reorthogonalize_(_solver.reorthogonalize_), isGram_(_solver.isGram_), invariant_(_solver.invariant_),
				krylovDim_(_solver.krylovDim_), V_(std::move(_solver.V_)), P_(std::move(_solver.P_)), H_(std::move(_solver.H_)), p_(std::move(_solver.p_)), 
				v_(std::move(_solver.v_)), Av_(std::move(_solver.Av_)), MAv_(std::move(_solver.MAv_)), vnorm_(_solver.vnorm_) {}
			// -----------------------------------------------------------------------------------------------------------------------------------------
		
			// single Arnoldi iteration
			void advance();
			// full Arnoldi iteration
			void iterate();

			// -----------------------------------------------------------------------------------------------------------------------------------------
			// getters
			inline const arma::Mat<_T>& getV() 					const { return this->V_; }
			inline const arma::Mat<_T>& getP() 					const { return this->P_; }
			inline const arma::Mat<_T>& getH() 					const { return this->H_; }
			inline const arma::Col<_T>& getAv() 				const { return this->Av_; }
			inline const arma::Col<_T>& getMAv() 				const { return this->MAv_; }
			inline const arma::subview_col<_T> getV(size_t _i) 	const { return this->V_.col(_i); }
			inline const arma::subview_col<_T> getP(size_t _i) 	const { return this->P_.col(_i); }
			inline const arma::subview_col<_T> getH(size_t _i) 	const { return this->H_.col(_i); }
			
			// -----------------------------------------------------------------------------------------------------------------------------------------
			void solve(const arma::Col<_T>& _F, arma::Col<_T>* _x0 = nullptr, Precond<_T, _symmetric>* _precond = nullptr) override final;
			// -----------------------------------------------------------------------------------------------------------------------------------------
			std::unique_ptr<General::Solver<_T, _symmetric>> clone() const override final { return std::make_unique<Arnoldi<_T, _symmetric, _reorthogonalize>>(*this); }
			std::unique_ptr<General::Solver<_T, _symmetric>> move() override final { return std::make_unique<Arnoldi<_T, _symmetric, _reorthogonalize>>(std::move(*this)); }
			std::shared_ptr<General::Solver<_T, _symmetric>> shared() override final { return std::make_shared<Arnoldi<_T, _symmetric, _reorthogonalize>>(*this); }
		};

		// #############################################################################################################################################
	
	};

