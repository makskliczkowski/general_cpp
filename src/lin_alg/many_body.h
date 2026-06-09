#pragma once

#include <armadillo>
#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace ManyBody
{

/**
 * Decompose BdG eigenvalues/vectors into (U, V) components.
 *
 * Parameters
 * ----------
 * eig_val : arma::Col<double>
 *     All eigenvalues of the BdG Hamiltonian.
 * eig_vec : arma::Mat<Scalar>
 *     All eigenvectors of the BdG Hamiltonian.
 * U : arma::Mat<Scalar>
 *     Output matrix for particle components.
 * V : arma::Mat<Scalar>
 *     Output matrix for hole components.
 * eig_val_pos : arma::Col<double>
 *     Output vector for positive eigenvalues.
 * tol : double
 *     Tolerance value for zero eigenvalues.
 */
template<typename Scalar>
void bogoliubov_decompose(
    const arma::Col<double>& eig_val,
    const arma::Mat<Scalar>& eig_vec,
    arma::Mat<Scalar>& U,
    arma::Mat<Scalar>& V,
    arma::Col<double>& eig_val_pos,
    double tol = 1e-9)
{
    const auto n_tot = eig_val.n_elem;
    const auto N = n_tot / 2;

    arma::uvec keep = arma::find(eig_val > tol);
    if(keep.n_elem != N)
    {
        arma::uvec sorted_indices = arma::stable_sort_index(eig_val, "descend");
        keep = sorted_indices.head(N);
        keep = arma::sort(keep);
    }

    eig_val_pos = eig_val(keep);
    arma::Mat<Scalar> eig_vec_pos = eig_vec.cols(keep);

    U = eig_vec_pos.rows(0, N - 1);
    V = eig_vec_pos.rows(N, 2 * N - 1);

    for(std::size_t k = 0; k < N; ++k)
    {
        double norm_sq = 0.0;
        for(std::size_t i = 0; i < N; ++i)
        {
            norm_sq += std::norm(U(i, k)) + std::norm(V(i, k));
        }
        double s = std::sqrt(norm_sq);
        if(s > 1e-15)
        {
            for(std::size_t i = 0; i < N; ++i)
            {
                U(i, k) /= s;
                V(i, k) /= s;
            }
        }
    }
}

/**
 * Compute pairing matrix F = V * U^-1.
 *
 * Parameters
 * ----------
 * U : arma::Mat<Scalar>
 *     Particle components matrix.
 * V : arma::Mat<Scalar>
 *     Hole components matrix.
 *
 * Returns
 * -------
 * arma::Mat<Scalar>
 *     Pairing matrix F.
 */
template<typename Scalar>
[[nodiscard]] arma::Mat<Scalar> pairing_matrix(const arma::Mat<Scalar>& U, const arma::Mat<Scalar>& V)
{
    return arma::trans(arma::solve(U.t(), V.t()));
}

/**
 * Compute permanent using Ryser's formula (O(2^n * n)).
 *
 * Parameters
 * ----------
 * M : MatrixType
 *     Input square matrix.
 *
 * Returns
 * -------
 * ElementType
 *     The permanent of matrix M.
 */
template<typename MatrixType>
[[nodiscard]] auto permanent_ryser(const MatrixType& M)
{
    using ElementType = typename MatrixType::elem_type;
    const auto n = M.n_rows;
    if(n == 0)
        return ElementType{1.0};

    ElementType total = ElementType{0.0};
    const auto limit = 1U << n;

    for(std::uint32_t k = 1; k < limit; ++k)
    {
        ElementType prod = ElementType{1.0};
        std::size_t popcount = 0;

        for(std::size_t i = 0; i < n; ++i)
        {
            ElementType row_sum = ElementType{0.0};
            auto mask = k;
            std::size_t col = 0;
            while(mask > 0)
            {
                if((mask & 1U) != 0)
                {
                    row_sum += M(i, col);
                }
                mask >>= 1U;
                ++col;
            }
            prod *= row_sum;
        }

        auto temp = k;
        while(temp > 0)
        {
            temp &= temp - 1;
            ++popcount;
        }

        double sign = ((n - popcount) % 2 == 0) ? 1.0 : -1.0;
        total += sign * prod;
    }

    return total;
}

/**
 * Pfaffian via Parlett-Reid algorithm.
 *
 * Parameters
 * ----------
 * A_in : MatrixType
 *     Input skew-symmetric matrix.
 *
 * Returns
 * -------
 * ElementType
 *     The Pfaffian of the matrix.
 */
template<typename MatrixType>
[[nodiscard]] auto pfaffian_parlett_reid(const MatrixType& A_in)
{
    using ElementType = typename MatrixType::elem_type;
    const auto N = A_in.n_rows;
    if(N == 0)
        return ElementType{1.0};
    if(N % 2 != 0)
        return ElementType{0.0};

    MatrixType A = A_in;
    ElementType pfaffian_val = ElementType{1.0};
    const double ZERO_TOL = 1e-15;

    for(std::size_t k = 0; k < N - 1; k += 2)
    {
        double max_val = 0.0;
        std::size_t kp = k + 1;
        for(std::size_t i = k + 1; i < N; ++i)
        {
            double av = std::abs(A(i, k));
            if(av > max_val)
            {
                max_val = av;
                kp = i;
            }
        }

        if(kp != k + 1)
        {
            for(std::size_t j = 0; j < N; ++j)
            {
                auto tmp = A(k + 1, j);
                A(k + 1, j) = A(kp, j);
                A(kp, j) = tmp;
            }
            for(std::size_t i = 0; i < N; ++i)
            {
                auto tmp = A(i, k + 1);
                A(i, k + 1) = A(i, kp);
                A(i, kp) = tmp;
            }
            pfaffian_val *= -1.0;
        }

        auto pivot_val = A(k + 1, k);
        if(std::abs(pivot_val) < ZERO_TOL)
        {
            return ElementType{0.0};
        }

        pfaffian_val *= A(k, k + 1);

        if(k + 2 < N)
        {
            auto inv_pivot = ElementType{1.0} / A(k, k + 1);
            for(std::size_t i = k + 2; i < N; ++i)
            {
                auto tau_i = A(k, i) * inv_pivot;
                for(std::size_t j = k + 2; j < N; ++j)
                {
                    A(i, j) += tau_i * A(k + 1, j) - A(i, k + 1) * (A(k, j) * inv_pivot);
                }
            }
        }
    }

    return pfaffian_val;
}

/**
 * Recursive helper: sum over perfect matchings encoded by bitmask.
 *
 * Parameters
 * ----------
 * A : MatrixType
 *     Input symmetric matrix.
 * mask : std::uint32_t
 *     Submatrix bitmask.
 *
 * Returns
 * -------
 * ElementType
 *     Hafnian term contribution.
 */
template<typename MatrixType>
[[nodiscard]] auto hafnian_prod_recursive(const MatrixType& A, std::uint32_t mask)
{
    using ElementType = typename MatrixType::elem_type;
    if(mask == 0)
        return ElementType{1.0};

    std::uint32_t lsb = mask & (-static_cast<int>(mask));
    std::size_t i = 0;
    auto tmp = lsb;
    while(tmp > 1)
    {
        tmp >>= 1;
        ++i;
    }
    mask ^= lsb;

    ElementType acc = ElementType{0.0};
    auto rest_mask = mask;
    while(rest_mask > 0)
    {
        std::uint32_t lsb2 = rest_mask & (-static_cast<int>(rest_mask));
        std::size_t j = 0;
        auto tmp2 = lsb2;
        while(tmp2 > 1)
        {
            tmp2 >>= 1;
            ++j;
        }
        rest_mask ^= lsb2;
        acc += A(i, j) * hafnian_prod_recursive(A, mask ^ lsb2);
    }
    return acc;
}

} // namespace ManyBody
