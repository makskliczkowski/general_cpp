#include "../../src/lattices/lattice_symmetry.h"

#include <map>
#include <stdexcept>
#include <utility>

namespace LatticeSym
{
	// ################################### TRANSLATIONS ####################################

	std::vector<Perm> translation_perms(std::size_t Lx, std::size_t Ly, std::size_t sites_per_cell)
	{
		if (Lx == 0 || Ly == 0 || sites_per_cell == 0)
			throw std::invalid_argument("Translation table requires positive extents and sites per cell.");

		const std::size_t Ns	= Lx * Ly * sites_per_cell;
		std::vector<Perm> perms;
		perms.reserve(Lx * Ly);

		for (std::size_t tx = 0; tx < Lx; ++tx)
		{
			for (std::size_t ty = 0; ty < Ly; ++ty)
			{
				Perm perm(Ns);
				for (std::size_t site = 0; site < Ns; ++site)
				{
					const auto cell	= site / sites_per_cell;
					const auto sub	= site % sites_per_cell;
					const auto x	= cell % Lx;
					const auto y	= cell / Lx;
					const auto nx	= (x + tx) % Lx;
					const auto ny	= (y + ty) % Ly;
					perm[site]		= static_cast<uint32_t>((ny * Lx + nx) * sites_per_cell + sub);
				}
				perms.push_back(std::move(perm));
			}
		}
		return perms;
	}

	// #################################### POINT GROUP ####################################

	std::vector<Perm> point_group_perms_square(std::size_t Lx, std::size_t Ly)
	{
		if (Lx == 0 || Ly == 0)
			throw std::invalid_argument("Point group table requires positive extents.");

		const std::size_t Ns	= Lx * Ly;
		std::vector<Perm> ops;

		Perm identity(Ns);
		for (std::size_t site = 0; site < Ns; ++site)
			identity[site]	= static_cast<uint32_t>(site);
		ops.push_back(std::move(identity));

		// D4 exists only for the square aspect ratio.
		if (Lx != Ly)
			return ops;

		const std::size_t L		= Lx;
		const auto site_at		= [L](std::size_t x, std::size_t y) { return (y % L) * L + (x % L); };

		// C4 rotations (90, 180, 270 degrees) followed by the four mirrors
		// (m_x, m_y, m_d1, m_d2); destination convention new = op(old).
		using CoordMap	= std::pair<std::size_t, std::size_t> (*)(std::size_t, std::size_t, std::size_t);
		const CoordMap maps[] = {
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(L - 1 - y, x); },
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(L - 1 - x, L - 1 - y); },
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(y, L - 1 - x); },
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(L - 1 - x, y); },
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(x, L - 1 - y); },
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(y, x); },
			[](std::size_t x, std::size_t y, std::size_t L) { return std::make_pair(L - 1 - y, L - 1 - x); },
		};

		for (const auto& map : maps)
		{
			Perm perm(Ns);
			for (std::size_t site = 0; site < Ns; ++site)
			{
				const auto [nx, ny]	= map(site % L, site / L, L);
				perm[site]			= static_cast<uint32_t>(site_at(nx, ny));
			}
			ops.push_back(std::move(perm));
		}
		return ops;
	}

	// #################################### SPACE GROUP ####################################

	std::vector<Perm> space_group_perms(std::size_t Lx, std::size_t Ly, std::size_t sites_per_cell, bool full_point_group)
	{
		auto trans	= translation_perms(Lx, Ly, sites_per_cell);

		// Point-group composition is implemented for Bravais square cells only.
		if (!full_point_group || sites_per_cell > 1)
			return trans;

		const auto point	= point_group_perms_square(Lx, Ly);

		std::vector<Perm> combined;
		std::map<Perm, bool> seen;
		combined.reserve(trans.size() * point.size());

		for (const auto& t : trans)
		{
			for (const auto& p : point)
			{
				// c = t o p: point-group element first, then the translation.
				Perm c(p.size());
				for (std::size_t site = 0; site < p.size(); ++site)
					c[site]	= t[p[site]];

				if (!seen.emplace(c, true).second)
					continue;
				combined.push_back(std::move(c));
			}
		}
		return combined;
	}

	// #################################### CAYLEY TABLE ###################################

	std::vector<std::vector<uint32_t>> cayley_table(const std::vector<Perm>& perms)
	{
		const std::size_t n	= perms.size();
		if (n == 0)
			return {};
		const std::size_t Ns	= perms.front().size();

		// Inverse permutations: inv[perm[s]] = s.
		std::vector<Perm> inverses(n, Perm(Ns));
		std::map<Perm, std::size_t> index_of;
		for (std::size_t i = 0; i < n; ++i)
		{
			for (std::size_t site = 0; site < Ns; ++site)
				inverses[i][perms[i][site]]	= static_cast<uint32_t>(site);
			index_of.emplace(perms[i], i);
		}

		std::vector<std::vector<uint32_t>> cayley(n, std::vector<uint32_t>(n));
		Perm composed(Ns);
		for (std::size_t i = 0; i < n; ++i)
		{
			for (std::size_t j = 0; j < n; ++j)
			{
				// composed = g_i^{-1} o g_j (destination arrays compose right to left).
				for (std::size_t site = 0; site < Ns; ++site)
					composed[site]	= inverses[i][perms[j][site]];

				const auto found	= index_of.find(composed);
				if (found == index_of.end())
					throw std::invalid_argument("Cayley table input is not closed under composition.");
				cayley[i][j]		= static_cast<uint32_t>(found->second);
			}
		}
		return cayley;
	}
};
