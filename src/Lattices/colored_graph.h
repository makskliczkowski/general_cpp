/******************************************************************************
 *
 *  @file src/Lattices/colored_graph.h
 *  @brief Colour-labelled graph utilities for lattice models.
 *
 *  Header-only, dependency-free (STL only) helpers for reasoning about lattices
 *  whose bonds carry a discrete colour/flavour label, as in Kitaev-type models
 *  where each honeycomb edge is an x-, y- or z-bond. The routines here were
 *  hoisted out of the qes-kitaev application driver so that the plaquette /
 *  flux and colour-symmetry bookkeeping lives with the lattice code it belongs
 *  to and can be reused and unit-tested independently of any solver.
 *
 *  Nothing here depends on Armadillo or on the rest of general_cpp; it operates
 *  purely on plain site indices and colour integers, so callers adapt their own
 *  bond/permutation types to `ColouredEdge` / destination vectors at the seam.
 *
 *  @project general_cpp
 *  @author  Maksymilian Kliczkowski
 *  @copyright   (c) 2024-2026 Maksymilian Kliczkowski
 *  SPDX-License-Identifier: MIT
 *
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace lattices::coloured
{

    using SiteIndex = std::size_t;
    using Colour    = unsigned int;

    //! Sentinel returned by edge_colour when two sites are not adjacent.
    inline constexpr Colour no_colour = static_cast<Colour>(-1);

    //! One undirected edge (i, j) carrying a colour label.
    struct ColouredEdge
    {
        SiteIndex i;
        SiteIndex j;
        Colour    colour;
    };

    //! One ring vertex together with the colour of its "spoke" - the single bond
    //! colour incident to the vertex that is not used by the two ring edges. On a
    //! Kitaev honeycomb the product of these spoke operators is the plaquette
    //! flux W_p.
    struct RingSpoke
    {
        SiteIndex site;
        Colour    spoke;
    };

    //! Adjacency list: site -> list of (neighbour, bond colour).
    using ColouredAdjacency = std::vector<std::vector<std::pair<SiteIndex, Colour>>>;

    /**
     * @brief Build a coloured adjacency list from an undirected edge set.
     * @param edges Coloured edges; endpoints must be < sites.
     * @param sites Number of vertices.
     * @returns Adjacency list of size `sites`.
     */
    [[nodiscard]] inline ColouredAdjacency build_adjacency(
        const std::vector<ColouredEdge>& edges, std::size_t sites)
    {
        ColouredAdjacency adj(sites);
        for(const auto& e : edges)
        {
            if(e.i >= sites || e.j >= sites)
                continue;
            adj[e.i].push_back({e.j, e.colour});
            adj[e.j].push_back({e.i, e.colour});
        }
        return adj;
    }

    /**
     * @brief Colour of the edge between two adjacent sites.
     * @returns The bond colour, or `no_colour` if a and b are not adjacent.
     */
    [[nodiscard]] inline Colour edge_colour(
        const ColouredAdjacency& adj, SiteIndex a, SiteIndex b)
    {
        if(a >= adj.size())
            return no_colour;
        for(const auto& [nbr, colour] : adj[a])
            if(nbr == b)
                return colour;
        return no_colour;
    }

    /**
     * @brief Enumerate the chordless rings (simple cycles) of a fixed length.
     *
     * Each ring is reported once, rooted at its minimum vertex (a DFS that only
     * ever steps to neighbours greater than the start vertex, then closes back
     * onto the start). The returned rings are ordered along the cycle, which the
     * plaquette routines rely on.
     *
     * On a periodic torus with both extents >= ring_length/2 the shortest
     * non-contractible loop is longer than the face perimeter, so for the
     * honeycomb (ring_length = 6) the result is exactly the set of hexagonal
     * faces; callers on very small periodic lattices should gate accordingly
     * (an extent of 2 admits spurious length-6 wraps).
     *
     * @param adj Coloured adjacency list.
     * @param ring_length Perimeter to search for (6 for honeycomb hexagons).
     * @returns Cycles of the requested length, each as an ordered vertex list.
     */
    [[nodiscard]] inline std::vector<std::vector<SiteIndex>> find_rings(
        const ColouredAdjacency& adj, std::size_t ring_length)
    {
        const std::size_t sites = adj.size();
        std::vector<std::vector<SiteIndex>> rings;
        if(ring_length == 0)
            return rings;

        std::set<std::vector<SiteIndex>> seen;
        std::vector<SiteIndex> path;
        std::vector<char> on_path(sites, 0);

        auto dfs = [&](auto&& self, SiteIndex start, SiteIndex current) -> void
        {
            if(path.size() == ring_length)
            {
                if(edge_colour(adj, current, start) != no_colour)
                {
                    std::vector<SiteIndex> key(path);
                    std::sort(key.begin(), key.end());
                    if(seen.insert(key).second)
                        rings.push_back(path);
                }
                return;
            }
            for(const auto& [nbr, colour] : adj[current])
            {
                (void)colour;
                if(nbr == start || on_path[nbr] || nbr < start)
                    continue;   // nbr < start roots each ring at its minimum vertex
                on_path[nbr] = 1;
                path.push_back(nbr);
                self(self, start, nbr);
                path.pop_back();
                on_path[nbr] = 0;
            }
        };

        for(SiteIndex s = 0; s < sites; ++s)
        {
            path = {s};
            on_path.assign(sites, 0);
            on_path[s] = 1;
            dfs(dfs, s, s);
        }
        return rings;
    }

    /**
     * @brief Spoke colours of a coloured ring (the Kitaev-plaquette fingerprint).
     *
     * A genuine Kitaev face has its perimeter edges coloured with each of the
     * `n_colours` colours the same number of times, and at every vertex its two
     * ring edges carry two distinct colours, leaving exactly one "spoke" colour
     * (for three colours, spoke = 3 - c1 - c2). Rings that fail either check -
     * for instance the spurious non-contractible cycles a bare search finds on a
     * small periodic torus - return std::nullopt.
     *
     * @param ring Ordered ring vertices (as returned by find_rings).
     * @param adj Coloured adjacency list.
     * @param n_colours Number of distinct bond colours (3 for Kitaev).
     * @returns Per-vertex spoke colours, or nullopt if `ring` is not a valid
     *          coloured face.
     */
    [[nodiscard]] inline std::optional<std::vector<RingSpoke>> ring_spokes(
        const std::vector<SiteIndex>& ring,
        const ColouredAdjacency& adj,
        unsigned n_colours = 3)
    {
        const std::size_t n = ring.size();
        if(n == 0 || n_colours == 0 || n % n_colours != 0)
            return std::nullopt;

        // each colour must appear the same number of times around the perimeter
        std::vector<int> edge_count(n_colours, 0);
        for(std::size_t k = 0; k < n; ++k)
        {
            const Colour c = edge_colour(adj, ring[k], ring[(k + 1) % n]);
            if(c >= n_colours)
                return std::nullopt;
            ++edge_count[c];
        }
        const int expected = static_cast<int>(n / n_colours);
        for(int count : edge_count)
            if(count != expected)
                return std::nullopt;

        // the spoke of a vertex is the colour missing from its two ring edges;
        // this is well defined only for three colours (c1 + c2 + spoke = 0+1+2)
        std::vector<RingSpoke> spokes;
        spokes.reserve(n);
        for(std::size_t k = 0; k < n; ++k)
        {
            const SiteIndex v    = ring[k];
            const SiteIndex prev = ring[(k + n - 1) % n];
            const SiteIndex next = ring[(k + 1) % n];
            const Colour    c1   = edge_colour(adj, v, prev);
            const Colour    c2   = edge_colour(adj, v, next);
            if(c1 == c2 || c1 >= n_colours || c2 >= n_colours)
                return std::nullopt;
            if(n_colours == 3)
                spokes.push_back({v, 3u - c1 - c2});
            else
            {
                // general case: the unique colour not equal to c1 or c2 only if
                // n_colours == 3; otherwise the spoke is ambiguous.
                return std::nullopt;
            }
        }
        return spokes;
    }

    /**
     * @brief Does a site permutation preserve the coloured-edge multiset?
     *
     * A permutation is a colour symmetry of the lattice iff mapping every edge
     * through it lands on an edge of the same colour. Used to reject lattice
     * layouts for which a translation would silently produce a wrong
     * symmetry-reduced spectrum.
     *
     * @param destination destination[i] = image of site i under the permutation.
     * @param edges Coloured edges.
     * @returns true iff every mapped edge is a same-colour edge of the set.
     */
    [[nodiscard]] inline bool permutation_preserves_colouring(
        const std::vector<SiteIndex>& destination,
        const std::vector<ColouredEdge>& edges)
    {
        std::set<std::array<SiteIndex, 3>> canonical;
        for(const auto& e : edges)
            canonical.insert({std::min(e.i, e.j), std::max(e.i, e.j),
                              static_cast<SiteIndex>(e.colour)});
        for(const auto& e : edges)
        {
            if(e.i >= destination.size() || e.j >= destination.size())
                return false;
            const SiteIndex pi = destination[e.i];
            const SiteIndex pj = destination[e.j];
            const std::array<SiteIndex, 3> mapped {
                std::min(pi, pj), std::max(pi, pj),
                static_cast<SiteIndex>(e.colour)};
            if(!canonical.count(mapped))
                return false;
        }
        return true;
    }

} // namespace lattices::coloured
