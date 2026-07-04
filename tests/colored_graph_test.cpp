// Standalone unit test for src/Lattices/colored_graph.h.
//
// Builds a 3x3 periodic Kitaev honeycomb by hand (18 sites, 27 coloured bonds,
// 9 hexagonal faces) and checks ring enumeration, spoke assignment, colour
// symmetry detection and the adjacency/colour helpers. No Armadillo, no
// general_cpp dependencies: compile with
//   g++ -std=c++20 -I../src colored_graph_test.cpp -o colored_graph_test
//
// Returns 0 on success, non-zero on the first failed check.

#include "Lattices/colored_graph.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace lattices::coloured;

namespace
{

int failures = 0;

void check(bool ok, const char* what)
{
    if(!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// A brick-wall honeycomb on an lx x ly torus with two sites per cell.
// Cell n = (cx, cy) hosts sublattice-A site 2n and sublattice-B site 2n+1.
// Bonds (matching the qes-kitaev colouring convention):
//   z (colour 2): A(n) - B(n)                      intra-cell
//   x (colour 0): B(n) - A(n + e_x)                +x
//   y (colour 1): B(n) - A(n + e_y)                +y
std::vector<ColouredEdge> honeycomb_bonds(std::size_t lx, std::size_t ly)
{
    std::vector<ColouredEdge> edges;
    auto cell = [lx, ly](std::size_t cx, std::size_t cy)
    { return (cy % ly) * lx + (cx % lx); };
    for(std::size_t cy = 0; cy < ly; ++cy)
        for(std::size_t cx = 0; cx < lx; ++cx)
        {
            const std::size_t n = cell(cx, cy);
            const std::size_t a = 2 * n;
            const std::size_t b = 2 * n + 1;
            edges.push_back({a, b, 2});                          // z
            edges.push_back({b, 2 * cell(cx + 1, cy), 0});       // x
            edges.push_back({b, 2 * cell(cx, cy + 1), 1});       // y
        }
    return edges;
}

} // namespace

int main()
{
    const std::size_t lx = 3, ly = 3;
    const std::size_t sites = 2 * lx * ly;
    const auto edges = honeycomb_bonds(lx, ly);
    const auto adj   = build_adjacency(edges, sites);

    check(edges.size() == 3 * lx * ly, "bond count = 3 * cells");

    // every site has exactly three coloured neighbours, one of each colour
    for(std::size_t s = 0; s < sites; ++s)
    {
        check(adj[s].size() == 3, "degree 3");
        bool has[3] = {false, false, false};
        for(const auto& [nbr, c] : adj[s])
        {
            (void)nbr;
            if(c < 3) has[c] = true;
        }
        check(has[0] && has[1] && has[2], "one bond of each colour per site");
    }

    // edge_colour round-trips and reports non-adjacency
    check(edge_colour(adj, 0, 1) == 2, "intra-cell z bond colour");
    check(edge_colour(adj, 0, 2) == no_colour, "non-adjacent -> no_colour");

    // A bare length-6 cycle search on a small periodic torus also picks up
    // spurious non-contractible wraps; the spoke fingerprint must filter those
    // out so that exactly `cells` genuine Kitaev faces survive.
    const auto rings = find_rings(adj, 6);
    check(rings.size() >= lx * ly, "at least `cells` six-cycles found");

    std::size_t valid_faces = 0;
    for(const auto& ring : rings)
    {
        const auto spokes = ring_spokes(ring, adj, 3);
        if(!spokes) continue;
        ++valid_faces;
        check(spokes->size() == 6, "6 spokes per hexagon");
        // each spoke colour is one of {0,1,2} and each colour appears twice
        int spoke_count[3] = {0, 0, 0};
        for(const auto& s : *spokes)
        {
            check(s.spoke < 3, "spoke colour in range");
            if(s.spoke < 3) ++spoke_count[s.spoke];
        }
        check(spoke_count[0] == 2 && spoke_count[1] == 2 && spoke_count[2] == 2,
              "each spoke colour twice");
    }
    check(valid_faces == lx * ly, "all faces are valid Kitaev plaquettes");

    // translation by one cell in x is a colour symmetry; a random transposition
    // that breaks the colouring is not.
    std::vector<SiteIndex> shift_x(sites);
    for(std::size_t cy = 0; cy < ly; ++cy)
        for(std::size_t cx = 0; cx < lx; ++cx)
        {
            const std::size_t from = (cy * lx + cx);
            const std::size_t to   = (cy * lx + (cx + 1) % lx);
            shift_x[2 * from]     = 2 * to;
            shift_x[2 * from + 1] = 2 * to + 1;
        }
    check(permutation_preserves_colouring(shift_x, edges),
          "x-translation preserves colouring");

    std::vector<SiteIndex> broken(sites);
    for(std::size_t s = 0; s < sites; ++s) broken[s] = s;
    std::swap(broken[0], broken[3]);
    check(!permutation_preserves_colouring(broken, edges),
          "colour-breaking permutation rejected");

    if(failures == 0)
        std::printf("colored_graph_test: all checks passed "
                    "(sites=%zu, six-cycles=%zu, valid faces=%zu)\n",
                    sites, rings.size(), valid_faces);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
