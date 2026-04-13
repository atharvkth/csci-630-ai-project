"""
test_traffic_routing.py
=======================
Unit tests for the Uncertainty-Aware Traffic Routing data structures.

Run from the project root with:
    python -m pytest test_traffic_routing.py -v

The classes (TrafficDistribution, Edge, Node, TrafficGraph) live inside a
Jupyter notebook (.ipynb), not a standalone .py module, so a normal import
won't work.  This file extracts all code cells from the notebook at collection
time, executes them into a shared namespace, and binds the four classes from
that namespace so the tests below can use them exactly as if they were imported.
"""

import json
import math
import os
import pytest

# ── Extract and exec the notebook source ─────────────────────────────────────

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(__file__),
    "uncertainty_aware_traffic_routing.ipynb",
)

def _load_notebook_classes() -> dict:
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    combined_source = "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )

    ns: dict = {}
    exec(compile(combined_source, NOTEBOOK_PATH, "exec"), ns)  # noqa: S102
    return ns


_NS = _load_notebook_classes()

TrafficDistribution = _NS["TrafficDistribution"]
Edge                = _NS["Edge"]
Node                = _NS["Node"]
TrafficGraph        = _NS["TrafficGraph"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_edge(
    from_city="Buffalo",
    to_city="Rochester",
    highway="I-90",
    miles=74.0,
    speed_mph=65.0,
    road_type="interstate",
    am_mean=1.1, am_std=0.05,
    op_mean=1.0, op_std=0.05,
    pm_mean=1.2, pm_std=0.05,
    free_flow=68.0,
) -> Edge:
    """Return a Buffalo->Rochester I-90 edge with real dataset values."""
    return Edge(
        from_city=from_city,
        to_city=to_city,
        highway=highway,
        miles=miles,
        speed_mph=speed_mph,
        road_type=road_type,
        am_peak=TrafficDistribution(am_mean, am_std),
        off_peak=TrafficDistribution(op_mean, op_std),
        pm_peak=TrafficDistribution(pm_mean, pm_std),
        free_flow_time_min=free_flow,
    )


def make_node(city="Buffalo", lat=42.89, lng=-78.87,
              population=278349, region="Western NY") -> Node:
    return Node(city=city, population=population, region=region,
                lat=lat, lng=lng)


# =============================================================================
# TrafficDistribution
# =============================================================================

class TestTrafficDistribution:

    def test_stores_mean_and_std(self):
        td = TrafficDistribution(mean=1.3, std=0.1)
        assert td.mean == 1.3
        assert td.std == 0.1

    def test_sample_returns_float(self):
        td = TrafficDistribution(mean=1.2, std=0.1)
        result = td.sample()
        assert isinstance(result, float)

    def test_sample_never_below_1(self):
        """
        The Gaussian can produce values < 1.0 for low-mean distributions, but
        sample() must clamp to 1.0 because traffic cannot make travel *faster*
        than free-flow speed.
        """
        td = TrafficDistribution(mean=0.5, std=0.5)
        for _ in range(500):
            assert td.sample() >= 1.0

    def test_sample_is_random(self):
        """Two calls on the same distribution should (almost certainly) differ."""
        td = TrafficDistribution(mean=2.0, std=0.5)
        samples = {td.sample() for _ in range(20)}
        assert len(samples) > 1, "sample() appears to be returning a constant"

    def test_sample_clusters_near_mean_for_tight_std(self):
        """With a very small std, samples should be very close to mean."""
        td = TrafficDistribution(mean=1.5, std=0.001)
        for _ in range(100):
            s = td.sample()
            assert abs(s - 1.5) < 0.05

    def test_repr(self):
        td = TrafficDistribution(mean=1.1, std=0.05)
        assert "1.1" in repr(td)
        assert "0.05" in repr(td)


# =============================================================================
# Edge
# =============================================================================

class TestEdge:

    # -- deterministic travel times -------------------------------------------

    def test_free_flow_time(self):
        """get_travel_time_without_traffic() should return free_flow_time_min exactly."""
        edge = make_edge(free_flow=68.0)
        assert edge.get_travel_time_without_traffic() == 68.0

    def test_deterministic_off_peak(self):
        """
        Buffalo->Rochester I-90: free_flow=68, off_peak mean=1.0
        Expected = round(68 * 1.0, 2) = 68.0
        """
        edge = make_edge(op_mean=1.0, free_flow=68.0)
        assert edge.get_expected_travel_time_with_traffic("off_peak") == 68.0

    def test_deterministic_am_peak(self):
        """
        Buffalo->Rochester I-90: free_flow=68, am_peak mean=1.1
        Expected = round(68 * 1.1, 2) = 74.8
        """
        edge = make_edge(am_mean=1.1, free_flow=68.0)
        assert edge.get_expected_travel_time_with_traffic("am_peak") == pytest.approx(74.8)

    def test_deterministic_pm_peak(self):
        """
        Buffalo->Rochester I-90: free_flow=68, pm_peak mean=1.2
        Expected = round(68 * 1.2, 2) = 81.6
        """
        edge = make_edge(pm_mean=1.2, free_flow=68.0)
        assert edge.get_expected_travel_time_with_traffic("pm_peak") == pytest.approx(81.6)

    def test_traffic_ordering_off_peak_fastest(self):
        """Off-peak should be the fastest (or equal to) AM and PM peak times."""
        edge = make_edge()
        off = edge.get_expected_travel_time_with_traffic("off_peak")
        am  = edge.get_expected_travel_time_with_traffic("am_peak")
        pm  = edge.get_expected_travel_time_with_traffic("pm_peak")
        assert off <= am
        assert off <= pm

    def test_traffic_always_at_least_free_flow(self):
        """No traffic period should make travel faster than free-flow."""
        edge = make_edge()
        ff = edge.get_travel_time_without_traffic()
        for period in ("am_peak", "off_peak", "pm_peak"):
            assert edge.get_expected_travel_time_with_traffic(period) >= ff

    # -- probabilistic travel times -------------------------------------------

    def test_probabilistic_returns_float(self):
        edge = make_edge()
        result = edge.get_expected_travel_time_with_probabilistic_traffic("off_peak")
        assert isinstance(result, float)

    def test_probabilistic_never_below_free_flow(self):
        """
        Even with sampling, the multiplier is clamped to >= 1.0 in
        TrafficDistribution.sample(), so stochastic time >= free_flow.
        """
        edge = make_edge(free_flow=68.0)
        for _ in range(200):
            t = edge.get_expected_travel_time_with_probabilistic_traffic("off_peak")
            assert t >= 68.0

    def test_probabilistic_varies_across_calls(self):
        """Stochastic method should produce different values (not a constant)."""
        edge = make_edge(op_std=0.5)  # high std so variance is obvious
        times = {edge.get_expected_travel_time_with_probabilistic_traffic("off_peak")
                 for _ in range(30)}
        assert len(times) > 1

    # -- invalid time period --------------------------------------------------

    def test_invalid_time_period_raises(self):
        edge = make_edge()
        with pytest.raises(ValueError, match="Invalid time_period"):
            edge.get_travel_time("rush_hour")

    # -- repr -----------------------------------------------------------------

    def test_repr_contains_cities_and_highway(self):
        edge = make_edge()
        r = repr(edge)
        assert "Buffalo" in r
        assert "Rochester" in r
        assert "I-90" in r


# =============================================================================
# Node / Haversine
# =============================================================================

class TestNode:

    def test_self_distance_is_zero(self):
        node = make_node()
        assert node.haversine_distance(node) == 0.0

    def test_haversine_buffalo_to_rochester(self):
        """
        Buffalo (42.89, -78.87) -> Rochester (43.16, -77.61).
        Straight-line distance is ~65 miles.  Road distance (I-90) is 74 miles,
        so the heuristic must be strictly less -- confirming admissibility.
        """
        buffalo   = make_node("Buffalo",   lat=42.89, lng=-78.87)
        rochester = make_node("Rochester", lat=43.16, lng=-77.61)
        d = buffalo.haversine_distance(rochester)
        assert 60 < d < 74, f"Expected ~65 mi, got {d}"

    def test_haversine_symmetry(self):
        """Distance A->B must equal distance B->A."""
        buffalo   = make_node("Buffalo",   lat=42.89, lng=-78.87)
        rochester = make_node("Rochester", lat=43.16, lng=-77.61)
        assert buffalo.haversine_distance(rochester) == rochester.haversine_distance(buffalo)

    def test_haversine_triangle_inequality(self):
        """
        d(A, C) <= d(A, B) + d(B, C)
        Using Buffalo, Rochester, Syracuse as A, B, C.
        """
        buffalo   = make_node("Buffalo",   lat=42.89, lng=-78.87)
        rochester = make_node("Rochester", lat=43.16, lng=-77.61)
        syracuse  = make_node("Syracuse",  lat=43.05, lng=-76.15)
        d_ab = buffalo.haversine_distance(rochester)
        d_bc = rochester.haversine_distance(syracuse)
        d_ac = buffalo.haversine_distance(syracuse)
        assert d_ac <= d_ab + d_bc + 0.01  # small epsilon for float rounding

    def test_haversine_admissibility_against_all_edges(self):
        """
        For every edge in the dataset, the straight-line Haversine distance
        between its two endpoints must be <= road distance in miles.
        This is the core admissibility property that makes A* optimal.
        """
        graph = TrafficGraph()
        graph.load_nodes(
            "data/CSCI 630 - Artificial Intelligence - Project Data - ny_graph_nodes.csv"
        )
        graph.load_edges(
            "data/CSCI 630 - Artificial Intelligence - Project Data - ny_graph_edges.csv"
        )
        seen = set()
        for edge in graph.edges:
            key = tuple(sorted((edge.from_city, edge.to_city)))
            if key in seen:
                continue
            seen.add(key)
            a = graph.get_node(edge.from_city)
            b = graph.get_node(edge.to_city)
            straight_line = a.haversine_distance(b)
            assert straight_line <= edge.miles + 0.5, (
                f"Heuristic NOT admissible: {edge.from_city}->{edge.to_city} "
                f"straight={straight_line:.1f} mi > road={edge.miles} mi"
            )

    def test_repr(self):
        node = make_node("Albany", population=99000, region="Capital Region")
        r = repr(node)
        assert "Albany" in r
        assert "Capital Region" in r


# =============================================================================
# TrafficGraph
# =============================================================================

class TestTrafficGraph:

    @pytest.fixture(scope="class")
    def graph(self):
        g = TrafficGraph()
        g.load_nodes(
            "data/CSCI 630 - Artificial Intelligence - Project Data - ny_graph_nodes.csv"
        )
        g.load_edges(
            "data/CSCI 630 - Artificial Intelligence - Project Data - ny_graph_edges.csv"
        )
        return g

    # -- loading --------------------------------------------------------------

    def test_node_count(self, graph):
        assert len(graph.nodes) == 28

    def test_edge_count(self, graph):
        """41 roads x 2 directions = 82 directed edge objects."""
        assert len(graph.edges) == 82

    def test_all_node_names_loaded(self, graph):
        expected = {
            "New York City", "Bronx", "Buffalo", "Rochester", "Syracuse",
            "Albany", "Schenectady", "Utica", "Binghamton", "Ithaca",
        }
        for city in expected:
            assert city in graph.nodes, f"{city} missing from graph"

    # -- get_node -------------------------------------------------------------

    def test_get_node_returns_correct_node(self, graph):
        node = graph.get_node("Buffalo")
        assert node is not None
        assert node.city == "Buffalo"
        assert node.region == "Western NY"

    def test_get_node_returns_none_for_unknown(self, graph):
        assert graph.get_node("Atlantis") is None

    # -- get_neighbors --------------------------------------------------------

    def test_get_neighbors_nonempty(self, graph):
        neighbors = graph.get_neighbors("Buffalo")
        assert len(neighbors) > 0

    def test_get_neighbors_returns_edges(self, graph):
        for edge in graph.get_neighbors("Rochester"):
            assert isinstance(edge, Edge)
            assert edge.from_city == "Rochester"

    def test_undirected_symmetry(self, graph):
        """
        If Buffalo has Rochester as a neighbor, Rochester must have Buffalo
        as a neighbor (undirected graph).
        """
        buf_neighbors = {e.to_city for e in graph.get_neighbors("Buffalo")}
        roc_neighbors = {e.to_city for e in graph.get_neighbors("Rochester")}
        assert "Rochester" in buf_neighbors
        assert "Buffalo" in roc_neighbors

    # -- get_edges_between ----------------------------------------------------

    def test_get_edges_between_returns_multiple_roads(self, graph):
        """Buffalo->Rochester has two roads in the dataset: I-90 and NY-33."""
        edges = graph.get_edges_between("Buffalo", "Rochester")
        assert len(edges) >= 2
        highways = {e.highway for e in edges}
        assert "I-90" in highways
        assert "NY-33" in highways

    def test_get_edges_between_unknown_pair_returns_empty(self, graph):
        """Cities with no direct road connection should return []."""
        result = graph.get_edges_between("Niagara Falls", "New York City")
        assert result == []

    def test_get_edges_between_known_values(self, graph):
        """
        Albany->Schenectady I-90: free_flow=15, off_peak mean=1.0
        Expected deterministic off-peak time = 15.0 min.
        """
        edges = graph.get_edges_between("Albany", "Schenectady")
        i90 = next((e for e in edges if e.highway == "I-90"), None)
        assert i90 is not None
        assert i90.get_expected_travel_time_with_traffic("off_peak") == pytest.approx(15.0)

    # -- adjacency list integrity ---------------------------------------------

    def test_adjacency_keys_match_nodes(self, graph):
        """Every city in nodes should have an entry in adjacency."""
        for city in graph.nodes:
            assert city in graph.adjacency, f"{city} missing from adjacency"

    def test_no_self_loops(self, graph):
        for edge in graph.edges:
            assert edge.from_city != edge.to_city, (
                f"Self-loop detected at {edge.from_city}"
            )
