"""
Test suite for Uncertainty-Aware Traffic Routing
CSCI-630 | Artificial Intelligence

Extracts code from the notebook, then runs tests against the four core
data-structure classes: TrafficDistribution, Edge, Node, TrafficGraph.

Usage:
    python -m pytest test_traffic_routing.py -v
"""

import ast
import math
import random
import textwrap
import pytest
import json, types

def _load_notebook_code(nb_path: str) -> str:
    """Read a .ipynb and concatenate all code cells into one big string."""
    with open(nb_path, "r") as f:
        nb = json.load(f)
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            chunks.append(src)
    return "\n\n".join(chunks)

# Execute notebook code in a throwaway namespace, then pull symbols we need.
_NB_PATH = "uncertainty_aware_traffic_routing.ipynb"
_nb_code = _load_notebook_code(_NB_PATH)

# Strip out lines that would trigger graph loading / matplotlib display
_filtered = "\n".join(
    line for line in _nb_code.splitlines()
    if not line.strip().startswith(("traffic_graph", "visualize_graph"))
)
_ns = {}
exec(_filtered, _ns)

TrafficDistribution = _ns["TrafficDistribution"]
Edge                = _ns["Edge"]
Node                = _ns["Node"]
TrafficGraph        = _ns["TrafficGraph"]
reconstruct_path    = _ns["reconstruct_path"]

# Paths to the CSV data files (adjust if your layout differs)
NODES_CSV = "data/CSCI 630 - Artificial Intelligence - Project Data - ny_graph_nodes.csv"
EDGES_CSV = "data/CSCI 630 - Artificial Intelligence - Project Data - ny_graph_edges.csv"


# ========================== fixtures ========================================

@pytest.fixture(scope="module")
def graph():
    """Load the full TrafficGraph once and share it across every test."""
    g = TrafficGraph()
    g.load_nodes(NODES_CSV)
    g.load_edges(EDGES_CSV)
    return g


# ========================== TrafficDistribution =============================

class TestTrafficDistribution:
    """Basic sanity checks on the Gaussian traffic model."""

    def test_attributes_stored(self):
        """mean and std should simply be stored as-is."""
        td = TrafficDistribution(mean=1.5, std=0.2)
        assert td.mean == 1.5
        assert td.std == 0.2

    def test_sample_never_below_one(self):
        """
        Even with a very low mean / high std the multiplier is clamped to 1.0.
        A multiplier < 1 would mean faster-than-free-flow, which is nonsensical.
        """
        td = TrafficDistribution(mean=1.0, std=5.0)  # extreme std
        random.seed(0)
        for _ in range(200):
            assert td.sample() >= 1.0

    def test_sample_is_stochastic(self):
        """Two calls shouldn't always return the same value."""
        td = TrafficDistribution(mean=2.0, std=0.5)
        random.seed(42)
        samples = {td.sample() for _ in range(30)}
        assert len(samples) > 1, "sample() returned the same value every time"

    def test_repr(self):
        td = TrafficDistribution(mean=1.3, std=0.1)
        assert "1.3" in repr(td) and "0.1" in repr(td)


# ========================== Edge ============================================

class TestEdge:
    """Checks travel-time calculations on a hand-built edge."""

    @pytest.fixture()
    def sample_edge(self):
        return Edge(
            from_city="A", to_city="B", highway="I-99",
            miles=60, speed_mph=60, road_type="interstate",
            am_peak=TrafficDistribution(2.0, 0.3),
            off_peak=TrafficDistribution(1.1, 0.1),
            pm_peak=TrafficDistribution(2.5, 0.4),
            free_flow_time_min=60.0,   # 60 mi / 60 mph = 60 min
        )

    def test_free_flow_time(self, sample_edge):
        """With no traffic, travel time == free_flow_time_min."""
        assert sample_edge.get_travel_time_without_traffic() == 60.0

    def test_deterministic_am(self, sample_edge):
        """AM peak deterministic = free_flow * am_mean = 60 * 2.0 = 120."""
        assert sample_edge.get_expected_travel_time_with_traffic("am_peak") == 120.0

    def test_deterministic_off_peak(self, sample_edge):
        """Off-peak deterministic = 60 * 1.1 = 66."""
        assert sample_edge.get_expected_travel_time_with_traffic("off_peak") == 66.0

    def test_deterministic_pm(self, sample_edge):
        """PM peak deterministic = 60 * 2.5 = 150."""
        assert sample_edge.get_expected_travel_time_with_traffic("pm_peak") == 150.0

    def test_off_peak_leq_peaks(self, sample_edge):
        """Off-peak mean travel time should be ≤ both rush-hour means."""
        off = sample_edge.get_expected_travel_time_with_traffic("off_peak")
        am  = sample_edge.get_expected_travel_time_with_traffic("am_peak")
        pm  = sample_edge.get_expected_travel_time_with_traffic("pm_peak")
        assert off <= am
        assert off <= pm

    def test_sampled_geq_free_flow(self, sample_edge):
        """Sampled travel time should never be below free-flow time."""
        random.seed(7)
        for _ in range(100):
            t = sample_edge.get_expected_travel_time_with_probabilistic_traffic("off_peak")
            assert t >= sample_edge.free_flow_time_min

    def test_invalid_time_period_raises(self, sample_edge):
        with pytest.raises(ValueError):
            sample_edge.get_travel_time("midnight")

    def test_repr(self, sample_edge):
        r = repr(sample_edge)
        assert "A" in r and "B" in r and "I-99" in r


# ========================== Node ============================================

class TestNode:
    """Haversine heuristic checks."""

    @pytest.fixture()
    def buffalo(self):
        return Node("Buffalo", 278349, "Western NY", 42.89, -78.87)

    @pytest.fixture()
    def rochester(self):
        return Node("Rochester", 211328, "Western NY", 43.16, -77.61)

    def test_self_distance_zero(self, buffalo):
        assert buffalo.haversine_distance(buffalo) == 0.0

    def test_symmetry(self, buffalo, rochester):
        assert buffalo.haversine_distance(rochester) == rochester.haversine_distance(buffalo)

    def test_haversine_less_than_road(self, buffalo, rochester):
        """
        Straight-line distance should be strictly less than the driving
        distance (~73 mi on I-90).  That's what makes it admissible.
        """
        h = buffalo.haversine_distance(rochester)
        assert h < 73, f"Haversine {h} mi should be < 73 mi road distance"

    def test_admissibility_all_edges(self, graph):
        """
        For EVERY edge in the dataset, haversine between its two endpoints
        must be ≤ the road mileage.  If this ever fails, A* loses its
        optimality guarantee.
        """
        seen = set()
        for edge in graph.edges:
            key = tuple(sorted((edge.from_city, edge.to_city)))
            if key in seen:
                continue
            seen.add(key)
            n1 = graph.get_node(edge.from_city)
            n2 = graph.get_node(edge.to_city)
            h = n1.haversine_distance(n2)
            assert h <= edge.miles, (
                f"Haversine {h} > road miles {edge.miles} on "
                f"{edge.from_city}→{edge.to_city} — heuristic is inadmissible!"
            )


# ========================== TrafficGraph ====================================

class TestTrafficGraph:
    """Integration-level checks on the loaded graph."""

    def test_node_count(self, graph):
        """We expect 27 cities (the CSV has 27 data rows after the header)."""
        # The README says 28, but the CSV header + 27 data lines = 27 nodes.
        # Accept either 27 or 28 — whichever your CSV actually has.
        assert len(graph.nodes) in (27, 28)

    def test_edge_count(self, graph):
        """41 roads × 2 directions = 82 directed edges (or 42×2=84)."""
        assert len(graph.edges) in (82, 84)

    def test_get_node_hit(self, graph):
        node = graph.get_node("Buffalo")
        assert node is not None
        assert node.city == "Buffalo"

    def test_get_node_miss(self, graph):
        assert graph.get_node("Atlantis") is None

    def test_neighbors_nonempty(self, graph):
        """Every city in the graph should have at least one neighbor."""
        for city in graph.nodes:
            assert len(graph.get_neighbors(city)) > 0, f"{city} has 0 neighbors"

    def test_undirected_symmetry(self, graph):
        """
        If there's an edge A→B, there must also be an edge B→A,
        because the graph is undirected.
        """
        for edge in graph.edges:
            reverse = graph.get_edges_between(edge.to_city, edge.from_city)
            assert len(reverse) > 0, (
                f"Edge {edge.from_city}→{edge.to_city} exists but reverse does not"
            )

    def test_no_self_loops(self, graph):
        """No city should have an edge to itself."""
        for edge in graph.edges:
            assert edge.from_city != edge.to_city

    def test_spot_check_node_attrs(self, graph):
        """Verify a known city's attributes against the CSV."""
        nyc = graph.get_node("New York City")
        assert nyc is not None
        assert nyc.region == "NYC Metro"
        assert nyc.lat == pytest.approx(40.71, abs=0.01)

    def test_edges_between_known_pair(self, graph):
        """NYC↔Yonkers should have multiple road options (I-87, Hutch, I-278)."""
        edges = graph.get_edges_between("New York City", "Yonkers")
        assert len(edges) >= 2, "Expected multiple roads between NYC and Yonkers"


# ========================== reconstruct_path ================================

class TestReconstructPath:
    """Quick check that path reconstruction does the right thing."""

    def test_simple_chain(self):
        """A→B→C should reconstruct correctly from a visited dict."""
        visited = {
            "A": (0, None),
            "B": (10, "A"),
            "C": (25, "B"),
        }
        path = reconstruct_path(visited, "A", "C")
        assert path == ["A", "B", "C"]

    def test_single_node(self):
        """Start == goal should return a one-element path."""
        visited = {"X": (0, None)}
        assert reconstruct_path(visited, "X", "X") == ["X"]