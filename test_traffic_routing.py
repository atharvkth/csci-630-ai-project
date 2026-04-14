"""
Test suite for Uncertainty-Aware Traffic Routing
CSCI-630 | Artificial Intelligence

Extracts code from the notebook, then runs tests against the four core
data-structure classes (TrafficDistribution, Edge, Node, TrafficGraph)
and the Phase 2 search algorithms (UCS, A*).

Usage:
    python3 -m pytest test_traffic_routing.py -v
"""

import math
import random
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: pull all code cells out of the .ipynb and exec them so we can
# import the classes without converting the notebook to a .py file.
# ---------------------------------------------------------------------------

import json

def _load_notebook_code(nb_path: str) -> str:
    """Read a .ipynb and concatenate all safe code cells into one string."""
    with open(nb_path, "r") as f:
        nb = json.load(f)
    chunks = []
    # Keywords that indicate a cell should be skipped entirely
    SKIP_CELL_IF_CONTAINS = (
        "run_phase_2_tests(",       # the inline test runner invocations
        "ucs_by_id",                # the comparison summary cell
        "visualize_graph(traffic",  # visualization calls
    )
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(marker in src for marker in SKIP_CELL_IF_CONTAINS):
            # Still keep function/class definitions from mixed cells
            # by extracting only def/class blocks
            safe_lines = []
            inside_def = False
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith(("def ", "class ")):
                    inside_def = True
                elif inside_def and stripped and not line[0] in (" ", "\t"):
                    # We've left the function body
                    inside_def = False
                if inside_def:
                    safe_lines.append(line)
            if safe_lines:
                chunks.append("\n".join(safe_lines))
            continue
        # Skip top-level statements that load data
        filtered_lines = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("traffic_graph", "visualize_graph")):
                continue
            filtered_lines.append(line)
        chunks.append("\n".join(filtered_lines))
    return "\n\n".join(chunks)

# Execute notebook code in a throwaway namespace, then pull symbols we need.
_NB_PATH = "uncertainty_aware_traffic_routing.ipynb"
_nb_code = _load_notebook_code(_NB_PATH)

_ns = {}
exec(_nb_code, _ns)

TrafficDistribution   = _ns["TrafficDistribution"]
Edge                  = _ns["Edge"]
Node                  = _ns["Node"]
TrafficGraph          = _ns["TrafficGraph"]
uniform_cost_search   = _ns["uniform_cost_search"]
a_star_search         = _ns["a_star_search"]
path_to_city_sequence = _ns["path_to_city_sequence"]
path_cost             = _ns["path_cost"]

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

    def test_lt_ordering(self):
        """Edge.__lt__ should allow edges to be compared for heap ordering."""
        e1 = Edge("A", "B", "I-1", 10, 60, "interstate",
                  TrafficDistribution(1.0, 0.1), TrafficDistribution(1.0, 0.1),
                  TrafficDistribution(1.0, 0.1), 10.0)
        e2 = Edge("A", "C", "I-2", 20, 60, "interstate",
                  TrafficDistribution(1.0, 0.1), TrafficDistribution(1.0, 0.1),
                  TrafficDistribution(1.0, 0.1), 20.0)
        assert isinstance(e1 < e2, bool)

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
        assert len(graph.nodes) in (27, 28)

    def test_edge_count(self, graph):
        """41 or 42 roads × 2 directions."""
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
        """If there's an edge A→B, there must also be an edge B→A."""
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


# ========================== Helpers (path_to_city_sequence, path_cost) =======

class TestHelpers:
    """Tests for the path utility functions added in Phase 2."""

    def test_path_to_city_sequence(self):
        """Should produce [start, edge1.to, edge2.to, ...]."""
        e1 = Edge("A", "B", "I-1", 10, 60, "interstate",
                  TrafficDistribution(1.0, 0.1), TrafficDistribution(1.0, 0.1),
                  TrafficDistribution(1.0, 0.1), 10.0)
        e2 = Edge("B", "C", "I-1", 20, 60, "interstate",
                  TrafficDistribution(1.0, 0.1), TrafficDistribution(1.0, 0.1),
                  TrafficDistribution(1.0, 0.1), 20.0)
        assert path_to_city_sequence("A", [e1, e2]) == ["A", "B", "C"]

    def test_path_to_city_sequence_none(self):
        """None path (unreachable) should return just the start city."""
        assert path_to_city_sequence("X", None) == ["X"]

    def test_path_cost_simple(self):
        """Cost should be the sum of free-flow times along the path."""
        e1 = Edge("A", "B", "I-1", 10, 60, "interstate",
                  TrafficDistribution(1.0, 0.1), TrafficDistribution(1.0, 0.1),
                  TrafficDistribution(1.0, 0.1), 10.0)
        e2 = Edge("B", "C", "I-1", 20, 60, "interstate",
                  TrafficDistribution(1.0, 0.1), TrafficDistribution(1.0, 0.1),
                  TrafficDistribution(1.0, 0.1), 25.0)
        assert path_cost([e1, e2]) == 35.0

    def test_path_cost_none(self):
        """None path should cost 0."""
        assert path_cost(None) == 0


# ========================== Phase 2: Search Algorithms ======================

class TestUCS:
    """Uniform Cost Search tests using the notebook's own test cases."""

    def test_1hop_albany_troy(self, graph):
        """Case 1: Albany→Troy should pick I-787 (9 min) over NY-2 (15 min)."""
        path, expanded = uniform_cost_search(graph, "Albany", "Troy")
        assert path_to_city_sequence("Albany", path) == ["Albany", "Troy"]
        assert path_cost(path) == 9.0

    def test_1hop_utica_rome(self, graph):
        """Case 2: Utica→Rome should pick I-90 (15 min) over NY-49 (19 min)."""
        path, expanded = uniform_cost_search(graph, "Utica", "Rome")
        assert path_to_city_sequence("Utica", path) == ["Utica", "Rome"]
        assert path_cost(path) == 15.0

    def test_direct_vs_detour_buffalo_niagara(self, graph):
        """Case 3: Direct I-190 (17 min) beats detour through Tonawanda (20 min)."""
        path, expanded = uniform_cost_search(graph, "Buffalo", "Niagara Falls")
        assert path_to_city_sequence("Buffalo", path) == ["Buffalo", "Niagara Falls"]
        assert path_cost(path) == 17.0

    def test_direct_binghamton_elmira(self, graph):
        """Case 4: Direct I-86 (48 min) beats the Ithaca detour (90 min)."""
        path, expanded = uniform_cost_search(graph, "Binghamton", "Elmira")
        assert path_to_city_sequence("Binghamton", path) == ["Binghamton", "Elmira"]
        assert path_cost(path) == 48.0

    def test_long_buffalo_utica(self, graph):
        """Case 5: Buffalo→Rochester→Syracuse→Utica (192 min)."""
        path, expanded = uniform_cost_search(graph, "Buffalo", "Utica")
        assert path_to_city_sequence("Buffalo", path) == ["Buffalo", "Rochester", "Syracuse", "Utica"]
        assert path_cost(path) == 192.0

    def test_unreachable_nyc_albany(self, graph):
        """Case 6: NYC and Albany are disconnected — must return None."""
        path, expanded = uniform_cost_search(graph, "New York City", "Albany")
        assert path is None

    def test_source_equals_goal(self, graph):
        """EC-1: Rochester→Rochester should return an empty path with cost 0."""
        path, expanded = uniform_cost_search(graph, "Rochester", "Rochester")
        assert path_to_city_sequence("Rochester", path) == ["Rochester"]
        assert path_cost(path) == 0.0

    def test_parallel_edges_nyc_yonkers(self, graph):
        """EC-2: NYC→Yonkers should pick the 20-min edge out of three options."""
        path, expanded = uniform_cost_search(graph, "New York City", "Yonkers")
        assert path_to_city_sequence("New York City", path) == ["New York City", "Yonkers"]
        assert path_cost(path) == 20.0

    def test_sink_node_amityville(self, graph):
        """EC-3: Amityville (near-leaf node) should still be reachable."""
        path, expanded = uniform_cost_search(graph, "Amityville", "Hempstead")
        assert path_to_city_sequence("Amityville", path) == ["Amityville", "Hempstead"]
        assert path_cost(path) == 24.0

    def test_invalid_start(self, graph):
        """A city not in the graph should return None gracefully."""
        path, expanded = uniform_cost_search(graph, "Atlantis", "Buffalo")
        assert path is None
        assert expanded == 0


class TestAStar:
    """A* Search tests — should produce the same optimal paths as UCS."""

    def test_1hop_albany_troy(self, graph):
        path, expanded = a_star_search(graph, "Albany", "Troy")
        assert path_to_city_sequence("Albany", path) == ["Albany", "Troy"]
        assert path_cost(path) == 9.0

    def test_1hop_utica_rome(self, graph):
        path, expanded = a_star_search(graph, "Utica", "Rome")
        assert path_to_city_sequence("Utica", path) == ["Utica", "Rome"]
        assert path_cost(path) == 15.0

    def test_direct_vs_detour_buffalo_niagara(self, graph):
        path, expanded = a_star_search(graph, "Buffalo", "Niagara Falls")
        assert path_to_city_sequence("Buffalo", path) == ["Buffalo", "Niagara Falls"]
        assert path_cost(path) == 17.0

    def test_direct_binghamton_elmira(self, graph):
        path, expanded = a_star_search(graph, "Binghamton", "Elmira")
        assert path_to_city_sequence("Binghamton", path) == ["Binghamton", "Elmira"]
        assert path_cost(path) == 48.0

    def test_long_buffalo_utica(self, graph):
        path, expanded = a_star_search(graph, "Buffalo", "Utica")
        assert path_to_city_sequence("Buffalo", path) == ["Buffalo", "Rochester", "Syracuse", "Utica"]
        assert path_cost(path) == 192.0

    def test_unreachable_nyc_albany(self, graph):
        path, expanded = a_star_search(graph, "New York City", "Albany")
        assert path is None

    def test_source_equals_goal(self, graph):
        path, expanded = a_star_search(graph, "Rochester", "Rochester")
        assert path_to_city_sequence("Rochester", path) == ["Rochester"]
        assert path_cost(path) == 0.0

    def test_parallel_edges_nyc_yonkers(self, graph):
        path, expanded = a_star_search(graph, "New York City", "Yonkers")
        assert path_to_city_sequence("New York City", path) == ["New York City", "Yonkers"]
        assert path_cost(path) == 20.0

    def test_sink_node_amityville(self, graph):
        path, expanded = a_star_search(graph, "Amityville", "Hempstead")
        assert path_to_city_sequence("Amityville", path) == ["Amityville", "Hempstead"]
        assert path_cost(path) == 24.0

    def test_invalid_start(self, graph):
        path, expanded = a_star_search(graph, "Atlantis", "Buffalo")
        assert path is None
        assert expanded == 0

    def test_a_star_expands_leq_ucs(self, graph):
        """
        A* with an admissible heuristic should never expand MORE nodes
        than UCS for the same query.  We check the longest route
        (Buffalo→Utica) where the difference is most visible.
        """
        _, ucs_exp = uniform_cost_search(graph, "Buffalo", "Utica")
        _, ast_exp = a_star_search(graph, "Buffalo", "Utica")
        assert ast_exp <= ucs_exp, (
            f"A* expanded {ast_exp} nodes vs UCS {ucs_exp} — heuristic may be broken"
        )
