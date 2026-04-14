# Uncertainty-Aware Traffic Routing Using Search-Based Algorithms

### CSCI-630 | Artificial Intelligence

---

## Project Description

Traditional route-planning systems compute a single shortest path assuming fixed and deterministic travel times. However, real-world traffic conditions are inherently uncertain due to congestion and incidents. Ignoring this uncertainty can lead to routes that are optimal on average but unreliable in practice.

This project designs and implements an AI-based traffic routing system that incorporates search algorithms and uncertainty modeling to compute routes that are both efficient and reliable. The road network is modeled as a graph over New York State, where travel times are treated as probabilistic values drawn from Gaussian distributions rather than fixed constants.

---

## Phase 1 — Dataset Construction & Data Structures

### Dataset

The dataset was manually constructed to represent a real-world road network across New York State, split into two CSV files:

- **`ny_graph_nodes.csv`** — 28 cities across NY State, each with a name, population, region, and GPS coordinates (latitude/longitude)
- **`ny_graph_edges.csv`** — 41 road connections between cities, each with distance, speed limit, road type, and time-period-specific traffic distributions

The cities span 5 geographic regions: NYC Metro, Western NY, Central NY, Capital Region, and Southern Tier.

Each road edge includes congestion multipliers with a mean and standard deviation for three time periods — AM peak, off-peak, and PM peak — to model the uncertainty in travel times.

> 📊 **View the dataset here:** [NY Traffic Graph Data](https://docs.google.com/spreadsheets/d/164fPYnmcTjsCdTGS8qHgueLM4mZT1Rs4fXORk0Mni_A/edit?usp=sharing)

### Data Structures

Four classes were implemented to represent the graph:

- **`TrafficDistribution`** — Represents the uncertainty in travel time for a given time period using a Gaussian model (mean + standard deviation). Supports both deterministic (mean) and probabilistic (sampled) travel time calculations.
- **`Edge`** — Represents a road between two cities, storing distance, speed, road type, and three `TrafficDistribution` objects for each time period.
- **`Node`** — Represents a city, storing its name, population, region, and GPS coordinates. Implements the Haversine formula to compute straight-line distances between cities, used as an admissible heuristic for A\* search.
- **`TrafficGraph`** — Represents the full road network as an undirected adjacency list graph. Loads nodes and edges from CSV files and supports neighbor lookups, node retrieval, and edge queries between cities.

The graph was loaded and verified successfully:

```
Loaded 28 nodes.
Loaded 84 edges.  ← 42 roads × 2 directions (undirected graph)
```

### Graph Visualization

The `visualize_graph` function renders the road network using real GPS coordinates, so the layout mirrors the actual geography of New York State. Edge labels show the highway name and the expected (mean) travel time for the selected time period. An optional `path` argument highlights a route in red.

---

## Phase 2 - Basic Search Algorithms

Phase 2 introduces baseline deterministic routing algorithms for the graph. In this phase, edge weights use **free-flow travel time only** (no traffic uncertainty yet), which makes this a clean reference point before risk-aware routing.

### Uniform Cost Search (UCS)

`uniform_cost_search(graph, start, goal)` computes the least-cost path using cumulative free-flow travel time.

- Cost function: sum of `edge.get_travel_time_without_traffic()`
- Guarantee: optimal path under deterministic free-flow costs
- Returns: `(path_edges, nodes_expanded)`

### A\* Search (Haversine Heuristic)

`a_star_search(graph, start, goal)` adds a heuristic based on straight-line Haversine distance between cities.

- `g(n)`: accumulated free-flow travel time
- `h(n)`: Haversine distance to goal
- `f(n) = g(n) + h(n)`
- Returns: `(path_edges, nodes_expanded)`

In practice, this reduces unnecessary expansions compared with UCS while preserving path optimality for the deterministic Phase 2 setup.

### Phase 2 Test Cases

Phase 2 tests are implemented directly in the notebook as a route-case harness (not in `pytest`).

- 5 primary route cases:
  - Albany -> Troy (parallel-edge choice)
  - Utica -> Rome (neighbor ordering / shortest direct road)
  - Buffalo -> Niagara Falls (multi-route local choice)
  - Binghamton -> Elmira (detour rejection)
  - Buffalo -> Utica (multi-hop chain benchmark)
- 2 edge cases:
  - Source == goal
  - Unreachable destination / disconnected graph component

The harness validates expected city sequence and total free-flow cost for both algorithms, and records node expansion counts.

### Phase 2 Result Summary

- All listed Phase 2 route/edge cases pass for both UCS and A\* in notebook execution.
- A\* expands fewer nodes overall than UCS.
- Reported notebook aggregate: **7 fewer nodes expanded by A\*** (**20.0% reduction**).

---

## Phase 3 - Traffic-Aware Search

Phase 3 extends the deterministic search algorithms to include time-dependent traffic costs. Instead of using only free-flow travel time, each edge cost is computed as:

```
free_flow_time_min × congestion_multiplier(time_period)
```

Each route query now accepts a time period:

- `am_peak`
- `off_peak`
- `pm_peak`

This lets the same graph produce different optimal routes depending on traffic conditions. For example, a route that is fastest during off-peak hours can become slower during AM or PM rush hour when the congestion multiplier increases.

### Traffic-Aware UCS

`uniform_cost_search_with_traffic(graph, start, goal, time_period)` keeps the same UCS structure from Phase 2, but replaces the free-flow cost with the deterministic mean travel time for the selected traffic period.

- Cost function: sum of `edge.get_expected_travel_time_with_traffic(time_period)`
- Uses mean congestion multipliers, not random samples
- Returns: `(path_edges, nodes_expanded)`

### Traffic-Aware A\*

`a_star_search_with_traffic(graph, start, goal, time_period)` applies the same traffic-aware edge cost to A\* search while continuing to use the Haversine heuristic.

- `g(n)`: accumulated congestion-adjusted travel time
- `h(n)`: Haversine distance to goal
- `f(n) = g(n) + h(n)`
- Returns: `(path_edges, nodes_expanded)`

### Phase 3 Test Cases

Phase 3 tests are implemented directly in the notebook as a traffic-aware route harness.

- Same-route, different-time-period checks:
  - New York City -> Long Island City during off-peak: `16 × 1.3 = 20.8 min`
  - New York City -> Long Island City during AM peak: `16 × 3.5 = 56.0 min`
  - New York City -> Long Island City during PM peak: `16 × 3.8 = 60.8 min`
- Path-flip checks:
  - Buffalo -> Niagara Falls during off-peak uses the direct I-190 route: `17.0 min`
  - Buffalo -> Niagara Falls during AM peak flips to Buffalo -> Tonawanda -> Niagara Falls: `(7 × 1.6) + (13 × 1.3) = 28.1 min`

The harness validates expected city sequence and congestion-adjusted route cost for both traffic-aware UCS and traffic-aware A\*, and records node expansion counts.

### Phase 3 Result Summary

- All listed Phase 3 traffic-aware route cases pass for both UCS and A\* in notebook execution.
- The Buffalo -> Niagara Falls example demonstrates that traffic can change the optimal path.
- The notebook also reports side-by-side node expansion counts for traffic-aware UCS vs traffic-aware A\*.

---

## Phase 4 - Probabilistic Traffic with Monte Carlo Simulation

Phase 4 moves from deterministic mean traffic multipliers to probabilistic traffic scenarios. Each edge's traffic multiplier is treated as a Gaussian random variable, so the best route can change from one sampled traffic scenario to another.

The Monte Carlo workflow is:

1. Sample one full traffic scenario by drawing a travel time for every edge.
2. Run UCS on that sampled scenario to choose a route.
3. Repeat the process many times and count how often each route is selected.
4. Return the most frequently selected route as the most likely route under uncertainty.
5. Run a second Monte Carlo simulation on that chosen route to estimate travel-time reliability.

This phase answers two questions:

- Which route is most often optimal under uncertain traffic?
- How reliable is that route once selected?

### Probabilistic Helper Functions

The notebook adds helper functions for the Monte Carlo workflow:

- `sample_edge_costs(graph, time_period)` samples a travel time for every directed edge in the graph.
- `uniform_cost_search_for_sampled_scenario(graph, start, goal, sampled_costs)` runs UCS using one sampled traffic scenario.
- `probabilistic_route_choice_report(graph, start, goal, time_period, num_trials, seed, threshold)` counts route-choice frequency across many sampled scenarios.
- `monte_carlo_path_simulation_report(path, time_period, num_trials, seed, threshold)` summarizes travel-time reliability for the selected route.

The report includes:

- Most common path and edge sequence
- Selection frequency
- Average nodes expanded across sampled searches
- Mean, standard deviation, minimum, and maximum travel time for the chosen path
- Optional probability that travel time exceeds a threshold
- Top route frequencies

### Phase 4 Test Cases

Phase 4 tests focus on behavioral guarantees and reproducibility instead of one exact fixed route cost, because Monte Carlo results are stochastic.

- Route validity under uncertainty:
  - Buffalo -> Niagara Falls during AM peak should most often choose Buffalo -> Tonawanda -> Niagara Falls.
  - The chosen-path Monte Carlo mean should stay in a reasonable interval near 28 minutes.
- Monte Carlo reproducibility:
  - Running the same report twice with the same random seed should produce the same route-choice summary.
- Edge cases:
  - Rochester -> Rochester during PM peak returns the trivial path with selection frequency `1.0`, mean travel time `0`, and standard deviation `0`.
  - New York City -> Albany during AM peak remains unreachable in every sampled scenario.

### Phase 4 Result Summary

- All listed Phase 4 sample tests pass in notebook execution.
- The Buffalo -> Niagara Falls AM peak case confirms that the probabilistic workflow preserves the expected route flip while still allowing sampled variation.
- Fixed seeding is used so Monte Carlo outputs are reproducible for debugging, grading, and comparison.

## Class Testing

Automated tests in `test_traffic_routing.py` currently focus on Phase 1 data structures and utilities. Because the source code is in a `.ipynb` notebook rather than a standalone `.py` module, the test file extracts and executes all code cells at collection time — no manual conversion needed.

Phase 2, Phase 3, and Phase 4 algorithm checks are currently run via notebook test cells and printed harness summaries.

Run the full suite from the project root:

```bash
python -m pytest test_traffic_routing.py -v
```

### Test Coverage

| Class                 | What's tested                                                                                                                                                                                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TrafficDistribution` | Attribute storage, `sample()` clamps to ≥ 1.0, randomness, tight-std clustering, `repr`                                                                                                                                                                              |
| `Edge`                | Free-flow time, deterministic AM/off-peak/PM times against real dataset values, off-peak ≤ peak ordering, stochastic ≥ free-flow, `ValueError` on invalid time period, `repr`                                                                                        |
| `Node`                | Self-distance = 0, Haversine value for Buffalo→Rochester (~65 mi < 74 mi road), symmetry, triangle inequality, full admissibility sweep across all 41 dataset edges, `repr`                                                                                          |
| `TrafficGraph`        | Node/edge counts (28 nodes, 82 directed edges), spot-check node attributes, `get_node` hit and miss, `get_neighbors` non-empty and typed, undirected symmetry, `get_edges_between` multiple roads and known travel time value, adjacency completeness, no self-loops |
| `reconstruct_path`    | Path reconstruction for chain and start==goal scenarios                                                                                                                                                                                                              |

---

> **Note:** AI assistance was used in generating code comments and documentation throughout this project.
