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
- 3 edge cases:
  - Source == goal
  - Multiple parallel edges with deterministic tie handling
  - Alternate local pair consistency

The harness validates expected city sequence and total free-flow cost for both algorithms, and records node expansion counts.

### Phase 2 Result Summary

- All listed Phase 2 route/edge cases pass for both UCS and A\* in notebook execution.
- A\* expands fewer nodes overall than UCS.
- Reported notebook aggregate: **7 fewer nodes expanded by A\*** (**20.0% reduction**).

## Class Testing

Automated tests in `test_traffic_routing.py` currently focus on Phase 1 data structures and utilities. Because the source code is in a `.ipynb` notebook rather than a standalone `.py` module, the test file extracts and executes all code cells at collection time — no manual conversion needed.

Phase 2 algorithm checks (UCS and A\*) are currently run via notebook test cells and printed harness summaries.

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
