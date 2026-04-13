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
- **`Node`** — Represents a city, storing its name, population, region, and GPS coordinates. Implements the Haversine formula to compute straight-line distances between cities, used as an admissible heuristic for A* search.
- **`TrafficGraph`** — Represents the full road network as an undirected adjacency list graph. Loads nodes and edges from CSV files and supports neighbor lookups, node retrieval, and edge queries between cities.

The graph was loaded and verified successfully:
```
Loaded 28 nodes.
Loaded 84 edges.  ← 42 roads × 2 directions (undirected graph)
```

### Graph Visualization

The `visualize_graph` function renders the road network using real GPS coordinates, so the layout mirrors the actual geography of New York State. Edge labels show the highway name and the expected (mean) travel time for the selected time period. An optional `path` argument highlights a route in red.

```python
visualize_graph(graph, path=None, time_period='off_peak')
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph` | `TrafficGraph` | — | A loaded `TrafficGraph` instance |
| `path` | `list[str]` or `None` | `None` | Optional list of city names to highlight in red |
| `time_period` | `str` | `'off_peak'` | One of `'am_peak'`, `'off_peak'`, or `'pm_peak'` — controls which traffic distribution is used for displayed travel times |

---

## Search Algorithms

### Uniform Cost Search

`uniform_cost_search(graph, start, goal)` expands nodes in order of cumulative path cost using a min-heap priority queue, guaranteeing the optimal (lowest-cost) path. Visited nodes are tracked with their best known cost and parent pointer to support path reconstruction.

### A\* Search

`a_star_search(graph, start, goal)` augments UCS with the Haversine straight-line distance to the goal as a heuristic, guiding expansion toward the destination and reducing nodes explored. Because Haversine distance is always a lower bound on true road distance, the heuristic is **admissible**, preserving optimality.

### Path Reconstruction

`reconstruct_path(visited, start, goal)` traces the parent-pointer chain stored during search to recover the full city-by-city route from start to goal.

---

## Testing

Tests live in `test_traffic_routing.py` and cover all four data structure classes. Because the source code is in a `.ipynb` notebook rather than a standalone `.py` module, the test file extracts and executes all code cells at collection time — no manual conversion needed.

Run the full suite from the project root:

```bash
python -m pytest test_traffic_routing.py -v
```

### Test Coverage

| Class | What's tested |
|---|---|
| `TrafficDistribution` | Attribute storage, `sample()` clamps to ≥ 1.0, randomness, tight-std clustering, `repr` |
| `Edge` | Free-flow time, deterministic AM/off-peak/PM times against real dataset values, off-peak ≤ peak ordering, stochastic ≥ free-flow, `ValueError` on invalid time period, `repr` |
| `Node` | Self-distance = 0, Haversine value for Buffalo→Rochester (~65 mi < 74 mi road), symmetry, triangle inequality, full admissibility sweep across all 41 dataset edges, `repr` |
| `TrafficGraph` | Node/edge counts (28 nodes, 82 directed edges), spot-check node attributes, `get_node` hit and miss, `get_neighbors` non-empty and typed, undirected symmetry, `get_edges_between` multiple roads and known travel time value, adjacency completeness, no self-loops |

---

> **Note:** AI assistance was used in generating code comments and documentation throughout this project.
