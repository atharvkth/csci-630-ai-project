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

---

> **Note:** AI assistance was used in generating code comments and documentation throughout this project.