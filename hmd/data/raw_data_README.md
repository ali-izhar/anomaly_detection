# Data Generation

This directory contains synthetic network sequences and the MIT Reality dataset used for evaluating the Horizon Martingale framework.

```bash
# Generate all data (synthetic + MIT Reality)
python src/scripts/generate_data.py --all

# Generate only synthetic networks
python src/scripts/generate_data.py --synthetic sbm er ba ws

# Process only MIT Reality dataset
python src/scripts/generate_data.py --mit data/mit_reality/Proximity.csv
```

## Synthetic Networks

Generate graph sequences with known change points for controlled experiments.

### Supported Network Types

| Network | Description | Change Type |
|---------|-------------|-------------|
| **SBM** | Stochastic Block Model | Community structure changes |
| **ER** | Erdős-Rényi | Edge probability changes |
| **BA** | Barabási-Albert | Preferential attachment parameter shifts |
| **WS** | Watts-Strogatz | Rewiring probability changes |

### Generation Commands

```bash
# Generate all 4 network types (10 sequences each)
python src/scripts/generate_data.py --synthetic sbm er ba ws -n 10

# Parallel generation
python src/scripts/generate_data.py --synthetic sbm er ba ws -n 10 -w 8
```

### Output Structure

```
data/synthetic/
├── sbm/
│   ├── sequence_000.pkl    # Full data (graphs, features, change_points)
│   ├── sequence_001.pkl
│   ├── features_000.csv    # Features for inspection
│   ├── features_001.csv
│   └── metadata.json       # Generation parameters
├── er/
│   └── ...
├── ba/
│   └── ...
└── ws/
    └── ...
```

### Data Format

Each `sequence_XXX.pkl` contains:
```python
{
    "graphs": [adj_matrix_0, adj_matrix_1, ...],  # List of numpy arrays
    "features": np.array(...),                    # Shape: (seq_len, 8)
    "change_points": [50, 100, ...],              # True change point indices
    "params": {...},                              # Generation parameters
    "seed": 42                                    # Random seed used
}
```

### Features Extracted

1. `mean_degree` - Average node degree
2. `density` - Graph density
3. `mean_clustering` - Average clustering coefficient
4. `mean_betweenness` - Average betweenness centrality
5. `mean_eigenvector` - Average eigenvector centrality
6. `mean_closeness` - Average closeness centrality
7. `max_singular_value` - Largest singular value of adjacency matrix
8. `min_nonzero_laplacian` - Smallest non-zero Laplacian eigenvalue

## MIT Reality Dataset

Real-world proximity network from MIT Media Lab (2004).

### Prerequisites

Download the MIT Reality Mining dataset:
1. Get `Proximity.csv` from the [MIT Reality Mining project](http://realitycommons.media.mit.edu/realitymining.html)

### Processing Command

```bash
# Process MIT Reality data
python src/scripts/generate_data.py --mit data/mit_reality/Proximity.csv
```

### Output Structure

```
data/mit_reality/
├── mit_reality.pkl         # Full processed data
├── metadata.json           # Dataset statistics
├── graphs/
│   ├── graph_000_2004-09-01.npy
│   ├── graph_001_2004-09-02.npy
│   └── ...
└── features/
    └── features.csv        # Daily features with dates
```

### Data Format

`mit_reality.pkl` contains:
```python
{
    "adjacency_matrices": [...],  # Daily adjacency matrices
    "features": np.array(...),    # Shape: (n_days, 8)
    "dates": ["2004-09-01", ...], # Date strings
    "n_users": 106                # Number of unique users
}
```

## Loading Data

### Python Example

```python
import pickle
import pandas as pd

# Load synthetic sequence
with open("data/synthetic/sbm/sequence_000.pkl", "rb") as f:
    data = pickle.load(f)

graphs = data["graphs"]           # List of adjacency matrices
features = data["features"]       # (seq_len, 8) array
change_points = data["change_points"]

# Load MIT Reality
with open("data/mit_reality/mit_reality.pkl", "rb") as f:
    mit = pickle.load(f)

# Load features as DataFrame
df = pd.read_csv("data/synthetic/sbm/features_000.csv")
```

## Reproducibility

Default random seeds used for synthetic data:
```python
SEEDS = [42, 142, 241, 342, 441, 542, 642, 741, 842, 1041]
```

To regenerate identical data, use the same seeds and parameters.
