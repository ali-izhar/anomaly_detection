# Change Point Detection in Dynamic Networks

A robust framework for detecting, predicting, and explaining significant structural changes in dynamic network data using martingale-based methods with explainable AI integration.

## Overview

This project implements a comprehensive pipeline for detecting changes in the underlying structure of evolving networks. It combines statistical detection methods with prediction-enhanced algorithms to identify meaningful changes while providing interpretable results through SHAP values and visualization tools.

Key features:
- Multiple graph models (SBM, Barabási-Albert, Watts-Strogatz, Erdős-Rényi)
- Advanced martingale-based detection with various betting functions
- Prediction-enhanced detection for earlier change point identification

## Installation

```bash
git clone https://github.com/your-repo/anomaly_detection.git
cd anomaly_detection
pip install -r requirements.txt
```

## Module Structure

- `src/algorithm.py`: Core pipeline for graph change point detection
- `src/changepoint/`: Martingale-based detection algorithms
- `src/graph/`: Graph generation, feature extraction, and utilities
- `src/predictor/`: Future state prediction models for graphs
- `src/configs/`: Configuration files for different detection scenarios
- `src/utils/`: Visualization, analysis, and helper functions

## Usage

### Basic Usage

```bash
python src/run.py -c src/configs/algorithm.yaml
```

CLI for overriding configuration parameters:

```bash
python src/run.py -c src/configs/algorithm.yaml [OPTIONS]
```

#### Available Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to configuration file (required) |
| `-ll, --log-level` | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `-n, --n-trials` | Number of detection trials to run |
| `-p, --prediction` | Enable (true) or disable (false) prediction |
| `-l, --threshold` | Detection threshold value |
| `-d, --distance` | Distance measure (euclidean/mahalanobis/manhattan/minkowski/cosine) |
| `-r, --reset-on-traditional` | Reset horizon martingales when traditional detection occurs (true/false) |
| `-net, --network` | Network type (sbm/ba/ws/er) |
| `-bf, --betting-func` | Betting function (power/exponential/mixture/constant/beta/kernel) |

Run with 5 trials on a Barabási-Albert network:
```bash
python src/run.py -c src/configs/algorithm.yaml -n 5 -net ba
```

Disable prediction and use a power betting function:
```bash
python src/run.py -c src/configs/algorithm.yaml -p false -bf power
```

Lower detection threshold and use Euclidean distance:
```bash
python src/run.py -c src/configs/algorithm.yaml -l 40 -d euclidean
```

Enable resetting horizon martingales on traditional detections:
```bash
python src/run.py -c src/configs/algorithm.yaml -r true
```

### Example 
```python
python src/run.py -c src/configs/algorithm.yaml -net ba -l 70 -bf power -d mahalanobis -r true
2025-04-20 21:53:26 - __main__ - INFO - Using configuration file: src/configs/algorithm.yaml
2025-04-20 21:53:26 - __main__ - INFO - Overriding network type: ba
2025-04-20 21:53:26 - __main__ - INFO - Overriding threshold: 70.0
2025-04-20 21:53:26 - __main__ - INFO - Overriding betting function: power
2025-04-20 21:53:26 - __main__ - INFO - Overriding distance measure: mahalanobis
2025-04-20 21:53:26 - __main__ - INFO - Overriding reset_on_traditional: True
2025-04-20 21:53:26 - src.graph.generator - INFO - Initialized generator for ba model
2025-04-20 21:53:26 - src.graph.generator - INFO - Generated 1 change points at: [40]
2025-04-20 21:53:44 - src.changepoint.martingale_traditional - INFO - Traditional martingale detected change at t=47: Sum=94.8952 > 70.0
2025-04-20 21:54:02 - src.changepoint.martingale_horizon - INFO - Horizon martingale detected change at t=43: Sum=72.5817 > 70.0
2025-04-20 21:54:02 - src.changepoint.martingale_horizon - INFO - Horizon martingale detected change at t=45: Sum=135.6778 > 70.0
2025-04-20 21:54:02 - src.changepoint.martingale_horizon - INFO - Horizon martingale detected change at t=47: Sum=186.9903 > 70.0
2025-04-20 21:54:31 - src.utils.output_manager - INFO - Results saved to results\ba_graph_mahalanobis_power_20250420_215326\detection_results.xlsx
2025-04-20 21:54:31 - __main__ - INFO - True change points: [40]
2025-04-20 21:54:31 - __main__ - INFO - Traditional change points detected: [47]
2025-04-20 21:54:31 - __main__ - INFO - Horizon change points detected: [43, 45, 47]
Change Point Detection Analysis
==============================

Detection Details:
╭───────────┬─────────────────────────┬─────────────────┬─────────────────────┬─────────────────┬───────────────────╮
│   True CP │   Traditional Detection │   Delay (steps) │   Horizon Detection │   Delay (steps) │ Delay Reduction   │
├───────────┼─────────────────────────┼─────────────────┼─────────────────────┼─────────────────┼───────────────────┤
│        40 │                      47 │               7 │                  43 │               3 │ 57.1%             │
╰───────────┴─────────────────────────┴─────────────────┴─────────────────────┴─────────────────┴───────────────────╯

Summary Statistics:
╭─────────────────────┬───────────────┬───────────╮
│ Metric              │ Traditional   │ Horizon   │
├─────────────────────┼───────────────┼───────────┤
│ Detection Rate      │ 100.0%        │ 100.0%    │
├─────────────────────┼───────────────┼───────────┤
│ Average Delay       │ 7.0           │ 3.0       │
├─────────────────────┼───────────────┼───────────┤
│ Avg Delay Reduction │               │ 57.1%     │
╰─────────────────────┴───────────────┴───────────╯
```

## Algorithm Overview

The detection pipeline consists of several key components:

1. **Graph Sequence Generation**: Creates a sequence of evolving graphs with predefined change points
2. **Feature Extraction**: Extracts topological features from each graph in the sequence
3. **Future State Prediction**: Predicts future graph states to enhance detection (optional)
4. **Change Point Detection**: Applies martingale-based detection methods
5. **Visualization & Analysis**: Generates research-quality visualizations and numerical analysis

## Data Sources

- [Synthetic Graph Data](src/config/synthetic_data_config.yaml)
- [MIT Reality Mining Dataset](https://realitycommons.media.mit.edu/realitymining.html)

## References

1. Ho, S. S., et al. (2005). "A martingale framework for concept change detection in time-varying data streams." ICML.
2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.
3. Newman, M. E. J. (2010). "Networks: An Introduction." Oxford University Press.

## Contributing

We welcome contributions to improve the project. Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.