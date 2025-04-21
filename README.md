# Change Point Detection in Dynamic Networks

A robust framework for detecting, predicting, and explaining significant structural changes in dynamic network data using martingale-based methods with explainable AI integration.

## Overview

This project implements a comprehensive pipeline for detecting changes in the underlying structure of evolving networks. It combines statistical detection methods with prediction-enhanced algorithms to identify meaningful changes while providing interpretable results through SHAP values and visualization tools.

Key features:
- Multiple graph models (SBM, Barabási-Albert, Watts-Strogatz, Erdős-Rényi)
- Advanced martingale-based detection with various betting functions
- Prediction-enhanced detection for earlier change point identification
- Detailed analysis and visualization of detection results
- Configurable via YAML and a flexible command-line interface

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

### Command-line Interface

The framework provides an extensive CLI for overriding configuration parameters:

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
| `-net, --network` | Network type (sbm/ba/ws/er) |
| `-bf, --betting-func` | Betting function (power/exponential/mixture/constant/beta/kernel) |

#### Examples

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