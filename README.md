# Change Point Detection in Dynamic Networks

Detect and explain significant structural changes in dynamic networks using martingale-based methods and SHAP values.

## Installation

```bash
git clone https://github.com/your-repo/anomaly_detection.git
cd anomaly_detection
pip install -r requirements.txt
```

## Quick Start

Run experiments using the main CLI:

```bash
# Run synthetic data experiments
python main.py synthetic -c config/synthetic_data.yaml

# Run reality mining analysis
python main.py reality -c config/reality_mining.yaml

# Run linear model experiments
python main.py linear -c config/linear_models.yaml
```

Available experiments:
- `synthetic`: Run synthetic data experiments
- `reality`: Analyze Reality Mining dataset
- `linear`: Run linear model experiments

Each experiment requires a corresponding YAML configuration file specified with the `-c` flag.

## Visual Analysis Examples

### Network Evolution Models
![Barabási-Albert Analysis](assets/comp_ba.png)

- Top row: Network evolution over time
- Second row: Martingale values over time (with and without detection)
- Third row: SHAP values evolution with heatmap
- Fourth row: Detected change points

### Change Point Detection Results
![SHAP Dashboard](assets/shap_dashboard.png)

The dashboard shows the sum and average martingale models for linear data defined by `config/linear_models.yaml`.

## Module Structure

- `src/changepoint/`: Core change detection algorithms
- `src/graph/`: Graph generation and analysis
- `src/models/`: Model implementations
- `src/utils/`: Helper functions
- `tests/`: Test suite

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