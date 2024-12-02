# Change Point Detection in Dynamic Networks

Detect, Predict, and Explain significant structural changes in dynamic networks using martingale-based methods and SHAP values.

## Installation

```bash
git clone https://github.com/your-repo/anomaly_detection.git
cd anomaly_detection
pip install -r requirements.txt
```

## Module Structure

- `src/changepoint/`: Core change detection algorithms
- `src/graph/`: Graph generation and analysis
- `src/models/`: Model implementations
- `src/utils/`: Helper functions
- `experiments/`: Experiment suite

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