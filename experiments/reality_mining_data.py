# experiments/reality_mining_data.py

"""Reality Mining data analysis module for change point detection.

Implements a temporal network analysis pipeline for the Reality Mining dataset:
1. Data preprocessing and graph construction
2. Feature extraction (centralities, embeddings)
3. Change point detection using martingales
4. SHAP-based interpretability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap  # type: ignore
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional

from config.config import load_config
from src.changepoint import ChangePointDetector
from src.graph import GraphGenerator
from src.models import CustomThresholdModel
from src.utils import RealityMiningEvaluator, get_logger

logger = get_logger(__name__)


class RealityMiningDataPipeline:
    """Analysis pipeline for Reality Mining temporal network data."""

    def __init__(self, config_path: str) -> None:
        """Initialize pipeline with configuration."""
        if not Path(config_path).exists():
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")
        self.config = load_config(config_path)
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize pipeline components and storage."""
        logger.debug("Initializing pipeline components")

        # Data components
        self.raw_data: Optional[pd.DataFrame] = None
        self.filtered_data: Optional[pd.DataFrame] = None
        self.filtered_dates: Optional[pd.Series] = None

        # Graph components
        self.graphs: List[nx.Graph] = []
        self.adj_matrices: List[np.ndarray] = []
        self.centralities: Dict[str, List[List[float]]] = {}

        # Analysis components
        self.cpd: Optional[ChangePointDetector] = None
        self.events: Dict[str, Dict[str, List[List[float]]]] = {}
        self.graph_generator = GraphGenerator()

        # Synthetic comparison components
        self.graphs_ba_p: Optional[List[np.ndarray]] = None
        self.graphs_ba_n: Optional[List[np.ndarray]] = None
        self.cpd_ba_p: Optional[ChangePointDetector] = None
        self.cpd_ba_n: Optional[ChangePointDetector] = None

        # Evaluation component
        self.evaluator = RealityMiningEvaluator(
            trials=self.config.analysis.parameters.trials,
            ground_truth=self.config.analysis.parameters.change_point,
            thresholds=self.config.analysis.thresholds,
        )

        logger.info("Pipeline components initialized")

    def load_data(self) -> None:
        """Load raw interaction data from the configured path."""
        try:
            data_path = Path(self.config.paths.data.proximity_file)
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")

            logger.info(f"Loading data from {data_path}")
            self.raw_data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.raw_data)} interactions")
            logger.debug(f"Columns: {list(self.raw_data.columns)}")

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

    def preprocess_data(self) -> None:
        """Filter and transform raw interaction data.

        Process:
        1. Remove low probability interactions (p < threshold).
        2. Convert timestamps to dates.
        3. Remove duplicates.
        4. Sort chronologically.

        Raises:
            ValueError: If raw data is not loaded.
        """
        if self.raw_data is None:
            logger.error("No data loaded. Call load_data() first")
            raise ValueError("No data loaded. Call load_data() first")

        try:
            logger.info("Starting data preprocessing")

            # Apply filters
            self.filtered_data = self.raw_data.copy()
            initial_rows = len(self.filtered_data)

            self.filtered_data.dropna(inplace=True, subset=["prob2"])
            logger.debug(
                f"Removed {initial_rows - len(self.filtered_data)} rows with NA probabilities"
            )

            prob_threshold = self.config.data_processing.probability_threshold
            self.filtered_data = self.filtered_data[
                self.filtered_data.prob2 > prob_threshold
            ]
            logger.debug(
                f"Kept {len(self.filtered_data)} rows after probability threshold {prob_threshold}"
            )

            # Transform timestamps
            self.filtered_data["time"] = pd.to_datetime(self.filtered_data["time"])
            self.filtered_data["date"] = self.filtered_data.time.dt.date
            self.filtered_data.drop(["time", "prob2"], axis=1, inplace=True)

            # Clean and sort
            initial_unique = len(self.filtered_data)
            self.filtered_data.drop_duplicates(inplace=True, keep="first")
            logger.debug(
                f"Removed {initial_unique - len(self.filtered_data)} duplicate interactions"
            )

            self.filtered_data.reset_index(drop=True, inplace=True)

            # Extract unique dates
            self.filtered_dates = self.filtered_data.drop_duplicates(
                subset=["date"], keep="first"
            ).reset_index(drop=True)["date"]
            self.filtered_dates.index += 1

            if self.config.data_processing.save_intermediate:
                output_dir = Path(self.config.paths.output.dir)
                self.filtered_data.to_csv(output_dir / "filtered_data.csv", index=False)
                pd.DataFrame(self.filtered_dates).to_csv(
                    output_dir / "filtered_dates.csv", index=False
                )
                logger.info("Saved intermediate data files")

            logger.info(
                f"Preprocessing complete: {len(self.filtered_data)} interactions across {len(self.filtered_dates)} dates"
            )

        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise ValueError(f"Data preprocessing failed: {str(e)}")

    def create_graphs(self) -> None:
        """Convert filtered interactions into a temporal graph sequence."""
        if self.filtered_data is None:
            logger.error("No filtered data. Call preprocess_data() first")
            raise ValueError("No filtered data. Call preprocess_data() first")

        try:
            logger.info("Creating temporal graph sequence")
            result_graphs = []
            current_date = None
            current_group = []

            # Group edges by date
            for _, row in self.filtered_data.iterrows():
                date = row["date"]
                edge = (row["user.id"], row["remote.user.id.if.known"])

                if date != current_date:
                    if current_group:
                        result_graphs.append(current_group)
                    current_group = [edge]
                    current_date = date
                else:
                    current_group.append(edge)

            if current_group:
                result_graphs.append(current_group)

            logger.debug(f"Created {len(result_graphs)} edge lists")

            # Create graphs with fixed node set
            self.graphs = []
            for i, group in enumerate(result_graphs):
                G = nx.Graph()
                G.add_nodes_from(range(1, 80))  # Fixed node set
                G.add_edges_from(group)
                self.graphs.append(G)
                logger.debug(
                    f"Graph {i+1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                )

            logger.info(f"Created {len(self.graphs)} temporal graphs")

            # Verify edge count
            total_edges = sum(len(group) for group in result_graphs)
            if total_edges != len(self.filtered_data):
                logger.error("Edge count mismatch in graph creation")
                raise ValueError("Edge count mismatch in graph creation")

        except Exception as e:
            logger.error(f"Graph creation failed: {str(e)}")
            raise RuntimeError(f"Graph creation failed: {str(e)}")

    def create_adjacency_matrices(self) -> None:
        """Convert graph sequence to adjacency matrix representation."""
        if not self.graphs:
            logger.error("No graphs created. Call create_graphs() first")
            raise ValueError("No graphs created. Call create_graphs() first")

        try:
            logger.info("Converting graphs to adjacency matrices")
            self.adj_matrices = [
                nx.to_numpy_array(graph, dtype=int) for graph in self.graphs
            ]
            logger.info(f"Created {len(self.adj_matrices)} adjacency matrices")
            logger.debug(f"Matrix shape: {self.adj_matrices[0].shape}")

        except Exception as e:
            logger.error(f"Adjacency matrix creation failed: {str(e)}")
            raise RuntimeError(f"Adjacency matrix creation failed: {str(e)}")

    def extract_centralities(self) -> None:
        """Extract topological features from the graph sequence."""
        if not self.adj_matrices:
            logger.error(
                "No adjacency matrices. Call create_adjacency_matrices() first"
            )
            raise ValueError(
                "No adjacency matrices. Call create_adjacency_matrices() first"
            )

        try:
            logger.info("Extracting centralities and embeddings")
            self.cpd = ChangePointDetector()
            self.cpd.initialize(self.adj_matrices)
            self.centralities = self.cpd.extract_features()

            # Log feature dimensions
            for name, centrality in self.centralities.items():
                logger.info(
                    f"{name.upper()} Centrality/Embedding Shapes: {len(centrality), len(centrality[0])}"
                )

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}")

    def define_events(self) -> None:
        """Define time windows for event analysis."""
        try:
            logger.info("Defining event windows")
            self.events = {
                "NewYear": {k: v[81:141] for k, v in self.centralities.items()},
                "ColumbusDay": {k: v[1:60] for k, v in self.centralities.items()},
            }
            logger.debug("NewYear window: days 81-141")
            logger.debug("ColumbusDay window: days 1-60")
            logger.info("Event windows defined")

        except Exception as e:
            logger.error(f"Event definition failed: {str(e)}")
            raise RuntimeError(f"Event definition failed: {str(e)}")

    def analyze_event(self, event_name: str) -> None:
        """Analyze structural changes during a specific event."""
        if event_name not in self.events:
            logger.error(f"Unknown event: {event_name}")
            raise ValueError(f"Unknown event: {event_name}")

        try:
            logger.info(f"Analyzing {event_name} event")
            event_data = self.events[event_name]

            # Compute martingales
            logger.debug("Computing martingales for each feature")
            martingales = {}
            for k, v in event_data.items():
                martingales[k] = self.cpd.martingale_test(
                    v, self.config.analysis.thresholds[-1], reset=False
                )

            # Calculate aggregates - convert to numpy arrays first
            martingale_arrays = [
                np.array(m["martingales"]) for m in martingales.values()
            ]
            Msum = np.sum(martingale_arrays, axis=0)
            Mavg = np.mean(martingale_arrays, axis=0)
            logger.debug(f"Maximum martingale sum: {np.max(Msum)}")

            # Prepare for SHAP analysis
            logger.info("Computing SHAP values")
            # Stack the martingale sequences as features
            X = np.vstack([m["martingales"] for m in martingales.values()]).T
            model = CustomThresholdModel(threshold=self.config.analysis.thresholds[-1])
            # Convert predictions to binary numpy array
            y = np.ones(X.shape[0])
            model.fit(X, y)

            # Compute SHAP values
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = np.array(
                    shap_values[0]
                )  # Take first class for binary classification
            logger.debug(f"SHAP values shape: {shap_values.shape}")

            # Set date range for plotting
            if event_name == "NewYear":
                start_date = pd.to_datetime("2008-12-01")
                end_date = pd.to_datetime("2009-01-31")
            else:  # ColumbusDay
                start_date = pd.to_datetime("2008-09-01")
                end_date = pd.to_datetime("2008-10-31")

            dates = pd.date_range(start=start_date, end=end_date, periods=len(Msum))
            logger.debug(f"Analysis period: {start_date.date()} to {end_date.date()}")

            # Create visualization
            self.plot_event_analysis(
                event_name, martingales, Msum, Mavg, shap_values, dates
            )
            logger.info(f"Completed {event_name} event analysis")

        except Exception as e:
            logger.error(f"Event analysis failed: {str(e)}")
            raise RuntimeError(f"Event analysis failed: {str(e)}")

    def plot_event_analysis(
        self,
        event_name: str,
        martingales: Dict,
        Msum: np.ndarray,
        Mavg: np.ndarray,
        shap_values: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> None:
        """Create visualization dashboard for event analysis."""
        try:
            logger.info(f"Creating visualization for {event_name} event")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

            # Plot martingales
            logger.debug("Plotting martingale sequences")
            ax1.plot(dates, Msum, color="cyan", label="Martingale Sum", linewidth=2.5)
            ax1.plot(dates, Mavg, color="teal", label="Martingale Avg", linewidth=2.5)

            colors = ["orange", "green", "red", "purple", "brown", "gray"]
            for (k, v), color in zip(martingales.items(), colors):
                ax1.plot(dates, v["martingales"], color=color, label=k, linewidth=2.5)
                logger.debug(f"Added {k} martingale sequence")

            ax1.set_title(f"Martingales for {event_name} Event", fontsize=15)
            ax1.set_xlabel("Date", fontsize=20)
            ax1.set_ylabel("Martingale", fontsize=20)
            ax1.legend(fontsize=15)
            ax1.tick_params(axis="x", rotation=90, labelsize=5)
            ax1.tick_params(axis="y", labelsize=12)

            # Plot SHAP values
            logger.debug("Plotting SHAP values")
            ax2.set_title(
                "SHAP Values with Kernel Explanation on Martingale Custom Threshold Model",
                fontsize=15,
            )
            ax2.set_xlabel("Date", fontsize=20)
            ax2.set_ylabel("SHAP Values", fontsize=20)

            for i, (k, color) in enumerate(zip(martingales.keys(), colors)):
                ax2.plot(dates, shap_values[:, i], color=color, label=k, linewidth=2.5)
                logger.debug(f"Added SHAP values for {k}")

            ax2.legend(fontsize=15)
            ax2.tick_params(axis="x", rotation=90, labelsize=5)
            ax2.tick_params(axis="y", labelsize=12)

            # Format date axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))

            plt.tight_layout()

            # Save if configured
            if self.config.visualization.save_plots:
                output_dir = Path(self.config.paths.output.dir)
                save_path = output_dir / f"{event_name}_analysis.png"
                plt.savefig(save_path)
                logger.info(f"Saved analysis plot to {save_path}")

            plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Event visualization failed: {str(e)}")
            raise RuntimeError(f"Event visualization failed: {str(e)}")

    def evaluate_results(self) -> None:
        """Evaluate change detection performance on real data."""
        try:
            logger.info("Evaluating results for original data")
            results_lsvd = self.evaluator.evaluate(
                self.cpd,
                self.cpd,
                self.adj_matrices,
                self.adj_matrices,  # Using same matrices for null case
                feature="lsvd",
            )

            logger.info("LSVD Evaluation Results:")
            logger.info(f"\n{results_lsvd}")

            # Save results if configured
            output_dir = Path(self.config.paths.output.dir)
            results_lsvd.to_csv(output_dir / "lsvd_evaluation_results.csv", index=False)
            logger.info("Saved evaluation results")

        except Exception as e:
            logger.error(f"Results evaluation failed: {str(e)}")
            raise RuntimeError(f"Results evaluation failed: {str(e)}")

    def generate_ba_graphs(self) -> None:
        """Generate Barabási-Albert graphs for comparison."""
        try:
            logger.info("Generating Barabasi-Albert comparison graphs")

            # Generate positive case
            logger.debug("Generating positive case graphs")
            self.graphs_ba_p = self.graph_generator.barabasi_albert(
                n=self.config.graph.barabasi_albert.n,
                m1=self.config.graph.barabasi_albert.edges.initial,
                m2=self.config.graph.barabasi_albert.edges.changed,
                set1=self.config.graph.barabasi_albert.set1,
                set2=self.config.graph.barabasi_albert.set2,
            )

            # Generate null case
            logger.debug("Generating null case graphs")
            self.graphs_ba_n = self.graph_generator.barabasi_albert(
                n=self.config.graph.barabasi_albert.n,
                m1=self.config.graph.barabasi_albert.edges.null_model.initial,
                m2=self.config.graph.barabasi_albert.edges.null_model.changed,
                set1=self.config.graph.barabasi_albert.set1,
                set2=self.config.graph.barabasi_albert.set2,
            )

            logger.info(
                f"Generated {len(self.graphs_ba_p)} positive and "
                f"{len(self.graphs_ba_n)} null BA graphs"
            )

        except Exception as e:
            logger.error(f"BA graph generation failed: {str(e)}")
            raise RuntimeError(f"BA graph generation failed: {str(e)}")

    def setup_ba_detectors(self) -> None:
        """Initialize change detectors for BA graphs."""
        if self.graphs_ba_p is None or self.graphs_ba_n is None:
            logger.error("BA graphs not generated. Call generate_ba_graphs() first")
            raise ValueError("BA graphs not generated. Call generate_ba_graphs() first")

        try:
            logger.info("Setting up BA change point detectors")
            self.cpd_ba_p = ChangePointDetector()
            self.cpd_ba_n = ChangePointDetector()

            self.cpd_ba_p.initialize(self.graphs_ba_p)
            self.cpd_ba_n.initialize(self.graphs_ba_n)
            logger.info("BA detectors initialized")

        except Exception as e:
            logger.error(f"BA detector setup failed: {str(e)}")
            raise RuntimeError(f"BA detector setup failed: {str(e)}")

    def evaluate_ba_results(self) -> None:
        """Evaluate change detection performance on BA graphs."""
        if self.cpd_ba_p is None or self.cpd_ba_n is None:
            logger.error("BA detectors not setup. Call setup_ba_detectors() first")
            raise ValueError("BA detectors not setup. Call setup_ba_detectors() first")

        try:
            logger.info("Evaluating Barabási-Albert results")
            results_ba = self.evaluator.evaluate(
                self.cpd_ba_p,
                self.cpd_ba_n,
                self.graphs_ba_p,
                self.graphs_ba_n,
                feature="lsvd",
            )

            logger.info("Barabasi-Albert Evaluation Results:")
            logger.info(f"\n{results_ba}")

            # Save results if configured
            output_dir = Path(self.config.paths.output.dir)
            results_ba.to_csv(output_dir / "ba_evaluation_results.csv", index=False)
            logger.info("Saved BA evaluation results")

        except Exception as e:
            logger.error(f"BA evaluation failed: {str(e)}")
            raise RuntimeError(f"BA evaluation failed: {str(e)}")

    def run(self) -> None:
        """Execute the complete analysis pipeline."""
        try:
            logger.info("Starting Reality Mining analysis pipeline")

            # Real data analysis
            self.load_data()
            self.preprocess_data()
            self.create_graphs()
            self.create_adjacency_matrices()
            self.extract_centralities()
            self.define_events()

            # Event analysis
            self.analyze_event("NewYear")
            self.analyze_event("ColumbusDay")
            self.evaluate_results()

            # Synthetic comparison
            self.generate_ba_graphs()
            self.setup_ba_detectors()
            self.evaluate_ba_results()

            logger.info("Analysis pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise RuntimeError(f"Pipeline execution failed: {str(e)}")
