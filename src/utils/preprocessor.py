# src/utils/preprocessor.py

import pandas as pd
import networkx as nx
import numpy as np
import logging
from typing import Tuple, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses temporal interaction data into graph sequences.
    Transforms raw interaction records (user1, user2, time) into a sequence of
    graphs G = [G1, ..., Gt] where each G_i represents the network state at time i.
    """

    def __init__(self, file_path: str) -> None:
        """Initialize preprocessor with data source.

        Args:
            file_path (str): Path to the CSV file containing interaction data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")

        self._raw_data: Optional[pd.DataFrame] = None
        self._filtered_data: Optional[pd.DataFrame] = None
        self._filtered_dates: Optional[pd.Series] = None
        self._graphs: Optional[List[nx.Graph]] = None
        self._adj_matrices: Optional[List[np.ndarray]] = None

        logger.debug(f"Initialized preprocessor for file: {file_path}")

    def load_data(self) -> None:
        """Load raw interaction data from file."""
        try:
            logger.info(f"Loading data from {self.file_path}")
            self._raw_data = pd.read_csv(self.file_path)
            logger.info(f"Loaded {len(self._raw_data)} interactions")
            logger.debug(f"Columns: {list(self._raw_data.columns)}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise ValueError(f"Failed to load data: {str(e)}")

    def preprocess_data(
        self,
        prob_threshold: float = 0.3,
        time_col: str = "time",
        prob_col: str = "prob2",
        date_col: str = "date",
    ) -> None:
        """Filter and transform raw interaction data.
        1. Remove rows with NA probabilities
        2. Keep rows with probability greater than prob_threshold
        3. Convert time to date
        4. Drop time and probability columns
        5. Remove duplicates
        """
        if self._raw_data is None:
            logger.error("No data loaded. Call load_data() first")
            raise ValueError("No data loaded. Call load_data() first")

        required_cols = {time_col, prob_col}
        if not required_cols.issubset(self._raw_data.columns):
            logger.error(f"Missing required columns: {required_cols}")
            raise ValueError(f"Missing required columns: {required_cols}")

        try:
            logger.info(f"Starting preprocessing with threshold={prob_threshold}")

            # Filter and transform
            df = self._raw_data.copy()
            initial_rows = len(df)

            df.dropna(inplace=True, subset=[prob_col])
            logger.debug(f"Removed {initial_rows - len(df)} rows with NA probabilities")

            df = df[df[prob_col] > prob_threshold]
            logger.debug(
                f"Kept {len(df)} rows after probability threshold {prob_threshold}"
            )

            df[date_col] = pd.to_datetime(df[time_col]).dt.date
            df.drop([time_col, prob_col], axis=1, inplace=True)

            initial_unique = len(df)
            df.drop_duplicates(inplace=True, keep="first")
            logger.debug(f"Removed {initial_unique - len(df)} duplicate interactions")

            # Store results
            self._filtered_data = df.reset_index(drop=True)
            self._filtered_dates = df.drop_duplicates(
                subset=[date_col], keep="first"
            ).reset_index(drop=True)[date_col]
            self._filtered_dates.index += 1

            logger.info(
                f"Preprocessing complete: {len(self._filtered_data)} interactions across {len(self._filtered_dates)} dates"
            )

        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise ValueError(f"Data preprocessing failed: {str(e)}")

    def create_graphs(
        self,
        user_col: str = "user.id",
        remote_user_col: str = "remote.user.id.if.known",
        date_col: str = "date",
        max_nodes: Optional[int] = None,
    ) -> None:
        """Convert filtered interactions into a temporal graph sequence.
        For each unique date, create a graph from the interactions of that day as:
        G_i = (V, E_i)
        - where V is the set of users and E_i is the set of edges between users on day i.
        """
        if self._filtered_data is None:
            logger.error("No filtered data. Call preprocess_data() first")
            raise ValueError("No filtered data. Call preprocess_data() first")

        try:
            logger.info("Starting graph sequence creation")

            # Group interactions by date
            result_graphs: List[List[Tuple[int, int]]] = []
            current_date = None
            current_group: List[Tuple[int, int]] = []

            for _, row in self._filtered_data.iterrows():
                date = row[date_col]
                edge = (row[user_col], row[remote_user_col])

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

            # Convert edge lists to graphs
            self._graphs = []
            for i, edges in enumerate(result_graphs):
                G = nx.Graph()
                if max_nodes is None:
                    max_node_id = max(max(e) for e in edges)
                    max_nodes = max_node_id + 1
                G.add_nodes_from(range(1, max_nodes))
                G.add_edges_from(edges)
                self._graphs.append(G)
                logger.debug(
                    f"Graph {i+1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                )

            # Convert to adjacency matrices
            self._adj_matrices = [
                nx.to_numpy_array(G, dtype=np.int32) for G in self._graphs
            ]

            logger.info(
                f"Created {len(self._graphs)} graphs with {max_nodes-1} nodes each"
            )

        except Exception as e:
            logger.error(f"Graph creation failed: {str(e)}")
            raise RuntimeError(f"Graph creation failed: {str(e)}")

    def run(
        self,
        prob_threshold: float = 0.3,
        time_col: str = "time",
        prob_col: str = "prob2",
        date_col: str = "date",
        user_col: str = "user.id",
        remote_user_col: str = "remote.user.id.if.known",
        max_nodes: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], pd.Series]:
        """Execute complete preprocessing pipeline."""
        logger.info("Starting complete preprocessing pipeline")

        self.load_data()
        logger.debug("Data loaded successfully")

        self.preprocess_data(prob_threshold, time_col, prob_col, date_col)
        logger.debug("Data preprocessing complete")

        self.create_graphs(user_col, remote_user_col, date_col, max_nodes)
        logger.debug("Graph creation complete")

        if self._adj_matrices is None or self._filtered_dates is None:
            logger.error("Processing pipeline failed")
            raise RuntimeError("Processing pipeline failed")

        logger.info("Preprocessing pipeline completed successfully")
        return self._adj_matrices, self._filtered_dates
