"""
Simplified Dataset Generator for Binary SBM Graphs
Creates sequences of Stochastic Block Model graphs with binary edges
"""

import os
import yaml
import numpy as np
import networkx as nx
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_sbm_graph(n: int, sizes: list, p_in: float, p_out: float) -> nx.Graph:
    """Generate a Stochastic Block Model graph with binary edges."""
    if sum(sizes) != n:
        raise ValueError(f"Block sizes {sizes} must sum to n={n}")
    if not (0 <= p_in <= 1 and 0 <= p_out <= 1):
        raise ValueError(f"Probabilities must be in [0,1], got p_in={p_in}, p_out={p_out}")

    # Create probability matrix
    num_blocks = len(sizes)
    p_matrix = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(p_matrix, p_in)

    return nx.stochastic_block_model(sizes, p_matrix)

def save_graph_as_edgelist(G: nx.Graph, filepath: str):
    """Save graph as edge list with binary edges."""
    with open(filepath, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v} 1\n")

class DatasetGenerator:
    def __init__(self, config_path: str):
        """Initialize with configuration file path."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']
        
        # Create output directory
        os.makedirs(self.config['output']['dir'], exist_ok=True)

    def _get_block_sizes(self) -> list:
        """Calculate block sizes based on configuration."""
        sbm_config = self.config['graph_params']['sbm']
        num_blocks = sbm_config['num_blocks']
        min_size = sbm_config['block_size_range']['min']
        max_size = sbm_config['block_size_range']['max']
        
        # Initialize with minimum sizes
        sizes = [min_size] * num_blocks
        remaining_nodes = self.config['graph_params']['nodes'] - sum(sizes)
        
        # Distribute remaining nodes randomly
        while remaining_nodes > 0:
            for i in range(num_blocks):
                if remaining_nodes <= 0:
                    break
                if sizes[i] < max_size:
                    sizes[i] += 1
                    remaining_nodes -= 1
        
        return sizes

    def _get_probabilities(self) -> tuple:
        """Get edge probabilities, possibly with changes."""
        sbm_config = self.config['graph_params']['sbm']
        p_in = sbm_config['edge_probabilities']['intra_block']
        p_out = sbm_config['edge_probabilities']['inter_block']
        
        # Randomly modify probabilities within change ranges
        if np.random.random() < 0.5:  # 50% chance of change
            p_in = np.random.uniform(
                sbm_config['change_ranges']['intra']['min'],
                sbm_config['change_ranges']['intra']['max']
            )
            p_out = np.random.uniform(
                sbm_config['change_ranges']['inter']['min'],
                sbm_config['change_ranges']['inter']['max']
            )
        
        return p_in, p_out

    def generate_sequence(self, sequence_id: int):
        """Generate a sequence of SBM graphs."""
        logger.info(f"Generating sequence {sequence_id}")
        
        for t in range(self.config['seq_len']):
            # Get block sizes and probabilities
            sizes = self._get_block_sizes()
            p_in, p_out = self._get_probabilities()
            
            # Generate graph
            G = generate_sbm_graph(
                n=self.config['graph_params']['nodes'],
                sizes=sizes,
                p_in=p_in,
                p_out=p_out
            )
            
            # Save as edge list
            filename = f"edge_list_{t}.txt"
            filepath = os.path.join(self.config['output']['dir'], filename)
            save_graph_as_edgelist(G, filepath)
            
            logger.info(f"Generated graph {t} in sequence {sequence_id} with {G.number_of_edges()} edges")

    def create_dataset(self):
        """Create the full dataset."""
        logger.info("Starting dataset generation...")
        
        for seq in range(self.config['num_sequences']):
            self.generate_sequence(seq)
            
        logger.info("Dataset generation complete")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate binary SBM graph dataset")
    parser.add_argument('--config', type=str, default='dataset_config.yaml',
                      help='Path to dataset configuration file')
    args = parser.parse_args()

    try:
        generator = DatasetGenerator(args.config)
        generator.create_dataset()
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
