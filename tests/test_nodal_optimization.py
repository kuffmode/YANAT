
import numpy as np
import torch
import sys
import os

# Add parent directory to path to import yanat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yanat.generative_game_theoric_pytorch import simulate_network_evolution_gradient_pytorch

def test_nodal_optimization():
    print("Testing Nodal Optimization...")
    
    # Create a small random distance matrix (5 nodes)
    n_nodes = 5
    np.random.seed(42)
    coords = np.random.rand(n_nodes, 2)
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    
    # Run simulation with nodal optimization
    try:
        history = simulate_network_evolution_gradient_pytorch(
            distance_matrix=dist_matrix,
            n_iterations=10,
            distance_fn="resistance",
            alpha=1.0,
            beta=1.0,
            learning_rate=0.1,
            optimization_type="nodal",
            verbose=True
        )
        print("Simulation completed successfully.")
        print("Final Adjacency Matrix shape:", history.shape)
        
        # Check if adjacency matrix is symmetric
        final_adj = history[:, :, -1]
        is_symmetric = np.allclose(final_adj, final_adj.T, atol=1e-5)
        print(f"Is symmetric: {is_symmetric}")
        
        if not is_symmetric:
            print("WARNING: Final adjacency matrix is not symmetric!")
            
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_nodal_optimization()
