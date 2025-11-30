
import numpy as np
import torch
import yanat.generative_game_theoric_pytorch as ggt
import matplotlib.pyplot as plt

def main():
    print("Generating synthetic data...")
    n_nodes = 50
    
    # 1. Create a random distance matrix
    # In a real scenario, this would be Euclidean distance between brain regions
    coords = np.random.rand(n_nodes, 3)
    dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    
    # 2. Create a synthetic "empirical" connectivity matrix
    # Let's say the ground truth was generated with alpha=2.5
    print("Simulating ground truth network with alpha=2.5...")
    true_alpha = 2.5
    # We use the non-differentiable simulation to generate "ground truth" binary network
    # or just use the differentiable one and threshold it.
    # Let's use a simple thresholded distance model for the "empirical" target to make it easy
    # Or better, use the simulation itself.
    
    # We need to use the library function to simulate.
    # Since we are outside the library, we can use the public API.
    
    # For the sake of the example, let's just create a random sparse matrix as "empirical"
    # that has some structure.
    empirical_connectivity = (np.random.rand(n_nodes, n_nodes) < 0.1).astype(float)
    empirical_connectivity = np.triu(empirical_connectivity, 1)
    empirical_connectivity = empirical_connectivity + empirical_connectivity.T
    
    target_density = np.sum(empirical_connectivity) / (n_nodes * (n_nodes - 1))
    print(f"Target Density: {target_density:.4f}")

    print("\nStarting Gradient-Based Alpha Optimization...")
    print("This method optimizes alpha by differentiating through the simulation steps.")
    
    # 3. Run the optimization
    # We start with an initial guess of alpha=1.0
    result = ggt.find_optimal_alpha_differentiable(
        distance_matrix=dist_matrix,
        empirical_connectivity=empirical_connectivity,
        distance_fn="resistance", # or "heat"
        n_iterations=100,         # Steps of network evolution per alpha update
        alpha_search_iterations=100, # Gradient descent steps for alpha
        initial_alpha=0.01,
        learning_rate_alpha=0.1,  # Learning rate for alpha
        learning_rate_network=0.01, # Learning rate for network evolution
        verbose=True,
        metric="mse", # Use MSE metric
        optimizer_class="sgd", # Use SGD for alpha
        stochastic=True, # Use random initialization for network simulation
        momentum=0.9 # Momentum for network simulation (passed via kwargs)
    )
    
    print("\nOptimization Complete!")
    print("-" * 30)
    print(f"Optimal Alpha Found: {result['alpha']:.4f}")
    print(f"Resulting Density:   {result['density']:.4f}")
    print(f"Target Density:      {target_density:.4f}")
    print("-" * 30)
    
    # Plot convergence if matplotlib is available
    try:
        history = result['history']
        alphas = [h['alpha'] for h in history]
        losses = [h['loss'] for h in history]
        densities = [h['density'] for h in history]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Alpha', color=color)
        ax1.plot(alphas, color=color, linewidth=2, label='Alpha')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Loss (Density MSE)', color=color)
        ax2.plot(losses, color=color, linestyle='--', label='Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Alpha Optimization Convergence")
        fig.tight_layout()
        plt.savefig("alpha_optimization_convergence.png")
        print("Convergence plot saved to 'alpha_optimization_convergence.png'")
    except ImportError:
        print("Matplotlib not found, skipping plot.")

if __name__ == "__main__":
    main()
