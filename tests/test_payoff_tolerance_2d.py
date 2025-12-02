
import numpy as np
import pytest
from yanat.generative_game_theoric import simulate_network_evolution, shortest_path_distance
from yanat.generative_game_theoric_numba import simulate_network_evolution as simulate_numba

def test_2d_payoff_tolerance_logic():
    n_nodes = 10
    n_iterations = 20
    distance_matrix = np.random.rand(n_nodes, n_nodes)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # High tolerance group: nodes 0-4
    # Low tolerance group: nodes 5-9
    high_tol_nodes = list(range(5))
    low_tol_nodes = list(range(5, 10))
    
    # Tolerance array: (n_iterations, n_nodes)
    payoff_tolerance = np.zeros((n_iterations, n_nodes))
    payoff_tolerance[:, high_tol_nodes] = 1000.0 # Huge tolerance, effectively frozen if both are high
    payoff_tolerance[:, low_tol_nodes] = 0.0
    
    # Initial adjacency
    initial_adjacency = np.random.randint(0, 2, (n_nodes, n_nodes)).astype(float)
    initial_adjacency = np.triu(initial_adjacency, 1)
    initial_adjacency = initial_adjacency + initial_adjacency.T
    
    # Run simulation (Python)
    history_py = simulate_network_evolution(
        distance_matrix=distance_matrix,
        n_iterations=n_iterations,
        distance_fn=shortest_path_distance,
        payoff_tolerance=payoff_tolerance,
        initial_adjacency=initial_adjacency,
        alpha=1.0,
        beta=1.0,
        connectivity_penalty=0.0,
        random_seed=42
    )
    
    # Run simulation (Numba)
    history_nb = simulate_numba(
        distance_matrix=distance_matrix,
        n_iterations=n_iterations,
        distance_fn="shortest",
        payoff_tolerance=payoff_tolerance,
        initial_adjacency=initial_adjacency,
        alpha=1.0,
        beta=1.0,
        connectivity_penalty=0.0,
        random_seed=42
    )
    
    for history, name in [(history_py, "Python"), (history_nb, "Numba")]:
        print(f"Verifying {name} implementation...")
        # Check that edges between high tolerance nodes did NOT change
        # We compare history at t=0 and t=end
        adj_start = history[:, :, 0]
        adj_end = history[:, :, -1]
        
        # Extract submatrix for high tolerance nodes
        sub_start = adj_start[np.ix_(high_tol_nodes, high_tol_nodes)]
        sub_end = adj_end[np.ix_(high_tol_nodes, high_tol_nodes)]
        
        # They should be identical because neither node would accept a change (unless gain > 1000, which is unlikely)
        if not np.array_equal(sub_start, sub_end):
            diff = sub_end - sub_start
            print(f"FAILURE: High tolerance nodes changed edges in {name}!")
            print(f"Diff:\n{diff}")
            assert False, f"High tolerance nodes changed edges in {name}"
            
        # Check that edges involving low tolerance nodes DID change (likely)
        # We can't guarantee they change, but with 0 tolerance and random flips, some should change.
        # Unless the network is already optimal.
        # But we started with random adjacency.
        sub_low_start = adj_start[np.ix_(low_tol_nodes, low_tol_nodes)]
        sub_low_end = adj_end[np.ix_(low_tol_nodes, low_tol_nodes)]
        
        # It's possible they didn't change if n_iterations is small or batch_size is small.
        # But let's just check that the whole matrix is NOT identical to start (some changes happened).
        if np.array_equal(adj_start, adj_end):
            print(f"WARNING: No changes at all in {name}. Might be due to parameters.")
        else:
            print(f"SUCCESS: Changes occurred in {name}, but high-tol submatrix remained stable.")

if __name__ == "__main__":
    test_2d_payoff_tolerance_logic()
