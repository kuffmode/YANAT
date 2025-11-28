
import pytest
import numpy as np
import time
from yanat.generative_game_theoric_numba import (
    simulate_network_evolution, 
    propagation_distance, 
    resistance_distance, 
    shortest_path_distance,
    topological_distance,
    heat_kernel_distance,
    find_optimal_alpha
)

def test_numba_simulation_runs():
    n_nodes = 20
    n_iterations = 10
    dist_mat = np.random.rand(n_nodes, n_nodes)
    np.fill_diagonal(dist_mat, 0)
    
    # Test with default (propagation)
    history = simulate_network_evolution(
        distance_matrix=dist_mat,
        n_iterations=n_iterations,
        distance_fn="propagation",
        alpha=1.0,
        beta=0.5,
        batch_size=4,
        random_seed=42,
        verbose=False
    )
    assert history.shape == (n_nodes, n_nodes, n_iterations)
    assert np.all(np.isin(history, [0, 1]))

def test_distance_functions():
    n = 10
    adj = np.random.randint(0, 2, (n, n)).astype(float)
    np.fill_diagonal(adj, 0)
    adj = (adj + adj.T) / 2 # Symmetric
    adj[adj > 0] = 1.0
    
    dist_mat = np.random.rand(n, n)
    
    # Propagation
    d1 = propagation_distance(adj, spatial_decay=0.5)
    assert d1.shape == (n, n)
    assert not np.any(np.isnan(d1))
    
    # Resistance
    d2 = resistance_distance(adj)
    assert d2.shape == (n, n)
    
    # Shortest Path
    d3 = shortest_path_distance(adj)
    assert d3.shape == (n, n)
    
    # Heat Kernel
    d4 = heat_kernel_distance(adj, t=0.5)
    assert d4.shape == (n, n)
    
    # Topological
    d5 = topological_distance(adj)
    assert d5.shape == (n, n)

def test_simulation_with_different_metrics():
    n_nodes = 10
    dist_mat = np.random.rand(n_nodes, n_nodes)
    
    metrics = ["resistance", "heat_kernel", "shortest_path", "topological"]
    
    for metric in metrics:
        print(f"Testing {metric}...")
        history = simulate_network_evolution(
            distance_matrix=dist_mat,
            n_iterations=5,
            distance_fn=metric,
            alpha=1.0,
            beta=0.5,
            batch_size=2,
            verbose=False
        )
        assert history.shape == (n_nodes, n_nodes, 5)


def test_find_optimal_alpha():
    n_nodes = 10
    dist_mat = np.random.rand(n_nodes, n_nodes)
    emp_conn = np.random.randint(0, 2, (n_nodes, n_nodes))
    
    # Run with verbose=False
    res = find_optimal_alpha(
        distance_matrix=dist_mat,
        empirical_connectivity=emp_conn,
        distance_fn="shortest_path",
        n_iterations=10,
        max_search_iterations=2,
        verbose=False
    )
    assert 'alpha' in res
    assert 'density' in res
