
import pytest
import numpy as np
import torch
from yanat.generative_game_theoric_pytorch import (
    resistance_distance_pt,
    heat_kernel_distance_pt,
    simulate_network_evolution_gradient_pytorch,
    find_optimal_alpha_pt,
)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_resistance_distance_pt(device):
    adj_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    adj_torch = torch.tensor(adj_np, device=device, dtype=torch.float64)
    
    dist = resistance_distance_pt(adj_torch)
    
    assert dist.shape == (3, 3)
    assert not torch.any(torch.isnan(dist))
    assert torch.all(dist >= 0)

def test_heat_kernel_distance_pt(device):
    adj_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    adj_torch = torch.tensor(adj_np, device=device, dtype=torch.float64)
    
    dist = heat_kernel_distance_pt(adj_torch, t=0.5)
    
    assert dist.shape == (3, 3)
    assert not torch.any(torch.isnan(dist))

def test_simulate_network_evolution_gradient_pytorch():
    n_nodes = 10
    n_iterations = 10
    dist_mat = np.random.rand(n_nodes, n_nodes)
    np.fill_diagonal(dist_mat, 0)
    
    history = simulate_network_evolution_gradient_pytorch(
        distance_matrix=dist_mat,
        n_iterations=n_iterations,
        distance_fn="resistance",
        alpha=1.0,
        beta=0.5,
        learning_rate=0.01,
        random_seed=42,
        verbose=False
    )
    
    assert history.shape == (n_nodes, n_nodes, n_iterations)
    assert not np.any(np.isnan(history))

def test_simulate_with_heat_kernel():
    n_nodes = 10
    dist_mat = np.random.rand(n_nodes, n_nodes)
    
    history = simulate_network_evolution_gradient_pytorch(
        distance_matrix=dist_mat,
        n_iterations=5,
        distance_fn="heat_kernel",
        alpha=1.0,
        beta=0.5,
        verbose=False,
        t=0.5
    )
    assert history.shape == (n_nodes, n_nodes, 5)

def test_find_optimal_alpha_pt():
    n_nodes = 10
    dist_mat = np.random.rand(n_nodes, n_nodes)
    emp_conn = np.random.randint(0, 2, (n_nodes, n_nodes))
    
    res = find_optimal_alpha_pt(
        distance_matrix=dist_mat,
        empirical_connectivity=emp_conn,
        distance_fn="resistance",
        n_iterations=10,
        max_search_iterations=2,
        verbose=False
    )
    
    assert 'alpha' in res
    assert 'density' in res
    assert res['evolution'].shape == (n_nodes, n_nodes, 10)
