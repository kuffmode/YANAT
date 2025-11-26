import pytest
import numpy as np
from yanat.generative import (
    jit_safe,
    _diag_indices,
    _set_diagonal,
    process_matrix,
    validate_parameters,
    get_param_value,
    compute_component_sizes,
    propagation_distance,
    resistance_distance,
    heat_kernel_distance,
    shortest_path_distance,
    topological_distance,
    compute_node_payoff,
    simulate_network_evolution,
    find_optimal_alpha,
)


def test_jit_safe():
    @jit_safe()
    def add(x, y):
        return x + y

    assert add(2, 3) == 5


def test_diag_indices():
    rows, cols = _diag_indices(3)
    assert np.array_equal(rows, np.array([0, 1, 2]))
    assert np.array_equal(cols, np.array([0, 1, 2]))


def test_set_diagonal():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
    result = _set_diagonal(matrix.copy())
    assert np.array_equal(result, expected)

    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = np.array([[2, 2, 3], [4, 2, 6], [7, 8, 2]])
    result = _set_diagonal(matrix.copy(), 2)
    assert np.array_equal(result, expected)


def test_process_matrix():
    matrix = np.array(
        [[1.0, np.nan, np.inf], [-np.inf, 2.0, 3.0], [4.0, 5.0, 6.0]]
    )
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = process_matrix(matrix)
    assert np.array_equal(result, expected)


def test_validate_parameters():
    # Test correct inputs
    validate_parameters(
        1000,
        1.0,
        np.ones(1000),
        np.ones(1000),
        np.ones(1000),
        np.ones(1000) * 2,
        names=("alpha", "beta", "noise", "connectivity", "batch_size"),
        allow_float=(True, False, False, True, True),
        allow_zero=(False, False, True, False, False),
    )

    # Test incorrect inputs
    with pytest.raises(ValueError):
        validate_parameters(
            1000,
            np.ones(1000),
            1.0,
            np.ones(1000),
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
        )
    with pytest.raises(ValueError):
        validate_parameters(
            1000,
            0.0,
            np.ones(1000),
            np.ones(1000),
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
        )
    with pytest.raises(ValueError):
        validate_parameters(
            1000,
            1.0,
            np.zeros(1000),
            np.ones(1000),
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
        )
    with pytest.raises(ValueError):
        validate_parameters(
            1000,
            1.0,
            np.ones(100),
            np.ones(1000),
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
        )
    with pytest.raises(ValueError):
        validate_parameters(
            1000,
            "hello",
            np.ones(1000),
            np.ones(1000),
            np.ones(1000),
            np.ones(1000) * 2,
            names=("alpha", "beta", "noise", "connectivity", "batch_size"),
            allow_float=(True, False, False, True, True),
            allow_zero=(False, False, True, False, False),
        )
    validate_parameters(
        1000,
        1.0,
        np.ones(1000),
        None,
        np.ones(1000),
        np.ones(1000) * 2,
        names=("alpha", "beta", "noise", "connectivity", "batch_size"),
        allow_float=(True, False, False, True, True),
        allow_zero=(False, False, True, False, False),
        allow_none=(False, False, True, False, False),
    )


def test_get_param_value():
    assert get_param_value(1.0, 0) == 1.0
    assert get_param_value(np.array([1.0, 2.0, 3.0]), 1) == 2.0


def test_compute_component_sizes():
    adjacency = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    expected = np.array([2.0, 2.0, 1.0])
    result = compute_component_sizes(adjacency)
    assert np.array_equal(result, expected)


def test_propagation_distance():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = propagation_distance(adjacency)
    assert result.shape == (3, 3)
    assert np.all(result >= 0)


def test_resistance_distance():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # coordinates removed
    result = resistance_distance(adjacency)
    assert result.shape == (3, 3)
    assert np.all(result >= 0)


def test_heat_kernel_distance():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = heat_kernel_distance(adjacency)
    assert result.shape == (3, 3)
    assert np.all(result >= 0)
    result = heat_kernel_distance(adjacency, normalize=True)
    assert result.shape == (3, 3)
    assert np.all(result >= 0)


def test_shortest_path_distance():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = shortest_path_distance(adjacency)
    assert result.shape == (3, 3)
    assert np.all(result >= 0)
    assert np.all(result < np.inf)


def test_topological_distance():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = topological_distance(adjacency)
    assert result.shape == (3, 3)
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_compute_node_payoff():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    distance_matrix = np.array([
        [0.0, 1.0, 1.414],
        [1.0, 0.0, 1.0],
        [1.414, 1.0, 0.0]
    ])
    distance_fn = shortest_path_distance

    # Test with distance_matrix
    result = compute_node_payoff(
        0, adjacency, distance_matrix, distance_fn, 1.0, 1.0, 0.0, 1.0
    )
    assert isinstance(result, float)
    
    # Test with kwargs
    result_kwargs = compute_node_payoff(
        0, adjacency, distance_matrix, distance_fn, 1.0, 1.0, 0.0, 1.0,
        distance_fn_kwargs={'dummy': 1}
    )
    assert isinstance(result_kwargs, float)


def test_simulate_network_evolution():
    n_nodes = 10
    distance_matrix = np.random.rand(n_nodes, n_nodes)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    history = simulate_network_evolution(
        distance_matrix=distance_matrix,
        n_iterations=10,
        distance_fn=shortest_path_distance,
        alpha=1.0,
        beta=1.0,
        noise=np.zeros(10),
        connectivity_penalty=0.0,
        n_jobs=1,
        batch_size=2
    )
    
    assert history.shape == (n_nodes, n_nodes, 10)
    assert np.all(history >= 0)
    assert np.all(history <= 1)


def test_find_optimal_alpha():
    n_nodes = 10
    distance_matrix = np.random.rand(n_nodes, n_nodes)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    # Create a dummy empirical connectivity
    empirical_connectivity = np.random.randint(0, 2, (n_nodes, n_nodes))
    empirical_connectivity = (empirical_connectivity + empirical_connectivity.T) // 2
    np.fill_diagonal(empirical_connectivity, 0)
    
    result = find_optimal_alpha(
        distance_matrix=distance_matrix,
        empirical_connectivity=empirical_connectivity,
        distance_fn=shortest_path_distance,
        n_iterations=10,
        beta=1.0,
        alpha_range=(0.1, 2.0),
        max_search_iterations=2,
        n_jobs=1,
        batch_size=2
    )
    
    assert isinstance(result, dict)
    assert 'alpha' in result
    assert 'density' in result
    assert 'evolution' in result
    assert isinstance(result['alpha'], float)
    assert isinstance(result['density'], float)
    assert result['evolution'].shape == (n_nodes, n_nodes, 10)
