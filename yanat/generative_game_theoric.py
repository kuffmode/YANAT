
from typing import Any, Callable, Dict, Union, Literal, TypeAlias, Optional
import numpy as np
import numba
import numpy.typing as npt
from numba.core.errors import TypingError
from numba import njit, jit, NumbaError
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
numba.config.DISABLE_JIT_WARNINGS = 1

FloatArray = npt.NDArray[np.float64]
Trajectory: TypeAlias = Union[float, FloatArray]
DistanceMetric = Callable[[FloatArray, FloatArray], FloatArray]
NoiseType = Union[Literal[0], FloatArray]



def _apply_weighting(adjacency_matrix: FloatArray, distance_matrix: Optional[FloatArray] = None, weight_coefficient: float = 0.0, **kwargs) -> FloatArray:
    """
    Applies distance-dependent weighting to the adjacency matrix.
    
    The weighting is exponential: W_ij = A_ij * exp(-weight_coefficient * D_ij).
    This penalizes long-distance connections if weight_coefficient > 0.

    Args:
        adjacency_matrix: The binary or weighted adjacency matrix (n_nodes, n_nodes).
        distance_matrix: The distance matrix (n_nodes, n_nodes). Required if weight_coefficient > 0.
        weight_coefficient: The decay coefficient for distance weighting. 
                            If 0.0, no weighting is applied.
        **kwargs: Additional arguments (ignored).

    Returns:
        The weighted adjacency matrix.
    
    Raises:
        ValueError: If weight_coefficient > 0 but distance_matrix is not provided.
    """
    if weight_coefficient == 0.0:
        return adjacency_matrix
        
    if distance_matrix is None:
        raise ValueError("distance_matrix must be provided for weighted distance calculation (weight_coefficient > 0)")
        
    weights = np.exp(-weight_coefficient * distance_matrix)
    return adjacency_matrix * weights

def validate_parameters(
    sim_length: int,
    *trajectories: Trajectory,
    names: tuple[str, ...],
    allow_float: tuple[bool, ...],
    allow_zero: tuple[bool, ...],
    allow_none: Optional[tuple[bool, ...]] = None
) -> None:
    """
    Validates simulation parameters to ensure they are of correct type and shape.

    This function checks if parameters are either scalars (floats/ints) or arrays 
    matching the simulation length. It also checks for allowed values (zero, None).

    Args:
        sim_length: The total number of iterations in the simulation.
        *trajectories: Variable number of parameter trajectories to validate.
        names: A tuple of names corresponding to the trajectories, used for error messages.
        allow_float: A tuple of booleans indicating if a parameter can be a scalar float.
        allow_zero: A tuple of booleans indicating if a parameter can be zero.
        allow_none: A tuple of booleans indicating if a parameter can be None.

    Raises:
        ValueError: If any parameter does not meet the specified criteria (type, shape, value).
        
    Example:
        >>> validate_parameters(
        ...     1000,
        ...     1.0, np.zeros(1000),
        ...     names=('alpha', 'beta'),
        ...     allow_float=(True, False),
        ...     allow_zero=(False, True)
        ... )
    """
    none_ok = allow_none if allow_none is not None else (False,) * len(trajectories)
    for traj, name, float_ok, zero_ok, none_ok in zip(
        trajectories, names, allow_float, allow_zero, none_ok
    ):
        if none_ok and traj is None:
            continue
        if isinstance(traj, (float, int)):
            if not float_ok:
                raise ValueError(
                    f"{name} must be an array, got {type(traj)}"
                )
            if not zero_ok and traj == 0:
                raise ValueError(f"{name} cannot be zero")
        elif isinstance(traj, np.ndarray):
            if traj.shape != (sim_length,):
                raise ValueError(
                    f"{name} trajectory length {len(traj)} doesn't match "
                    f"simulation length {sim_length}"
                )
            if not zero_ok and np.any(traj == 0):
                raise ValueError(f"{name} cannot contain zeros")
        else:
            raise ValueError(
                f"{name} must be float or array, got {type(traj)}"
            )


def get_param_value(param: Trajectory, t: int) -> float:
    """
    Retrieves the value of a parameter at a specific time step `t`.

    If the parameter is a scalar, it returns the scalar.
    If it is an array (trajectory), it returns the value at index `t`.

    Args:
        param: The parameter, either a float or a numpy array.
        t: The current time step (index).
        
    Returns:
        The parameter value at time t.
    """
    if isinstance(param, np.ndarray):
        return param[t]
    return param



@njit
def compute_component_sizes(adjacency: FloatArray) -> FloatArray:
    """
    Computes the size of the connected component for each node using BFS.

    This function uses Numba for performance. It iterates over all nodes,
    and for each unvisited node, it performs a Breadth-First Search (BFS)
    to find all reachable nodes, counting them to determine the component size.

    Args:
        adjacency: The adjacency matrix (n_nodes, n_nodes).

    Returns:
        An array of size (n_nodes,) where the i-th element is the size of the
        connected component containing node i.
    """
    n_nodes = len(adjacency)
    visited = np.zeros(n_nodes, dtype=np.bool_)
    sizes = np.zeros(n_nodes, dtype=np.float64)
    
    for start_node in range(n_nodes):
        if visited[start_node]:
            continue
            
        # BFS initialization
        component = np.zeros(n_nodes, dtype=np.bool_)
        queue = np.zeros(n_nodes, dtype=np.int64)
        queue_size = 1
        queue[0] = start_node
        component[start_node] = True
        
        # BFS loop
        idx = 0
        while idx < queue_size:
            node = queue[idx]
            idx += 1
            
            for neighbor in range(n_nodes):
                if (adjacency[node, neighbor] and 
                    not component[neighbor]):
                    component[neighbor] = True
                    queue[queue_size] = neighbor
                    queue_size += 1
        
        # Assign component size to all nodes in this component
        comp_size = float(queue_size)
        for i in range(queue_size):
            node = queue[i]
            visited[node] = True
            sizes[node] = comp_size
            
    return sizes

def propagation_distance(adjacency_matrix: FloatArray, spatial_decay: float = 0.8, symmetric: bool = True, **kwargs) -> FloatArray:
    """
    Computes the propagation distance matrix.

    The propagation distance is defined as the negative log of the influence matrix.
    The influence matrix is computed using either the LAM (Linear Attenuation Model)
    or SAR (Spatial Autoregressive) model.

    Args:
        adjacency_matrix: The adjacency matrix (n_nodes, n_nodes).
        spatial_decay: Decay parameter (0 < spatial_decay < 1/spectral_radius).
        symmetric: If True, uses SAR model (symmetric influence). 
                   If False, uses LAM model (directed influence).
        **kwargs: Additional arguments passed to `_apply_weighting`.

    Returns:
        The propagation distance matrix (n_nodes, n_nodes).
    """
    from yanat.core import lam, sar
    from yanat.utils import log_normalize, spectral_normalization
    
    # Apply weighting if weight_coefficient is present
    weighted_adj = spectral_normalization(1.0, _apply_weighting(adjacency_matrix, **kwargs))
    
    if symmetric:
        # SAR model returns covariance (influence) matrix
        influence = sar(weighted_adj, alpha=spatial_decay)
    else:
        # LAM model returns influence matrix
        influence = lam(weighted_adj, alpha=spatial_decay)
    
    return -log_normalize(influence)

def resistance_distance(adjacency: FloatArray, **kwargs) -> FloatArray:
    """
    Computes resistance distances between all pairs of nodes.

    The resistance distance is computed using the Moore-Penrose pseudoinverse
    of the Laplacian matrix. It treats the graph as an electrical network
    where edges are resistors.

    Args:
        adjacency: The adjacency matrix (n_nodes, n_nodes).
        **kwargs: Additional arguments passed to `_apply_weighting`.

    Returns:
        The resistance distance matrix (n_nodes, n_nodes).
    """
    # Apply weighting if weight_coefficient is present
    weighted_adj = _apply_weighting(adjacency, **kwargs)
    
    # Ensure symmetry for Laplacian calculation
    weights = weighted_adj.astype(np.float64)
    weights = (weights + weights.T) / 2.0
    
    # Laplacian: L = D - W
    degree = np.sum(weights, axis=1)
    laplacian = np.diag(degree) - weights
    
    # Compute pseudoinverse
    pinv = np.linalg.pinv(laplacian)
    
    # R_ij = L+_ii + L+_jj - 2*L+_ij
    diag_pinv = np.diag(pinv)
    resistance = diag_pinv[:, None] + diag_pinv[None, :] - 2 * pinv
    
    # Ensure non-negative and zero diagonal
    resistance[resistance < 0] = 0
    np.fill_diagonal(resistance, 0)
            
    return resistance

def heat_kernel_distance(adjacency_matrix: FloatArray, t: float = 0.5, eps: float = 1e-10, **kwargs) -> FloatArray:
    """
    Computes the heat kernel distance matrix at diffusion time t.

    The heat kernel distance is defined as -log(exp(-t * L)), where L is the Laplacian.

    Args:
        adjacency_matrix: The adjacency matrix (n_nodes, n_nodes).
        t: Diffusion time parameter.
        eps: Small constant to avoid log(0).
        **kwargs: Additional arguments passed to `_apply_weighting`.

    Returns:
        The heat kernel distance matrix (n_nodes, n_nodes).
    """
    # Apply weighting if weight_coefficient is present
    weighted_adj = _apply_weighting(adjacency_matrix, **kwargs)
    
    # Laplacian
    degree = np.sum(weighted_adj, axis=1)
    laplacian = np.diag(degree) - weighted_adj
    
    # Heat kernel: exp(-t*L)
    from scipy.linalg import expm
    kernel = expm(-t * laplacian)
    
    # Distance: -log(kernel)
    kernel[kernel < eps] = eps
    
    return -np.log(kernel)

def shortest_path_distance(adjacency_matrix: FloatArray, **kwargs) -> FloatArray:
    """
    Computes shortest-path distances between all pairs of nodes.

    Uses Dijkstra's algorithm via `scipy.sparse.csgraph.shortest_path`.
    If the graph is weighted (via `weight_coefficient`), edge weights are treated
    as costs (inverted if they represent strength).

    Args:
        adjacency_matrix: The adjacency matrix (n_nodes, n_nodes).
        **kwargs: Additional arguments passed to `_apply_weighting`.

    Returns:
        The shortest path distance matrix (n_nodes, n_nodes).
    """
    from scipy.sparse.csgraph import shortest_path
    
    # Apply weighting if weight_coefficient is present
    weighted_adj = _apply_weighting(adjacency_matrix, **kwargs)
    
    # Prepare graph for shortest path
    # If weights represent strength/similarity, we invert them for distance/cost.
    # If unweighted (binary), we use 1/1 = 1.
    graph = weighted_adj.copy()
    mask = graph > 0
    if np.any(mask):
        graph[mask] = 1.0 / graph[mask]
    
    dist_matrix = shortest_path(graph, method='auto', directed=False)
    
    return dist_matrix

def search_information(W, symmetric=False, **kwargs):
    """
    Calculate search information for a memoryless random walker.
    
    Args:
        W (np.ndarray): Adjacency matrix (N x N)
        symmetric (bool): If True, symmetrize W.
        **kwargs: Additional arguments.
        
    Returns
    -------
    SI : (N, N) ndarray
        Pairwise search information matrix (>=0); diagonal = 0; unreachable pairs = inf.
    """
    from scipy.sparse.csgraph import shortest_path

    # Apply weighting if weight_coefficient is present
    W = _apply_weighting(W, **kwargs)
    
    N = W.shape[0]

    # Safe row-normalization to transition probabilities
    T = np.zeros((N, N), dtype=np.float64)
    row_sums = np.sum(W, axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            T[i, :] = W[i, :] / row_sums[i]
        # If row sum is zero, leave zeros (no outgoing probability)

    # Compute shortest paths on binary graph (hop count)
    # Treat existing edges as length 1
    binary_adj = (W > 0).astype(np.float64)
    
    dist_matrix, predecessors = shortest_path(
        binary_adj, 
        method='auto', 
        directed=True, 
        return_predecessors=True,
        unweighted=True # Use unweighted to get hop counts
    )

    # Compute search information using log-sum for numerical stability
    SI = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(SI, 0.0)
    
    # Pre-calculate log probabilities
    with np.errstate(divide='ignore'):
        log_T = -np.log2(T)
    # Replace inf with large number or handle carefully?
    # If T=0, log_T=inf. Correct.
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dist_matrix[i, j] == np.inf:
                continue

            current = j
            prob_sum = 0.0
            valid = True
            
            # Backtrack from j to i
            while current != i:
                pred = predecessors[i, current]
                if pred == -9999: # No path, should not happen if dist != inf
                    valid = False
                    break
                
                # Edge pred -> current
                weight = log_T[pred, current]
                if weight == np.inf:
                    valid = False
                    break
                prob_sum += weight
                current = pred
            
            if valid:
                SI[i, j] = prob_sum

    if symmetric:
        # Symmetrize by taking min(SI[i,j], SI[j,i])
        SI = np.minimum(SI, SI.T)
        
    return SI

def topological_distance(adj_matrix: FloatArray, **kwargs) -> FloatArray:
    """
    Computes pairwise topological distance based on cosine similarity of neighbors.

    Returns 1 - cosine_similarity. Nodes with similar connectivity patterns 
    will have small topological distance.

    Args:
        adj_matrix: The binary adjacency matrix (n_nodes, n_nodes).
        **kwargs: Additional arguments passed to `_apply_weighting`.

    Returns:
        The topological distance matrix (n_nodes, n_nodes).
    """
    # Apply weighting if weight_coefficient is present
    weighted_adj = _apply_weighting(adj_matrix, **kwargs)
    
    n_nodes = weighted_adj.shape[0]
    matching_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        edges_i = weighted_adj[i]
        norm_i = np.sqrt(np.sum(edges_i * edges_i))
        
        for j in range(i, n_nodes):
            edges_j = weighted_adj[j]
            norm_j = np.sqrt(np.sum(edges_j * edges_j))
            
            # Handle zero-degree nodes
            if norm_i == 0 or norm_j == 0:
                matching_matrix[i, j] = matching_matrix[j, i] = 0
                continue
                
            # Compute cosine similarity
            dot_product = np.sum(edges_i * edges_j)
            similarity = dot_product / (norm_i * norm_j)
            
            # Fill both triangles due to symmetry
            matching_matrix[i, j] = matching_matrix[j, i] = similarity
            
    return 1.0 - matching_matrix

    
def compute_node_payoff(
    node: int,
    adjacency: FloatArray,
    distance_matrix: FloatArray,
    distance_fn: Callable[..., FloatArray],
    alpha: float,
    beta: float,
    connectivity_penalty: float,
    node_resources: Optional[FloatArray] = None,
    distance_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Computes the payoff for a single node based on distance, wiring cost, and connectivity.

    The payoff is defined as:
    Payoff = - (alpha * distance_term + beta * wiring_cost + connectivity_penalty * disconnected_nodes)

    Args:
        node: Index of the node.
        adjacency: The current adjacency matrix (n_nodes, n_nodes).
        distance_matrix: Pre-computed Euclidean distance matrix (n_nodes, n_nodes) for wiring cost.
        distance_fn: Function to compute the 'communication' distance metric (e.g., shortest path).
        alpha: Weight of the distance term (communication efficiency).
        beta: Weight of the wiring cost term (physical distance).
        connectivity_penalty: Penalty for each node not in the same connected component.
        node_resources: Optional vector of node resources to subsidize wiring cost.
        distance_fn_kwargs: Additional keyword arguments for `distance_fn`.

    Returns:
        The calculated payoff value (typically negative).
    """
    if distance_fn_kwargs is None:
        distance_fn_kwargs = {}
        
    n_nodes = len(adjacency)
    payoff = 0.0
    
    # 1. Communication Distance Term
    if alpha != 0:
        # Compute distances using the provided metric function
        # Note: Some metrics might use distance_matrix for weighting
        distances = distance_fn(adjacency, distance_matrix=distance_matrix, **distance_fn_kwargs)
        # Sum of distances from this node to all other nodes
        payoff -= alpha * np.sum(distances.T[node])
        
    # 2. Wiring Cost Term
    if beta != 0:
        # Euclidean distances from this node to all others
        euclidean = distance_matrix[node]
        
        # Apply resource subsidy if provided
        if node_resources is not None:
            # Effective distance is reduced by the node's resource
            # We clip at 0 to avoid negative costs (profit)
            effective_dist = np.maximum(0.0, euclidean - node_resources[node])
            payoff -= beta * np.sum(adjacency[node] * effective_dist)
        else:
            payoff -= beta * np.sum(adjacency[node] * euclidean)
        
    # 3. Connectivity Penalty Term
    if connectivity_penalty != 0:
        comp_sizes = compute_component_sizes(adjacency)
        # Penalty proportional to number of nodes NOT in the same component
        payoff -= connectivity_penalty * (n_nodes - comp_sizes[node])
        
    return payoff
    

def simulate_network_evolution(
    distance_matrix: FloatArray,
    n_iterations: int,
    distance_fn: DistanceMetric,
    alpha: Trajectory,
    beta: Trajectory,
    connectivity_penalty: Trajectory,
    initial_adjacency: Optional[FloatArray] = None,
    n_jobs: int = -1,
    batch_size: Trajectory = 32,
    node_resources: Optional[FloatArray] = None,
    payoff_tolerance: Trajectory = 0.0,
    random_seed: Optional[int] = None,
    symmetric: Optional[bool] = True,
    **kwargs
) -> FloatArray:
    """
    Simulates the evolution of a network through game-theoretic payoff optimization.

    At each step, random edges are selected and "flipped" (added or removed). 
    The change is accepted if it improves the payoff for at least one of the nodes involved 
    (unilateral consent), subject to a tolerance threshold.

    Args:
        distance_matrix: Pre-computed Euclidean distance matrix (n_nodes, n_nodes).
        n_iterations: Total number of simulation steps.
        distance_fn: Function to compute the 'nonphysical' distance metric, e.g., communication.
        alpha: Weight of the distance term (float or trajectory array).
        beta: Weight of the wiring cost term (float or trajectory array).
        connectivity_penalty: Penalty for disconnected components (float or trajectory array).
        initial_adjacency: Starting adjacency matrix. If None, starts with a ring lattice.
        n_jobs: Number of parallel jobs for payoff computation (-1 for all cores).
        batch_size: Number of potential edge flips to evaluate per iteration.
        node_resources: Optional resources for each node to subsidize wiring costs.
        payoff_tolerance: Minimum payoff improvement required to accept a change.
        random_seed: Seed for random number generator.
        symmetric: If True, enforces undirected edges (symmetry).
        **kwargs: Additional arguments passed to `distance_fn`.

    Returns:
        A 3D array of shape (n_nodes, n_nodes, n_iterations) containing the 
        adjacency matrix at each time step.

    Example:
        >>> dist_mat = np.random.rand(10, 10)
        >>> history = simulate_network_evolution(
        ...     distance_matrix=dist_mat,
        ...     n_iterations=100,
        ...     distance_fn=shortest_path_distance,
        ...     alpha=1.0,
        ...     beta=0.5,
        ...     connectivity_penalty=10.0
        ... )
    """
    # Parameter validation
    validate_parameters(
        n_iterations,
        alpha, beta, connectivity_penalty, batch_size, payoff_tolerance,
        names=('alpha', 'beta', 'connectivity_penalty', 'batch_size', 'payoff_tolerance'),
        allow_float=(True, True, True, True, True),
        allow_zero=(True, True, True, False, True)
    )
    
    distance_fn_kwargs = kwargs.get('distance_fn_kwargs', {})
    # Pass weight_coefficient to distance_fn if present
    if 'weight_coefficient' in kwargs:
        distance_fn_kwargs['weight_coefficient'] = kwargs['weight_coefficient']
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_nodes = len(distance_matrix)
    
    # Initialize adjacency if not provided
    if initial_adjacency is None:
        # Start with ring structure (k=2 regular graph)
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        idx = np.arange(n_nodes)
        adjacency[idx, (idx + 1) % n_nodes] = 1
        adjacency[(idx + 1) % n_nodes, idx] = 1
    else:
        adjacency = initial_adjacency.copy()
        
    # Pre-allocate history
    history = np.zeros((n_nodes, n_nodes, n_iterations))
    history[:, :, 0] = adjacency
    
    # Simulation loop with progress bar
    with tqdm(total=n_iterations-1, desc="Simulating network evolution") as pbar:
        for t in range(1, n_iterations):
            # Get parameters for this step
            alpha_t = get_param_value(alpha, t)
            beta_t = get_param_value(beta, t)
            penalty_t = get_param_value(connectivity_penalty, t)
            batch_size_t = int(get_param_value(batch_size, t))
            tolerance_t = get_param_value(payoff_tolerance, t)
            
            # Select random edges to flip
            i_indices = np.random.randint(0, n_nodes, batch_size_t)
            j_indices = np.random.randint(0, n_nodes, batch_size_t)
            
            # Ensure i != j (no self-loops)
            mask = i_indices == j_indices
            while np.any(mask):
                j_indices[mask] = np.random.randint(0, n_nodes, np.sum(mask))
                mask = i_indices == j_indices
            
            # Compute payoff impact for selected edges in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_impact_for_edge)(
                    i, j, adjacency, distance_matrix, distance_fn,
                    alpha_t, beta_t, penalty_t, node_resources, distance_fn_kwargs
                )
                for i, j in zip(i_indices, j_indices)
            )
            
            # Process results and update adjacency
            for idx, (i, j, diff_i, diff_j) in enumerate(results):
                # Check if change is beneficial
                # We require at least one node to benefit (unilateral consent)
                if diff_i > tolerance_t or diff_j > tolerance_t:
                    # Flip edge
                    val = 1.0 - adjacency[i, j]
                    adjacency[i, j] = val
                    if symmetric:
                        adjacency[j, i] = val
                    
            history[:, :, t] = adjacency
            pbar.update(1)
            
    return history

def _compute_impact_for_edge(
    i: int,
    j: int,
    original_adjacency: FloatArray,
    distance_matrix: FloatArray,
    distance_fn: DistanceMetric,
    alpha: float,
    beta: float,
    connectivity_penalty: float,
    node_resources: Optional[FloatArray],
    distance_fn_kwargs: Dict[str, Any]
) -> tuple[int, int, float, float]:
    """
    Helper function to compute payoff impact of flipping an edge (i, j).
    
    This function is designed for parallel execution. It creates a temporary
    copy of the adjacency matrix with the edge (i, j) flipped, and computes
    the change in payoff for both nodes i and j.

    Args:
        i: Index of the first node.
        j: Index of the second node.
        original_adjacency: The current adjacency matrix.
        distance_matrix: The distance matrix.
        distance_fn: The distance metric function.
        alpha: Distance weight.
        beta: Wiring cost weight.
        connectivity_penalty: Connectivity penalty.
        node_resources: Node resources.
        distance_fn_kwargs: Additional kwargs for distance_fn.

    Returns:
        A tuple (i, j, diff_i, diff_j) where diff_i and diff_j are the 
        payoff differences (new - old) for nodes i and j respectively.
    """
    # 1. Create the lesioned adjacency matrix (a copy)
    adj_lesioned = original_adjacency.copy()
    # Flip the edge value (0->1 or 1->0)
    val = 1.0 - adj_lesioned[i, j]
    adj_lesioned[i, j] = val
    adj_lesioned[j, i] = val # Assuming symmetric adjacency

    # 2. Calculate new payoffs for nodes i and j using the lesioned matrix
    new_payoff_i = compute_node_payoff(
        node=i,
        adjacency=adj_lesioned,
        distance_matrix=distance_matrix,
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        connectivity_penalty=connectivity_penalty,
        node_resources=node_resources,
        distance_fn_kwargs=distance_fn_kwargs
    )

    new_payoff_j = compute_node_payoff(
        node=j,
        adjacency=adj_lesioned,
        distance_matrix=distance_matrix,
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        connectivity_penalty=connectivity_penalty,
        node_resources=node_resources,
        distance_fn_kwargs=distance_fn_kwargs
    )
    
    # 3. Calculate original payoffs
    old_payoff_i = compute_node_payoff(
        node=i,
        adjacency=original_adjacency,
        distance_matrix=distance_matrix,
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        connectivity_penalty=connectivity_penalty,
        node_resources=node_resources,
        distance_fn_kwargs=distance_fn_kwargs
    )
    
    old_payoff_j = compute_node_payoff(
        node=j,
        adjacency=original_adjacency,
        distance_matrix=distance_matrix,
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        connectivity_penalty=connectivity_penalty,
        node_resources=node_resources,
        distance_fn_kwargs=distance_fn_kwargs
    )

    # 4. Compute differences
    diff_i = new_payoff_i - old_payoff_i
    diff_j = new_payoff_j - old_payoff_j

    # Ensure differences are finite numbers, default to 0.0 otherwise
    diff_i = diff_i if np.isfinite(diff_i) else 0.0
    diff_j = diff_j if np.isfinite(diff_j) else 0.0

    return i, j, diff_i, diff_j

def compute_local_payoff_impact(
    distance_matrix: FloatArray,
    adjacency_matrix: FloatArray,
    distance_fn: DistanceMetric,
    alpha: float,
    beta: float = 1.0,
    distance_fn_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1
) -> FloatArray:
    """
    Computes the impact on payoff for connected nodes when their edge is removed.

    This function iterates through all existing edges, temporarily removes each one,
    and calculates the change in payoff for the two connected nodes.

    Args:
        distance_matrix: Pre-computed distance matrix (n_nodes, n_nodes).
        adjacency_matrix: The binary, symmetric adjacency matrix (n_nodes, n_nodes).
        distance_fn: Function to compute the 'communication' distance metric.
        alpha: Weight of the distance term.
        beta: Weight of the wiring cost term.
        distance_fn_kwargs: Additional kwargs for distance_fn.
        n_jobs: Number of parallel jobs (-1 for all cores).

    Returns:
        An N x N matrix where `matrix[i, j]` contains the change in payoff for node `i`
        when the edge `(i, j)` is removed. The matrix is sparse (zeros where no edge exists).
    """
    n_nodes = adjacency_matrix.shape[0]
    if distance_fn_kwargs is None:
        distance_fn_kwargs = {}

    # 2. Identify existing edges (upper triangle)
    edge_indices = np.argwhere(np.triu(adjacency_matrix, k=1) > 0)
    num_edges = len(edge_indices)

    if num_edges == 0:
        return np.zeros((n_nodes, n_nodes))
        
    # 3. Parallel Edge Lesioning
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_impact_for_edge)(
            i, j, 
            original_adjacency=adjacency_matrix,
            distance_matrix=distance_matrix,
            distance_fn=distance_fn,
            alpha=alpha,
            beta=beta,
            connectivity_penalty=0.0, # Assuming no penalty for this analysis
            node_resources=None,
            distance_fn_kwargs=distance_fn_kwargs
        ) for i, j in edge_indices
    )

    # 4. Aggregation
    payoff_impact_matrix = np.zeros((n_nodes, n_nodes))
    for i, j, diff_i, diff_j in results:
        payoff_impact_matrix[i, j] = diff_i
        payoff_impact_matrix[j, i] = diff_j
        
    return payoff_impact_matrix

def find_optimal_alpha(
    distance_matrix: np.ndarray,
    empirical_connectivity: np.ndarray,
    distance_fn: Callable,
    n_iterations: int = 10_000,
    beta: float = 1.0,
    alpha_range: tuple[float, float] = (1.0, 100.0),
    tolerance: float = 0.01,
    max_search_iterations: int = 20,
    random_seed: int = 11,
    batch_size: int = 16,
    symmetric: bool = True,
    n_jobs: int = -1,
    connectivity_penalty: float = 0.0,
    payoff_tolerance: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Finds the optimal alpha value that produces a network with density closest to empirical.

    This function uses a bisection search (with linear interpolation) to find the alpha
    parameter that results in a generated network with the same density as the provided
    empirical network.

    Args:
        distance_matrix: Precomputed distance matrix (n_nodes, n_nodes).
        empirical_connectivity: Target connectivity matrix to match density with.
        distance_fn: Distance metric function.
        n_iterations: Number of iterations for each simulation.
        beta: Wiring cost parameter.
        alpha_range: Range for alpha search (min, max).
        tolerance: Acceptable difference between densities.
        max_search_iterations: Maximum number of search iterations.
        random_seed: Random seed for reproducibility.
        batch_size: Batch size for parallel processing.
        symmetric: If True, enforces symmetry in generated networks.
        n_jobs: Number of parallel jobs.
        connectivity_penalty: Penalty for connectivity.
        payoff_tolerance: Threshold for accepting new configuration.
        **kwargs: Additional arguments passed to `simulate_network_evolution`.

    Returns:
        Dictionary containing:
            - 'alpha': Optimal alpha value.
            - 'density': Density of the resulting network.
            - 'evolution': Full history of adjacency matrices.

    Example:
        >>> dist_mat = np.random.rand(10, 10)
        >>> emp_conn = np.random.randint(0, 2, (10, 10))
        >>> result = find_optimal_alpha(
        ...     distance_matrix=dist_mat,
        ...     empirical_connectivity=emp_conn,
        ...     distance_fn=shortest_path_distance
        ... )
        >>> print(f"Optimal alpha: {result['alpha']}")
    """
    # Calculate empirical density
    n_nodes = empirical_connectivity.shape[0]
    empirical_density = np.sum(empirical_connectivity.astype(bool).astype(int)) / (n_nodes * (n_nodes - 1))
    
    # Set up fixed parameters as vectors
    beta_vec = np.full(n_iterations, beta)
    penalty_vec = np.full(n_iterations, connectivity_penalty)
    batch_size_vec = np.full(n_iterations, batch_size)
    tolerance_vec = np.full(n_iterations, payoff_tolerance)
    
    # Function to simulate network and get density
    def simulate_with_alpha(alpha_value):
        alpha_vec = np.full(n_iterations, alpha_value)
        network = simulate_network_evolution(
            distance_matrix=distance_matrix,
            n_iterations=n_iterations,
            distance_fn=distance_fn,
            alpha=alpha_vec,
            beta=beta_vec,
            connectivity_penalty=penalty_vec,
            n_jobs=n_jobs,
            random_seed=random_seed,
            batch_size=batch_size_vec,
            symmetric=symmetric,
            payoff_tolerance=tolerance_vec,
            **kwargs
        )
        
        # Get final network
        final_adj = network[:, :, -1]
        density = np.sum(final_adj.astype(bool).astype(int)) / (n_nodes * (n_nodes - 1))
        
        return density, network
    
    # Initialize search with provided range
    alpha_min, alpha_max = alpha_range
    min_density, min_net = simulate_with_alpha(alpha_min)
    max_density, max_net = simulate_with_alpha(alpha_max)
    
    print(f"Initial range: alpha=[{alpha_min}, {alpha_max}], density=[{min_density}, {max_density}]")
    print(f"Target density: {empirical_density}")
    
    # Track all tested points
    tested_points = [
        {'alpha': alpha_min, 'density': min_density, 'evolution': min_net},
        {'alpha': alpha_max, 'density': max_density, 'evolution': max_net}
    ]
    
    # Check if range brackets the target
    if (min_density > empirical_density and max_density > empirical_density) or \
       (min_density < empirical_density and max_density < empirical_density):
        print("Warning: Initial range does not bracket the target density!")
        # Return best of the two
        return min(tested_points, key=lambda p: abs(p['density'] - empirical_density))
    
    # Bisection search with linear interpolation
    iterations = 0
    while iterations < max_search_iterations:
        # Use linear interpolation to estimate next alpha
        alpha_mid = alpha_min + (alpha_max - alpha_min) * \
                    (empirical_density - min_density) / (max_density - min_density)
        
        # Fallback to bisection if interpolation gives unusual values
        if not (alpha_min < alpha_mid < alpha_max):
            alpha_mid = (alpha_min + alpha_max) / 2
        
        print(f"Iteration {iterations+1}: Testing alpha={alpha_mid}")
        mid_density, mid_net = simulate_with_alpha(alpha_mid)
        tested_points.append({'alpha': alpha_mid, 'density': mid_density, 'evolution': mid_net})
        print(f"Alpha={alpha_mid} → density={mid_density}, diff from target={mid_density-empirical_density}")
        
        # Check if we're close enough
        if abs(mid_density - empirical_density) < tolerance:
            return {'alpha': alpha_mid, 'density': mid_density, 'evolution': mid_net}
        
        # Update range
        if mid_density < empirical_density:
            alpha_min = alpha_mid
            min_density = mid_density
        else:
            alpha_max = alpha_mid
            max_density = mid_density
        
        # Check if range is small enough
        if alpha_max - alpha_min < tolerance:
            return min(tested_points, key=lambda p: abs(p['density'] - empirical_density))
        
        iterations += 1
    
    # If we reach max iterations, return best point found
    return min(tested_points, key=lambda p: abs(p['density'] - empirical_density))