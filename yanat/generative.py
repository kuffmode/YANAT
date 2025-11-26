
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

def jit_safe(nopython=True, **jit_kwargs):
    """
    A safe JIT wrapper that falls back to normal Python if
    Numba cannot compile the function in nopython mode.

    Parameters
    ----------
    nopython : bool
        Whether to try nopython mode first.
    jit_kwargs : dict
        Additional arguments passed to @jit.
    """
    def decorator(func):
        if not nopython:
            return func  # Skip JIT entirely if nopython=False
        
        try:
            # Try compiling in nopython mode
            jitted = jit(nopython=True, **jit_kwargs)(func)
            # Test compilation with dummy input if possible
            # (you can define test cases if needed)
            return jitted
        except (TypingError, NumbaError) as e:
            # Fallback to non-JIT version
            print(f"Numba could not compile {func.__name__}: {e}")
            return func  # Return the original function if JIT fails
    return decorator

@njit
def _diag_indices(n):
    """
    Returns the indices of the diagonal elements of an n x n matrix.
    """
    rows = np.arange(n)
    cols = np.arange(n)
    return rows, cols

@njit
def _set_diagonal(matrix:np.ndarray, value:float=0.0):
    n = matrix.shape[0]
    rows, cols = _diag_indices(n)
    for i in range(n):
        matrix[rows[i], cols[i]] = value
    return matrix

@jit_safe()
def process_matrix(matrix):
    # Replace np.nan_to_num with a manual implementation
    n, m = matrix.shape
    result = np.empty_like(matrix)
    for i in range(n):
        for j in range(m):
            val = matrix[i, j]
            if np.isnan(val):
                result[i, j] = 0
            elif val == np.inf:
                result[i, j] = 0
            elif val == -np.inf:
                result[i, j] = 0
            else:
                result[i, j] = val
    return result

def validate_parameters(
    sim_length: int,
    *trajectories: Trajectory,
    names: tuple[str, ...],
    allow_float: tuple[bool, ...],
    allow_zero: tuple[bool, ...],
    allow_none: Optional[tuple[bool, ...]] = None
) -> None:
    """
    Validate simulation parameters.
    
    Args:
        sim_length: Length of simulation
        *trajectories: Parameter trajectories to validate
        names: Names of parameters for error messages
        allow_float: Whether each parameter can be float
        allow_zero: Whether each parameter can be zero
    
    Raises:
        ValueError: If any parameter is invalid
        
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
    Get parameter value at time t efficiently.
    
    Args:
        param: Parameter trajectory (float or array)
        t: Time step
        
    Returns:
        Parameter value at time t
    """
    if isinstance(param, np.ndarray):
        return param[t]
    return param

def normalize_distances(
    coordinates: FloatArray,
    normalization: Literal["sqrt_dim", "max", "mean"] = "sqrt_dim"
) -> FloatArray:
    """
    Compute and normalize pairwise distances.
    
    Args:
        coordinates: Node coordinates (n_nodes, n_dimensions)
        normalization: Normalization method
            - "sqrt_dim": Divide by sqrt(dimensionality)
            - "max": Divide by maximum distance
            - "mean": Divide by mean distance
            
    Returns:
        Normalized distance matrix (n_nodes, n_nodes)
        
    Raises:
        ValueError: If normalization factor is zero or method unknown
    """
    if coordinates.ndim != 2:
        raise ValueError("Coordinates must be 2D array (n_nodes, n_dimensions)")
        
    distances = squareform(pdist(coordinates, metric='euclidean'))
    
    if normalization == "sqrt_dim":
        norm_factor = np.sqrt(coordinates.shape[1])
    elif normalization == "max":
        norm_factor = np.max(distances)
    elif normalization == "mean":
        norm_factor = np.mean(distances)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
        
    if norm_factor == 0:
        raise ValueError("Normalization factor is zero")
        
    return distances / norm_factor

@njit
def compute_component_sizes(adjacency: FloatArray) -> FloatArray:
    """
    Compute size of connected component for each node efficiently.
    Uses numba-optimized BFS to find connected components.
    
    Args:
        adjacency: Adjacency matrix (n_nodes, n_nodes)
    
    Returns:
        Array of component sizes (n_nodes,)
    """
    n_nodes = len(adjacency)
    visited = np.zeros(n_nodes, dtype=np.bool_)
    sizes = np.zeros(n_nodes, dtype=np.float64)
    
    for start_node in range(n_nodes):
        if visited[start_node]:
            continue
            
        # BFS from start_node
        component = np.zeros(n_nodes, dtype=np.bool_)
        queue = np.zeros(n_nodes, dtype=np.int64)
        queue_size = 1
        queue[0] = start_node
        component[start_node] = True
        
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
        
        # Update all nodes in component
        comp_size = float(queue_size)
        for i in range(queue_size):
            node = queue[i]
            visited[node] = True
            sizes[node] = comp_size
            
    return sizes

def propagation_distance(adjacency_matrix, alpha=0.8, eps=1e-10, **kwargs):
    """
    Computes the propagation distance matrix using:
        -log((I - α*A)^{-1} * (I - α*A)^{-1}.T)
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square adjacency matrix (must be invertible for I - α*A).
    coordinates : np.ndarray
        Not used in computation, kept for signature consistency.
    alpha : float, optional
        Scaling factor for the adjacency matrix (default: 0.5).
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    
    Returns
    -------
    dist : np.ndarray
        The elementwise -log of the propagation matrix.
    """
    N = adjacency_matrix.shape[0]
    A = adjacency_matrix.astype(np.float64)
    
    # Normalize adjacency matrix
    spectral_radius = np.max(np.abs(np.linalg.eigvalsh(A)))
    if spectral_radius > eps:
        A /= spectral_radius
    
    # Compute M = I - α*A
    I = np.eye(N, dtype=np.float64)
    M = I - alpha * A
    
    # Compute inverse and propagation matrix
    M_inv = np.linalg.inv(M)
    prop_matrix = M_inv @ M_inv.T
    
    # Set diagonal to eps and ensure positivity
    prop_matrix = _set_diagonal(prop_matrix, 0.)
    
    # Manual element-wise maximum with eps (numba-friendly)
    for i in range(N):
        for j in range(N):
            if i != j and prop_matrix[i,j] < eps:
                prop_matrix[i,j] = eps
    
    # Compute distance
    return process_matrix(np.abs(-np.log(prop_matrix)))

def resistance_distance(adjacency: FloatArray, **kwargs) -> FloatArray:
    """
    Compute resistance distances between all pairs of nodes using the binary graph.
    
    Args:
        adjacency: Binary adjacency matrix (n_nodes, n_nodes)
        **kwargs: Additional arguments (ignored)
            
    Returns:
        Matrix of resistance distances (n_nodes, n_nodes)
    """
    n_nodes = len(adjacency)
    distances = np.zeros((n_nodes, n_nodes))
    
    # Use binary weights (1.0 for connected nodes)
    # We can just use the adjacency matrix directly as the weight matrix for connected edges
    # But we need to ensure it's symmetric and float
    weights = adjacency.astype(np.float64)
    # Ensure symmetry if not already
    weights = (weights + weights.T) / 2.0
    weights[weights > 0] = 1.0
                
    # Compute weighted Laplacian (which is just the normal Laplacian for binary graph)
    diag = np.sum(weights, axis=1)
    laplacian = np.diag(diag) - weights
    
    # Compute pseudoinverse using eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    mask = eigvals > 1e-10  # Filter out numerical zeros
    eigvals_inv = np.zeros_like(eigvals)
    eigvals_inv[mask] = 1.0 / eigvals[mask]
    
    # Compute resistance distances
    L_plus = (eigvecs * eigvals_inv.reshape(1, -1)) @ eigvecs.T
    resistance = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            r = L_plus[i,i] + L_plus[j,j] - 2*L_plus[i,j]
            resistance[i,j] = r
            resistance[j,i] = r
            
    return resistance

def heat_kernel_distance(adjacency_matrix, t=0.5, eps=1e-10, normalize=False, **kwargs):
    """
    Computes the heat kernel distance matrix at diffusion time t.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square adjacency matrix.
    t : float, optional
        Diffusion time parameter controlling the balance between local and 
        global structure (default: 1.0).
        - Small t values focus on local structure (similar to shortest path)
        - Large t values focus on global structure (approaching resistance distance)
    eps : float, optional
        Small constant to ensure numerical stability (default: 1e-10).
    normalize : bool, optional
        Whether to normalize the distance matrix.
    **kwargs : dict
        Additional arguments (ignored).
    
    Returns
    -------
    dist : np.ndarray
        The heat kernel distance matrix.
    """
    N = adjacency_matrix.shape[0]
    A = adjacency_matrix.astype(np.float64)
    
    # Compute degree matrix and Laplacian
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        D[i, i] = np.sum(A[i])
    
    # Compute Laplacian: L = D - A
    L = D - A
    
    # Compute eigenvalues and eigenvectors of Laplacian
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Compute heat kernel matrix: H_t = exp(-t*L)
    # Using eigendecomposition: H_t = U * exp(-t*Λ) * U^T
    H_t = np.zeros((N, N), dtype=np.float64)
    
    # Compute exp(-t*λ) for each eigenvalue
    exp_vals = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if eigvals[i] > eps:  # Exclude zero eigenvalues (connected components)
            exp_vals[i] = np.exp(-t * eigvals[i])
        else:
            exp_vals[i] = 1.0  # For numerical stability with zero eigenvalues
    
    # Compute H_t using the eigendecomposition
    for i in range(N):
        for j in range(N):
            val = 0.0
            for k in range(N):
                val += exp_vals[k] * eigvecs[i, k] * eigvecs[j, k]
            H_t[i, j] = val
    
    # Compute heat kernel distance: sqrt(H_t(i,i) + H_t(j,j) - 2*H_t(i,j))
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist_ij = H_t[i, i] + H_t[j, j] - 2.0 * H_t[i, j]
                # Ensure non-negative distance due to numerical errors
                if dist_ij > 0:
                    dist_matrix[i, j] = np.sqrt(dist_ij)
                else:
                    dist_matrix[i, j] = 0.0
    if normalize:
        strength = dist_matrix.sum(1)
        normalized_strength: np.ndarray = np.power(strength, -0.5)
        diagonalized_normalized_strength: np.ndarray = np.diag(normalized_strength)
        normalized_dist_matrix: np.ndarray = (
            diagonalized_normalized_strength
            @ dist_matrix
            @ diagonalized_normalized_strength
        )
        return normalized_dist_matrix
    else:
        return dist_matrix

def shortest_path_distance(adjacency_matrix, **kwargs):
    """
    Computes shortest-path distances between all pairs of nodes
    using the Floyd-Warshall algorithm.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        A square adjacency matrix where entry (i, j) represents
        the weight of the edge from node i to j. Use np.inf for no direct edge.
    **kwargs : dict
        Additional arguments (ignored).

    Returns
    -------
    dist_matrix : np.ndarray
        A square matrix where entry (i, j) represents the shortest path
        distance from node i to j.
    """
    N = adjacency_matrix.shape[0]
    
    # Create distance matrix as a copy of the adjacency matrix
    dist_matrix = adjacency_matrix.astype(np.float64)
    
    # Convert zero entries (non-diagonal) to np.inf (no connection)
    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] == 0:
                dist_matrix[i, j] = np.inf
    
    # Ensure diagonal is 0 (distance from a node to itself)
    for i in range(N):
        dist_matrix[i, i] = 0.0

    # Floyd-Warshall algorithm
    for k in range(N):  # Intermediate node
        for i in range(N):  # Source node
            for j in range(N):  # Destination node
                # Relax the distance via intermediate node k
                dist_matrix[i, j] = min(
                    dist_matrix[i, j],
                    dist_matrix[i, k] + dist_matrix[k, j]
                )

    return dist_matrix

def search_information(W, symmetric=False, **kwargs):
    """
    Calculate search information for a memoryless random walker.

    Parameters
    ----------
    W : (N, N) ndarray
        Weighted/unweighted, directed/undirected connection weight matrix (non-negative).
    symmetric : bool
        If True, returns a symmetric matrix using min(SI_ij, SI_ji).
    **kwargs : dict
        Additional arguments (ignored).

    Returns
    -------
    SI : (N, N) ndarray
        Pairwise search information matrix (>=0); diagonal = 0; unreachable pairs = inf.
    Notes
    -----
    Negative values previously arose when transition probabilities were not properly
    normalized (rows summing to zero or numerical issues causing products > 1); this
    implementation uses stable log accumulation and guards zero-strength rows.
    """

    N = W.shape[0]

    # Safe row-normalization to transition probabilities
    T = np.zeros((N, N), dtype=np.float64)
    row_sums = np.sum(W, axis=1)
    for i in range(N):
        if row_sums[i] > 0:
            for j in range(N):
                T[i, j] = W[i, j] / row_sums[i]
        # If row sum is zero, leave zeros (no outgoing probability)

    # Floyd-Warshall (on unweighted structure: treat existing edges as length 1)
    dist_matrix = np.full((N, N), np.inf, dtype=np.float64)
    p_mat = np.zeros((N, N), dtype=np.int32) - 1
    for i in range(N):
        dist_matrix[i, i] = 0.0
        for j in range(N):
            if i != j and W[i, j] > 0:
                dist_matrix[i, j] = 1.0
                p_mat[i, j] = i

    for k in range(N):
        for i in range(N):
            dik = dist_matrix[i, k]
            if dik == np.inf:
                continue
            for j in range(N):
                tmp = dik + dist_matrix[k, j]
                if tmp < dist_matrix[i, j]:
                    dist_matrix[i, j] = tmp
                    p_mat[i, j] = p_mat[k, j]

    # Compute search information using log-sum for numerical stability
    SI = np.zeros((N, N), np.float64)
    for i in range(N):
        SI[i, i] = 0.0
        for j in range(N):
            if i == j:
                continue
            if dist_matrix[i, j] == np.inf:
                continue  # unreachable
            # Reconstruct path j -> i via predecessors
            path = []
            current = j
            while current != -1 and current != i:
                path.append(current)
                current = p_mat[i, current]
            if current != i:
                continue
            path.append(i)
            path.reverse()
            # Accumulate -log2 probabilities along path
            log_si = 0.0
            valid = True
            for k in range(len(path) - 1):
                p = T[path[k], path[k + 1]]
                if p <= 0.0:
                    log_si = np.inf
                    valid = False
                    break
                log_si += -np.log2(p)
            SI[i, j] = log_si if valid else np.inf

    if symmetric:
        result = np.full((N, N), np.inf, dtype=np.float64)
        for i in range(N):
            result[i, i] = 0.0
            for j in range(i + 1, N):
                val = SI[i, j]
                other = SI[j, i]
                m = val if val < other else other
                result[i, j] = m
                result[j, i] = m
        return result
    return SI

def topological_distance(adj_matrix, **kwargs):
    """
    Compute pairwise cosine similarity between nodes based on their edge patterns.
    
    Args:
        adj_matrix (np.ndarray): Binary adjacency matrix (N x N)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        np.ndarray: Matching index matrix (N x N)
    """
    N = adj_matrix.shape[0]
    matching_matrix = np.zeros((N, N))
    
    for i in range(N):
        edges_i = adj_matrix[i]
        norm_i = np.sqrt(np.sum(edges_i * edges_i))
        
        for j in range(i, N):  # Symmetric matrix, compute upper triangle
            edges_j = adj_matrix[j]
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
            
    return 1-matching_matrix


def matching_distance(adj_matrix, **kwargs):
    """
    Compute pairwise matching index between nodes.
    Matching index = 2 * (shared connections) / (total unshared connections)
    
    Args:
        adj_matrix (np.ndarray): Binary adjacency matrix (N x N)
        **kwargs: Additional arguments (ignored)
        
    Returns:
        np.ndarray: Matching index matrix (N x N)
    """
    N = adj_matrix.shape[0]
    matching_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):  # Compute upper triangle only
            # Get neighbors excluding i and j
            edges_i = adj_matrix[i].copy()
            edges_j = adj_matrix[j].copy()
            
            # Remove mutual connection and self-connections
            edges_i[i] = edges_i[j] = 0
            edges_j[i] = edges_j[j] = 0
            
            # Count shared and total connections (numba-friendly)
            shared = np.sum(np.logical_and(edges_i, edges_j))
            total = np.sum(np.logical_or(edges_i, edges_j))
            
            # Compute matching index
            if total > 0:
                similarity = shared / total
            else:
                similarity = 0.0
                
            # Fill both triangles due to symmetry
            matching_matrix[i, j] = matching_matrix[j, i] = similarity
            
    return 1-matching_matrix
    
def compute_node_payoff(
    node: int,
    adjacency: FloatArray,
    distance_matrix: FloatArray,
    distance_fn: Callable[..., FloatArray],
    alpha: float,
    beta: float,
    noise: float,
    connectivity_penalty: float,
    distance_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute payoff for a single node.
    
    Args:
        node: Index of node
        adjacency: Current adjacency matrix
        distance_matrix: Pre-computed distance matrix (n_nodes, n_nodes) for wiring cost
        distance_fn: Function computing distance metric
        alpha: Weight of distance term
        beta: Weight of wiring cost
        noise: Noise value (0 for deterministic)
        connectivity_penalty: Penalty for disconnected components
        distance_fn_kwargs: Kwargs for distance_fn
            
    Returns:
        Total payoff value
    """
    if distance_fn_kwargs is None:
        distance_fn_kwargs = {}
        
    n_nodes = len(adjacency)
    payoff = 0.0
    
    # Distance term
    if alpha != 0:
        distances = distance_fn(adjacency, **distance_fn_kwargs)
        payoff -= alpha * np.sum(distances.T[node])
        
    # Wiring cost
    if beta != 0:
        # Use pre-computed wiring distance matrix
        euclidean = distance_matrix[node]
        payoff -= beta * np.sum(adjacency[node] * euclidean)
        
    # Connectivity penalty
    if connectivity_penalty != 0:
        comp_sizes = compute_component_sizes(adjacency)
        payoff -= connectivity_penalty * (n_nodes - comp_sizes[node])
        
    # Add noise
    if noise != 0:
        payoff += noise
        
    return payoff

@njit
def sample_nodes_with_centers(
    n_nodes: int,
    centers: FloatArray,
    distance_matrix: FloatArray,
    width: float,
    t: int
) -> tuple[int, int]:
    """
    Sample nodes using gaussian distribution around given centers.
    
    Args:
        n_nodes: Number of nodes in network
        centers: Center nodes for sampling (n_centers, n_iterations) or (1, n_iterations)
        distance_matrix: Pre-computed distance matrix (n_nodes, n_nodes)
        width: Width of gaussian distribution
        t: Current iteration
        
    Returns:
        Tuple of sampled node indices (i, j)
    """
    # Get current centers
    current_centers = centers[:, t]
    n_centers = len(current_centers)
    
    # Randomly select a center
    center_idx = np.random.randint(0, n_centers)
    center = int(current_centers[center_idx])
    
    # Get distances from center to all nodes
    distances = distance_matrix[center]
    
    # Compute gaussian probabilities
    probs = np.exp(-distances**2 / (2 * width**2))
    probs /= np.sum(probs)
    
    # Sample two nodes based on probabilities
    i = np.random.choice(n_nodes, p=probs)
    j = np.random.choice(n_nodes, p=probs)
    
    return i, j

def simulate_network_evolution(
    distance_matrix: FloatArray,
    n_iterations: int,
    distance_fn: DistanceMetric,
    alpha: Trajectory,
    beta: Trajectory,
    noise: NoiseType,
    connectivity_penalty: Trajectory,
    initial_adjacency: Optional[FloatArray] = None,
    n_jobs: int = -1,
    batch_size: Trajectory = 32,
    sampling_centers: Optional[FloatArray] = None,
    sampling_width: Optional[float] = None,
    random_seed: Optional[int] = None,
    symmetric: Optional[bool] = True,
    **kwargs
) -> FloatArray:
    """
    Simulate network evolution through payoff optimization.
    
    Args:
        distance_matrix: Pre-computed distance matrix (n_nodes, n_nodes)
        n_iterations: Number of simulation steps
        distance_fn: Function computing distance metric
        alpha: Weight of distance term (float or array)
        beta: Weight of wiring cost (float or array)
        noise: Noise values (0 or array)
        connectivity_penalty: Penalty for disconnected components
        initial_adjacency: Starting adjacency matrix (optional)
        n_jobs: Number of parallel jobs (-1 for all cores)
        batch_size: Number of edge flips to process in parallel
        sampling_centers: Centers for node sampling (optional)
        sampling_width: Width for sampling distribution (optional)
        random_seed: Random seed for reproducibility
        symmetric: if True (default) it enforces connectivity to the otherside if a connection is accepted by one side.
        **kwargs: Additional arguments passed to internal functions
            - distance_fn_kwargs: Dict of kwargs for distance_fn
        
    Returns:
        History of adjacency matrices (n_nodes, n_nodes, n_iterations)
    """
    # Parameter validation
    validate_parameters(
        n_iterations,
        alpha, beta, noise, connectivity_penalty, batch_size,
        names=('alpha', 'beta', 'noise', 'connectivity_penalty', 'batch_size'),
        allow_float=(True, True, False, True, True),
        allow_zero=(True, True, True, True, False)
    )
    
    if sampling_centers is not None:
        if sampling_width is None:
            raise ValueError("sampling_width must be provided when using sampling_centers")
        if sampling_centers.ndim != 2:
            raise ValueError("sampling_centers must be 2D array (n_centers, n_iterations)")
        if sampling_centers.shape[1] != n_iterations:
            raise ValueError(
                f"sampling_centers length {sampling_centers.shape[1]} doesn't match "
                f"simulation length {n_iterations}"
            )
            
    distance_fn_kwargs = kwargs.get('distance_fn_kwargs', {})
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    n_nodes = len(distance_matrix)
    
    # Initialize adjacency if not provided
    if initial_adjacency is None:
        # Start with ring structure
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
            # Get current parameters
            alpha_t = get_param_value(alpha, t)
            beta_t = get_param_value(beta, t)
            noise_t = get_param_value(noise, t) if isinstance(noise, np.ndarray) else 0
            penalty_t = get_param_value(connectivity_penalty, t)
            
            # Get current batch size
            batch_size_t = get_param_value(batch_size, t)
            
            # Process edge flips in parallel batches
            def process_flip(_):
                if sampling_centers is not None:
                    i, j = sample_nodes_with_centers(
                        n_nodes, sampling_centers, distance_matrix, sampling_width, t
                    )
                else:
                    i, j = np.random.randint(0, n_nodes, size=2)
                if i == j:
                    return None
                    
                # Compute current payoff
                current = compute_node_payoff(
                    i, adjacency, distance_matrix, distance_fn,
                    alpha_t, beta_t, noise_t, penalty_t,
                    distance_fn_kwargs
                )
                
                # Test flip
                adj_test = adjacency.copy()
                adj_test[i, j] = 1 - adj_test[i, j]
                adj_test[j, i] = adj_test[i, j]
                
                # Compute new payoff
                new = compute_node_payoff(
                    i, adj_test, distance_matrix, distance_fn,
                    alpha_t, beta_t, noise_t, penalty_t,
                    distance_fn_kwargs
                )
                
                return {
                    'i': i, 'j': j,
                    'accepted': new > current
                }
                
            # Process batch in parallel
            n_flips = int(batch_size_t)
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_flip)(i) for i in range(n_flips)
            )
            
            # Apply accepted flips
            for result in results:
                if result is not None and result['accepted']:
                    i, j = result['i'], result['j']
                    adjacency[i, j] = 1 - adjacency[i, j]
                    if symmetric:
                        adjacency[j, i] = adjacency[i, j]
                    
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
    original_payoffs: FloatArray,
    distance_fn_kwargs: Dict[str, Any]
) -> tuple[int, int, float, float]:
    """
    Helper function for parallel computation. Computes the payoff change
    for nodes i and j when edge (i, j) is removed. This function is
    intended to be called by joblib.Parallel.

    Assumes compute_node_payoff is available in the module scope.
    """
    # 1. Create the lesioned adjacency matrix (a copy)
    adj_lesioned = original_adjacency.copy()
    adj_lesioned[i, j] = 0
    adj_lesioned[j, i] = 0 # Assuming symmetric adjacency

    # 2. Calculate new payoffs for nodes i and j using the lesioned matrix
    # Calls the compute_node_payoff function assumed to be in this module
    new_payoff_i = compute_node_payoff(
        node=i,
        adjacency=adj_lesioned,
        distance_matrix=distance_matrix,
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        noise=0.0,                # Noise is set to 0 for this analysis
        connectivity_penalty=0.0, # Penalty is set to 0 for this analysis
        distance_fn_kwargs=distance_fn_kwargs      # Pass any extra arguments for the distance function
    )

    new_payoff_j = compute_node_payoff(
        node=j,
        adjacency=adj_lesioned,
        distance_matrix=distance_matrix,
        distance_fn=distance_fn,
        alpha=alpha,
        beta=beta,
        noise=0.0,
        connectivity_penalty=0.0,
        distance_fn_kwargs=distance_fn_kwargs
    )

    # 3. Compute differences from the pre-calculated original payoffs
    diff_i = new_payoff_i - original_payoffs[i]
    diff_j = new_payoff_j - original_payoffs[j]

    # Ensure differences are finite numbers, default to 0.0 otherwise
    diff_i = diff_i if np.isfinite(diff_i) else 0.0
    diff_j = diff_j if np.isfinite(diff_j) else 0.0

    # Return the node indices and their respective payoff differences
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
    Computes the impact on the payoff of connected nodes when their edge is removed.

    Assumes it is part of a module where FloatArray, DistanceMetric, and
    compute_node_payoff are defined.

    Iterates through each existing edge, removes it temporarily, and calculates
    how the payoff changes for the two nodes that were connected by that edge.
    Uses parallel processing to speed up the computation for multiple edges.

    Args:
        distance_matrix: Pre-computed distance matrix (n_nodes, n_nodes).
        adjacency_matrix: The binary, symmetric adjacency matrix (n_nodes, n_nodes).
                          Assumes 1 for connection, 0 otherwise.
        distance_fn: Function that computes the 'communication' distance metric used in payoff.
                     Expected signature: distance_fn(adj, **kwargs) -> distance_matrix.
        alpha: Weight parameter for the distance term in the payoff function.
        beta: Weight parameter for the wiring cost term (Euclidean distance)
              in the payoff function (default: 1.0).
        distance_fn_kwargs: Optional dictionary of keyword arguments to pass to the
                            distance_fn (e.g., {'alpha': 0.8} for propagation distance).
        n_jobs: Number of CPU cores to use for parallel processing.
                -1 means using all available cores (default: -1).

    Returns:
        An N x N numpy array (`payoff_impact_matrix`) where `matrix[i, j]` contains
        the change in payoff for node `i` when the edge `(i, j)` is removed
        (calculated as `new_payoff_i - original_payoff_i`).
        The matrix will be sparse (zeros where no edge existed in the input matrix)
        and generally asymmetric, as the impact on node i might differ from the impact on node j.
    """
    n_nodes = adjacency_matrix.shape[0]
    if distance_fn_kwargs is None:
        distance_fn_kwargs = {}

    # Calculate the initial payoff for all nodes using the original network
    original_payoffs = np.zeros(n_nodes)
    # This loop calculates the baseline payoff before any edges are removed
    for node_idx in range(n_nodes):
         # Calls compute_node_payoff from the module scope
         original_payoffs[node_idx] = compute_node_payoff(
            node=node_idx,
            adjacency=adjacency_matrix, # Use the original adjacency matrix
            distance_matrix=distance_matrix,
            distance_fn=distance_fn,
            alpha=alpha,
            beta=beta,
            noise=0.0,                # Analysis assumes no noise
            connectivity_penalty=0.0, # Analysis assumes no connectivity penalty
            distance_fn_kwargs=distance_fn_kwargs      # Pass extra args to distance function if any
        )

    # --- 2. Parallel Edge Lesioning ---
    # Find the indices (row, col) of existing edges in the upper triangle (k=1 excludes diagonal)
    edge_indices = np.argwhere(np.triu(adjacency_matrix, k=1) > 0)
    num_edges = len(edge_indices)

    if num_edges == 0:
        print("  No edges found in the adjacency matrix. Returning zero matrix.")
        return np.zeros((n_nodes, n_nodes))
    # Use joblib.Parallel to run the helper function for each edge
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_impact_for_edge)(
            i, j, # Pass the indices of the edge
            original_adjacency=adjacency_matrix,
            distance_matrix=distance_matrix,
            distance_fn=distance_fn,
            alpha=alpha,
            beta=beta,
            original_payoffs=original_payoffs, # Pass the precomputed original payoffs
            distance_fn_kwargs=distance_fn_kwargs
        ) for i, j in edge_indices # Iterate directly over the found edge indices
    )

    # --- 3. Aggregation ---
    # Initialize the output matrix with zeros
    payoff_impact_matrix = np.zeros((n_nodes, n_nodes))
    # Populate the matrix using the results from the parallel computation
    for i, j, diff_i, diff_j in results:
        # Store the impact on node i at matrix[i, j]
        payoff_impact_matrix[i, j] = diff_i
        # Store the impact on node j at matrix[j, i]
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
    **kwargs
) -> Dict[str, Any]:
    """
    Find the optimal alpha value that produces a network with density closest to empirical.
    It uses a bisection search with linear interpolation.
    
    Args:
        distance_matrix: Precomputed distance matrix (n_nodes, n_nodes)
        empirical_connectivity: Target connectivity matrix to match density with
        distance_fn: Distance metric function 
        n_iterations: Number of iterations for each simulation
        beta: Wiring cost parameter
        alpha_range: Range for alpha search (min, max)
        tolerance: Acceptable difference between densities
        max_search_iterations: Maximum number of search iterations
        random_seed: Random seed for reproducibility
        batch_size: Batch size for parallel processing
        symmetric: ensures symmetry.
        n_jobs: Number of parallel jobs
        **kwargs: Additional arguments passed to simulate_network_evolution
        
    Returns:
        Dictionary containing:
            - 'alpha': Optimal alpha value
            - 'density': Density of the resulting network
            - 'evolution': Full history of adjacency matrices (n_nodes, n_nodes, n_iterations)
    """
    # Calculate empirical density
    n_nodes = empirical_connectivity.shape[0]
    empirical_density = np.sum(empirical_connectivity) / (n_nodes * (n_nodes - 1))
    
    # Set up fixed parameters as vectors
    beta_vec = np.full(n_iterations, beta)
    noise_vec = np.zeros(n_iterations)
    penalty_vec = np.zeros(n_iterations)
    batch_size_vec = np.full(n_iterations, batch_size)
    
    # Function to simulate network and get density
    def simulate_with_alpha(alpha_value):
        alpha_vec = np.full(n_iterations, alpha_value)
        network = simulate_network_evolution(
            distance_matrix=distance_matrix,
            n_iterations=n_iterations,
            distance_fn=distance_fn,
            alpha=alpha_vec,
            beta=beta_vec,
            noise=noise_vec,
            connectivity_penalty=penalty_vec,
            n_jobs=n_jobs,
            random_seed=random_seed,
            batch_size=batch_size_vec,
            symmetric=symmetric,
            **kwargs
        )
        
        # Get final network
        final_adj = network[:, :, -1]
        density = np.sum(final_adj) / (n_nodes * (n_nodes - 1))
        
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