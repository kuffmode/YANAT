
import numpy as np
import numba as nb
from numba import njit, prange
from typing import Optional, Tuple, Union, Callable, Dict, Any, List
from tqdm import tqdm

__all__ = [
    "propagation_distance",
    "resistance_distance",
    "heat_kernel_distance",
    "shortest_path_distance",
    "topological_distance",
    "compute_node_payoff",
    "simulate_network_evolution",
    "find_optimal_alpha",
    "validate_parameters"
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def _power_iteration(A: np.ndarray, num_iters: int = 100, eps: float = 1e-6) -> float:
    """
    Computes spectral radius using power iteration.
    Robust to asymmetric matrices by tracking norm growth.
    """
    n = A.shape[0]
    v = np.ones(n, dtype=np.float64)

    for i in range(n):
        v[i] += 1e-3 * (i % 2)
        
    norm_v = np.linalg.norm(v)
    if norm_v == 0: return 0.0
    v = v / norm_v
    
    prev_rho = 0.0
    rho = 0.0
    
    for _ in range(num_iters):
        v_next = A @ v
        norm_next = np.linalg.norm(v_next)
        
        if norm_next < 1e-12:
            return 0.0
            
        v = v_next / norm_next
        rho = norm_next
        
        if np.abs(rho - prev_rho) < eps * rho:
            return rho
            
        prev_rho = rho
        
    return rho

@njit(fastmath=True)
def _spectral_normalization(adjacency: np.ndarray, target_radius: float = 1.0) -> np.ndarray:
    # Use power iteration for speed (O(N^2) vs O(N^3))
    spec_rad = _power_iteration(adjacency)
    
    if spec_rad > 1e-9:
        return adjacency * (target_radius / spec_rad)
    return adjacency

@njit(fastmath=True)
def _apply_weighting(adjacency: np.ndarray, distance_matrix: np.ndarray, weight_coefficient: float) -> np.ndarray:
    if weight_coefficient == 0.0:
        return adjacency.copy()
    weights = np.exp(-weight_coefficient * distance_matrix)
    return adjacency * weights

@njit(fastmath=True)
def _log_normalize_neg(matrix: np.ndarray) -> np.ndarray:
    """Computes -log(matrix) safely."""
    n = matrix.shape[0]
    res = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 1e-20:
                res[i, j] = -np.log(val)
            else:
                res[i, j] = 0.0
    return res

@njit(fastmath=True)
def _get_component_size(adjacency: np.ndarray, node: int) -> int:
    """
    Computes the size of the connected component containing the given node using BFS.
    """
    n = len(adjacency)
    visited = np.zeros(n, dtype=np.bool_)
    q = np.zeros(n, dtype=np.int64)
    head = 0
    tail = 0
    
    q[tail] = node
    tail += 1
    visited[node] = True
    
    count = 0
    while head < tail:
        u = q[head]
        head += 1
        count += 1
        for v in range(n):
            if adjacency[u, v] > 0 and not visited[v]:
                visited[v] = True
                q[tail] = v
                tail += 1
                
    return count

# -----------------------------------------------------------------------------
# Distance Functions
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def propagation_distance(
    adjacency_matrix: np.ndarray, 
    spatial_decay: float = 0.8, 
    symmetric: bool = True, 
    distance_matrix: Optional[np.ndarray] = None, 
    weight_coefficient: float = 0.0
) -> np.ndarray:
    """
    Computes the propagation distance matrix using Numba. 
    
    NOTE: This is a bit unstable under Numba with larger spatial decay values. 
    Try heat kernel distance for symmetric graphs, they're highly correlated.

    The propagation distance is defined as the negative log of the influence matrix.
    The influence matrix is computed using either the LAM (Linear Attenuation Model)
    or SAR (Spatial Autoregressive) model.

    Args:
        adjacency_matrix: The adjacency matrix (n_nodes, n_nodes).
        spatial_decay: Decay parameter (0 < spatial_decay < 1/spectral_radius).
        symmetric: If True, uses SAR model (symmetric influence). 
                   If False, uses LAM model (directed influence).
        distance_matrix: The distance matrix (n_nodes, n_nodes). Required if weight_coefficient > 0.
        weight_coefficient: The decay coefficient for distance weighting.

    Returns:
        The propagation distance matrix (n_nodes, n_nodes).
    """
    if distance_matrix is not None:
        weighted_adj = _apply_weighting(adjacency_matrix, distance_matrix, weight_coefficient)
    else:
        weighted_adj = adjacency_matrix.copy()
        
    weighted_adj = _spectral_normalization(weighted_adj, 1.0)
    
    n = len(adjacency_matrix)
    I = np.eye(n, dtype=np.float64)
    A_sys = I - spatial_decay * weighted_adj
    
    # Invert
    inv_matrix = np.linalg.solve(A_sys, I)
    
    if symmetric:
        influence = inv_matrix @ inv_matrix.T
    else:
        influence = inv_matrix
        
    return _log_normalize_neg(influence)

@njit(fastmath=True)
def resistance_distance(
    adjacency: np.ndarray, 
    distance_matrix: Optional[np.ndarray] = None, 
    weight_coefficient: float = 0.0
) -> np.ndarray:
    """
    Computes resistance distances between all pairs of nodes using Numba.

    The resistance distance is computed using the Moore-Penrose pseudoinverse
    of the Laplacian matrix. It treats the graph as an electrical network
    where edges are resistors.

    Args:
        adjacency: The adjacency matrix (n_nodes, n_nodes).
        distance_matrix: The distance matrix (n_nodes, n_nodes). Required if weight_coefficient > 0.
        weight_coefficient: The decay coefficient for distance weighting.

    Returns:
        The resistance distance matrix (n_nodes, n_nodes).
    """
    if distance_matrix is not None:
        weighted_adj = _apply_weighting(adjacency, distance_matrix, weight_coefficient)
    else:
        weighted_adj = adjacency.copy()
        
    # Symmetrize
    W = (weighted_adj + weighted_adj.T) / 2.0
    
    degree = np.sum(W, axis=1)
    L = np.diag(degree) - W
    
    # Pseudoinverse
    pinv = np.linalg.pinv(L)
    
    diag_pinv = np.diag(pinv)
    
    n = len(adjacency)
    resistance = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = diag_pinv[i] + diag_pinv[j] - 2 * pinv[i, j]
            if val < 0: val = 0.0
            if i == j: val = 0.0
            resistance[i, j] = val
            
    return resistance

@njit(fastmath=True)
def heat_kernel_distance(
    adjacency_matrix: np.ndarray, 
    t: float = 0.5, 
    eps: float = 1e-10, 
    distance_matrix: Optional[np.ndarray] = None, 
    weight_coefficient: float = 0.0
) -> np.ndarray:
    """
    Computes the heat kernel distance matrix at diffusion time t using Numba.

    The heat kernel distance is defined as -log(exp(-t * L)), where L is the Laplacian.

    Args:
        adjacency_matrix: The adjacency matrix (n_nodes, n_nodes).
        t: Diffusion time parameter.
        eps: Small constant to avoid log(0).
        distance_matrix: The distance matrix (n_nodes, n_nodes). Required if weight_coefficient > 0.
        weight_coefficient: The decay coefficient for distance weighting.

    Returns:
        The heat kernel distance matrix (n_nodes, n_nodes).
    """
    if distance_matrix is not None:
        weighted_adj = _apply_weighting(adjacency_matrix, distance_matrix, weight_coefficient)
    else:
        weighted_adj = adjacency_matrix.copy()
        
    degree = np.sum(weighted_adj, axis=1)
    L = np.diag(degree) - weighted_adj
    
    # Use eigh for symmetric matrices (Laplacian is symmetric for undirected)
    # If directed, this might be wrong, but heat kernel is typically for undirected.
    # Numba supports eigh.
    vals, vecs = np.linalg.eigh(L)
    
    # exp(-t * L) = V * diag(exp(-t * vals)) * V.T
    exp_vals = np.exp(-t * vals)
    
    # kernel = vecs @ diag(exp_vals) @ vecs.T
    # Optimized: (vecs * exp_vals) @ vecs.T
    kernel = (vecs * exp_vals) @ vecs.T
    
    # -log
    n = len(adjacency_matrix)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            val = kernel[i, j]
            if val < eps: val = eps
            dist[i, j] = -np.log(val)
            
    return dist

@njit(fastmath=True)
def shortest_path_distance(
    adjacency_matrix: np.ndarray, 
    distance_matrix: Optional[np.ndarray] = None, 
    weight_coefficient: float = 0.0
) -> np.ndarray:
    """
    Computes shortest-path distances between all pairs of nodes using Floyd-Warshall (Numba).

    Args:
        adjacency_matrix: The adjacency matrix (n_nodes, n_nodes).
        distance_matrix: The distance matrix (n_nodes, n_nodes). Required if weight_coefficient > 0.
        weight_coefficient: The decay coefficient for distance weighting.

    Returns:
        The shortest path distance matrix (n_nodes, n_nodes).
    """
    if distance_matrix is not None:
        weighted_adj = _apply_weighting(adjacency_matrix, distance_matrix, weight_coefficient)
    else:
        weighted_adj = adjacency_matrix.copy()
        
    n = len(adjacency_matrix)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    
    # Initialize
    for i in range(n):
        dist[i, i] = 0.0
        for j in range(n):
            w = weighted_adj[i, j]
            if w > 0:
                dist[i, j] = 1.0 / w
                
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d = dist[i, k] + dist[k, j]
                if d < dist[i, j]:
                    dist[i, j] = d
                    
    return dist

@njit(fastmath=True)
def topological_distance(
    adj_matrix: np.ndarray, 
    distance_matrix: Optional[np.ndarray] = None, 
    weight_coefficient: float = 0.0
) -> np.ndarray:
    """
    Computes pairwise topological distance based on cosine similarity of neighbors using Numba.

    Returns 1 - cosine_similarity. Nodes with similar connectivity patterns 
    will have small topological distance.

    Args:
        adj_matrix: The adjacency matrix (n_nodes, n_nodes).
        distance_matrix: The distance matrix (n_nodes, n_nodes). Required if weight_coefficient > 0.
        weight_coefficient: The decay coefficient for distance weighting.

    Returns:
        The topological distance matrix (n_nodes, n_nodes).
    """
    if distance_matrix is not None:
        weighted_adj = _apply_weighting(adj_matrix, distance_matrix, weight_coefficient)
    else:
        weighted_adj = adj_matrix.copy()
        
    n = len(adj_matrix)
    dist = np.zeros((n, n), dtype=np.float64)
    
    # Precompute norms
    norms = np.zeros(n, dtype=np.float64)
    for i in range(n):
        norms[i] = np.sqrt(np.sum(weighted_adj[i]**2))
        
    for i in range(n):
        for j in range(i, n):
            if norms[i] == 0 or norms[j] == 0:
                sim = 0.0
            else:
                dot = np.sum(weighted_adj[i] * weighted_adj[j])
                sim = dot / (norms[i] * norms[j])
            
            d = 1.0 - sim
            dist[i, j] = d
            dist[j, i] = d
            
    return dist

# -----------------------------------------------------------------------------
# Simulation Logic
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def compute_node_payoff(
    node: int,
    adjacency: np.ndarray,
    distance_matrix: np.ndarray,
    distance_fn_type: int, # 0: prop, 1: res, 2: heat, 3: sp, 4: topo
    alpha: float,
    beta: float,
    connectivity_penalty: float,
    node_resources: Optional[np.ndarray],
    # Distance fn params
    spatial_decay: float,
    symmetric: bool,
    weight_coefficient: float,
    t_param: float
) -> float:
    """
    Computes the payoff for a single node based on distance, wiring cost, and connectivity.

    The payoff is defined as:
    Payoff = - (alpha * distance_term + beta * wiring_cost + connectivity_penalty * disconnected_nodes)

    Args:
        node: Index of the node.
        adjacency: The current adjacency matrix (n_nodes, n_nodes).
        distance_matrix: Pre-computed Euclidean distance matrix (n_nodes, n_nodes).
        distance_fn_type: Integer code for distance function (0: prop, 1: res, 2: heat, 3: sp, 4: topo).
        alpha: Weight of the distance term.
        beta: Weight of the wiring cost term.
        connectivity_penalty: Penalty for disconnected components.
        node_resources: Optional vector of node resources.
        spatial_decay: Parameter for propagation distance.
        symmetric: Parameter for propagation distance.
        weight_coefficient: Parameter for distance weighting.
        t_param: Parameter for heat kernel distance.

    Returns:
        The calculated payoff value.
    """
    
    # Dispatch
    if distance_fn_type == 0:
        dists = propagation_distance(adjacency, spatial_decay, symmetric, distance_matrix, weight_coefficient)
    elif distance_fn_type == 1:
        dists = resistance_distance(adjacency, distance_matrix, weight_coefficient)
    elif distance_fn_type == 2:
        dists = heat_kernel_distance(adjacency, t_param, 1e-10, distance_matrix, weight_coefficient)
    elif distance_fn_type == 3:
        dists = shortest_path_distance(adjacency, distance_matrix, weight_coefficient)
    else:
        dists = topological_distance(adjacency, distance_matrix, weight_coefficient)
        
    # Payoff
    comm_cost = np.sum(dists[node])
    
    euclidean = distance_matrix[node]
    if node_resources is not None:
        effective_dist = np.maximum(0.0, euclidean - node_resources[node])
        wiring_cost = np.sum(adjacency[node] * effective_dist)
    else:
        wiring_cost = np.sum(adjacency[node] * euclidean)
        
    payoff = - (alpha * comm_cost + beta * wiring_cost)
    
    if connectivity_penalty != 0:
        comp_size = _get_component_size(adjacency, node)
        payoff -= connectivity_penalty * (len(adjacency) - comp_size)
        
    return payoff

@njit(parallel=True, fastmath=True)
def _evaluate_candidates_batch(
    current_adj: np.ndarray,
    current_payoffs: np.ndarray,
    candidates_u: np.ndarray,
    candidates_v: np.ndarray,
    distance_matrix: np.ndarray,
    alpha: float,
    beta: float,
    connectivity_penalty: float,
    node_resources: np.ndarray,
    distance_fn_type: int,
    spatial_decay: float,
    symmetric: bool,
    weight_coefficient: float,
    t_param: float,
    tolerance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    n_candidates = len(candidates_u)
    is_beneficial = np.zeros(n_candidates, dtype=np.bool_)
    n_nodes = len(current_adj)
    
    for k in prange(n_candidates):
        u = candidates_u[k]
        v = candidates_v[k]
        
        # Copy adj
        cand_adj = current_adj.copy()
        val = 1.0 - cand_adj[u, v]
        cand_adj[u, v] = val
        if symmetric:
            cand_adj[v, u] = val
            
        # Compute new payoffs for u and v
        payoff_u = 0.0
        payoff_v = 0.0
        
        # Wiring
        if beta != 0:
            euclidean_u = distance_matrix[u]
            euclidean_v = distance_matrix[v]
            
            effective_dist_u = np.maximum(0.0, euclidean_u - node_resources[u])
            effective_dist_v = np.maximum(0.0, euclidean_v - node_resources[v])
            
            wiring_u = np.sum(cand_adj[u] * effective_dist_u)
            wiring_v = np.sum(cand_adj[v] * effective_dist_v)
            
            payoff_u -= beta * wiring_u
            payoff_v -= beta * wiring_v
        
        # Comm
        if alpha != 0:
            if distance_fn_type == 0:
                dists = propagation_distance(cand_adj, spatial_decay, symmetric, distance_matrix, weight_coefficient)
            elif distance_fn_type == 1:
                dists = resistance_distance(cand_adj, distance_matrix, weight_coefficient)
            elif distance_fn_type == 2:
                dists = heat_kernel_distance(cand_adj, t_param, 1e-10, distance_matrix, weight_coefficient)
            elif distance_fn_type == 3:
                dists = shortest_path_distance(cand_adj, distance_matrix, weight_coefficient)
            else:
                dists = topological_distance(cand_adj, distance_matrix, weight_coefficient)
            
            comm_u = np.sum(dists[u])
            comm_v = np.sum(dists[v])
            
            payoff_u -= alpha * comm_u
            payoff_v -= alpha * comm_v
            
        new_payoff_u = payoff_u
        new_payoff_v = payoff_v
        
        # Connectivity penalty
        if connectivity_penalty != 0:

            comp_size_u = _get_component_size(cand_adj, u)
            new_payoff_u -= connectivity_penalty * (n_nodes - comp_size_u)
            
            comp_size_v = _get_component_size(cand_adj, v)
            new_payoff_v -= connectivity_penalty * (n_nodes - comp_size_v)
        
        diff_u = new_payoff_u - current_payoffs[u]
        diff_v = new_payoff_v - current_payoffs[v]
        
        if diff_u > tolerance or diff_v > tolerance:
            is_beneficial[k] = True
            
    return candidates_u, candidates_v, is_beneficial

@njit(fastmath=True)
def _compute_all_payoffs(
    adj: np.ndarray, 
    d_mat: np.ndarray, 
    a: float, 
    b: float, 
    cp: float,
    node_res: np.ndarray,
    d_type: int, 
    s_decay: float, 
    sym: bool, 
    w_coef: float, 
    t_par: float
) -> np.ndarray:
    n = len(adj)
    payoffs = np.zeros(n, dtype=np.float64)
    
    if a != 0:
        if d_type == 0:
            dists = propagation_distance(adj, s_decay, sym, d_mat, w_coef)
        elif d_type == 1:
            dists = resistance_distance(adj, d_mat, w_coef)
        elif d_type == 2:
            dists = heat_kernel_distance(adj, t_par, 1e-10, d_mat, w_coef)
        elif d_type == 3:
            dists = shortest_path_distance(adj, d_mat, w_coef)
        else:
            dists = topological_distance(adj, d_mat, w_coef)
            
        comm = np.sum(dists, axis=1)
        payoffs -= a * comm
    
    # Wiring with resources
    if b != 0:
        wiring = np.zeros(n, dtype=np.float64)
        for i in range(n):
            eff_dist = np.maximum(0.0, d_mat[i] - node_res[i])
            wiring[i] = np.sum(adj[i] * eff_dist)
        payoffs -= b * wiring
    
    if cp != 0:
        for i in range(n):
            sz = _get_component_size(adj, i)
            payoffs[i] -= cp * (n - sz)
            
    return payoffs

def simulate_network_evolution(
    distance_matrix: np.ndarray,
    n_iterations: int,
    distance_fn: Union[str, Callable] = "propagation", # String or dummy
    alpha: Union[float, np.ndarray] = 1.0,
    beta: Union[float, np.ndarray] = 1.0,
    connectivity_penalty: Union[float, np.ndarray] = 0.0,
    initial_adjacency: Optional[np.ndarray] = None,
    n_jobs: int = -1, # Ignored, used for compat
    batch_size: Union[int, np.ndarray] = 32,
    node_resources: Optional[np.ndarray] = None,
    payoff_tolerance: Union[float, np.ndarray] = 0.0,
    random_seed: Optional[int] = None,
    symmetric: bool = True,
    # Distance fn specific args
    spatial_decay: float = 0.8,
    weight_coefficient: float = 0.0,
    t: float = 0.5, # for heat kernel
    verbose: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Simulates the evolution of a network through game-theoretic payoff optimization using Numba.

    At each step, random edges are selected and "flipped" (added or removed). 
    The change is accepted if it improves the payoff for at least one of the nodes involved 
    (unilateral consent), subject to a tolerance threshold.

    Args:
        distance_matrix: Pre-computed Euclidean distance matrix (n_nodes, n_nodes).
        n_iterations: Total number of simulation steps.
        distance_fn: Name of the distance function ("propagation", "resistance", "heat", "shortest", "topological") or the function itself.
        alpha: Weight of the distance term (float or trajectory array).
        beta: Weight of the wiring cost term (float or trajectory array).
        connectivity_penalty: Penalty for disconnected components (float or trajectory array).
        initial_adjacency: Starting adjacency matrix. If None, starts with a ring lattice.
        n_jobs: Ignored in Numba implementation (uses internal parallelism).
        batch_size: Number of potential edge flips to evaluate per iteration.
        node_resources: Optional resources for each node to subsidize wiring costs.
        payoff_tolerance: Minimum payoff improvement required to accept a change.
        random_seed: Seed for random number generator.
        symmetric: If True, enforces undirected edges (symmetry).
        spatial_decay: Decay parameter for propagation distance.
        weight_coefficient: Weight coefficient for distance weighting.
        t: Time parameter for heat kernel distance.
        verbose: If True, shows progress bar.
        **kwargs: Additional arguments (ignored).

    Returns:
        A 3D array of shape (n_nodes, n_nodes, n_iterations) containing the 
        adjacency matrix at each time step.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Map distance_fn to int
    # 0: prop, 1: res, 2: heat, 3: sp, 4: topo
    if isinstance(distance_fn, str):
        name = distance_fn.lower()
    else:
        name = distance_fn.__name__.lower()
        
    if "propagation" in name: dist_type = 0
    elif "resistance" in name: dist_type = 1
    elif "heat" in name: dist_type = 2
    elif "shortest" in name: dist_type = 3
    elif "topological" in name: dist_type = 4
    else: dist_type = 0 # Default
    
    n_nodes = distance_matrix.shape[0]
    
    if initial_adjacency is None:
        adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        idx = np.arange(n_nodes)
        adjacency[idx, (idx + 1) % n_nodes] = 1.0
        adjacency[(idx + 1) % n_nodes, idx] = 1.0
    else:
        adjacency = initial_adjacency.astype(np.float64).copy()
        
    if node_resources is None:
        node_resources_arr = np.zeros(n_nodes, dtype=np.float64)
    else:
        node_resources_arr = node_resources.astype(np.float64)
        
    history = np.zeros((n_nodes, n_nodes, n_iterations))
    history[:, :, 0] = adjacency
    
    def get_val(param, idx):
        if isinstance(param, (np.ndarray, list)):
            return param[idx]
        return param
    
    # Initial payoffs
    current_payoffs = _compute_all_payoffs(
        adj=adjacency, 
        d_mat=distance_matrix, 
        a=get_val(alpha, 0), 
        b=get_val(beta, 0), 
        cp=get_val(connectivity_penalty, 0),
        node_res=node_resources_arr,
        d_type=dist_type, 
        s_decay=spatial_decay, 
        sym=symmetric, 
        w_coef=weight_coefficient, 
        t_par=t
    )
    
    for step in tqdm(range(1, n_iterations), desc="Simulating network evolution", disable=not verbose):
        a_t = get_val(alpha, step)
        b_t = get_val(beta, step)
        cp_t = get_val(connectivity_penalty, step)
        bs_t = int(get_val(batch_size, step))
        tol_t = get_val(payoff_tolerance, step)
        
        # Recompute current payoffs if params changed
        current_payoffs = _compute_all_payoffs(
            adj=adjacency, 
            d_mat=distance_matrix, 
            a=a_t, 
            b=b_t, 
            cp=cp_t,
            node_res=node_resources_arr,
            d_type=dist_type, 
            s_decay=spatial_decay, 
            sym=symmetric, 
            w_coef=weight_coefficient, 
            t_par=t
        )
        
        u_indices = np.random.randint(0, n_nodes, bs_t)
        v_indices = np.random.randint(0, n_nodes, bs_t)
        
        mask = u_indices == v_indices
        while np.any(mask):
            v_indices[mask] = np.random.randint(0, n_nodes, np.sum(mask))
            mask = u_indices == v_indices
            
        _, _, beneficial = _evaluate_candidates_batch(
            current_adj=adjacency, 
            current_payoffs=current_payoffs, 
            candidates_u=u_indices, 
            candidates_v=v_indices,
            distance_matrix=distance_matrix, 
            alpha=a_t, 
            beta=b_t, 
            connectivity_penalty=cp_t,
            node_resources=node_resources_arr,
            distance_fn_type=dist_type,
            spatial_decay=spatial_decay, 
            symmetric=symmetric, 
            weight_coefficient=weight_coefficient, 
            t_param=t, 
            tolerance=tol_t
        )
        
        if np.any(beneficial):
            idx_ben = np.where(beneficial)[0]
            for k in idx_ben:
                u = u_indices[k]
                v = v_indices[k]
                val = 1.0 - adjacency[u, v]
                adjacency[u, v] = val
                if symmetric:
                    adjacency[v, u] = val
                    
        history[:, :, step] = adjacency
        
    return history


@njit(parallel=True, fastmath=True)
def _compute_impact_batch(
    adjacency: np.ndarray,
    distance_matrix: np.ndarray,
    edges: np.ndarray,
    dist_type: int,
    alpha: float,
    beta: float,
    spatial_decay: float,
    weight_coefficient: float,
    t_param: float
) -> np.ndarray:
    n_edges = len(edges)
    n_nodes = len(adjacency)
    impact_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    
    for k in prange(n_edges):
        u = edges[k, 0]
        v = edges[k, 1]
        
        payoff_u_old = compute_node_payoff(
            u, adjacency, distance_matrix, dist_type, alpha, beta, 0.0, None,
            spatial_decay, True, weight_coefficient, t_param
        )
        payoff_v_old = compute_node_payoff(
            v, adjacency, distance_matrix, dist_type, alpha, beta, 0.0, None,
            spatial_decay, True, weight_coefficient, t_param
        )

        adj_lesioned = adjacency.copy()
        adj_lesioned[u, v] = 0.0
        adj_lesioned[v, u] = 0.0
        
        payoff_u_new = compute_node_payoff(
            u, adj_lesioned, distance_matrix, dist_type, alpha, beta, 0.0, None,
            spatial_decay, True, weight_coefficient, t_param
        )
        payoff_v_new = compute_node_payoff(
            v, adj_lesioned, distance_matrix, dist_type, alpha, beta, 0.0, None,
            spatial_decay, True, weight_coefficient, t_param
        )
        
        impact_matrix[u, v] = payoff_u_new - payoff_u_old
        impact_matrix[v, u] = payoff_v_new - payoff_v_old
        
    return impact_matrix

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
    verbose: bool = True,
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
        verbose: If True, prints search progress.
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
            verbose=verbose,
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
    
    if verbose:
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
        if verbose:
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
        
        if verbose:
            print(f"Iteration {iterations+1}: Testing alpha={alpha_mid}")
        mid_density, mid_net = simulate_with_alpha(alpha_mid)
        tested_points.append({'alpha': alpha_mid, 'density': mid_density, 'evolution': mid_net})
        if verbose:
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