
import numpy as np
import torch
from typing import Optional, Union, Callable, Dict, Any
from tqdm import tqdm
from functools import partial
import yanat.utils as ut
__all__ = [
    "resistance_distance_pt",
    "heat_kernel_distance_pt",
    "simulate_network_evolution_gradient_pytorch",
    "find_optimal_alpha_pt",
]

# -----------------------------------------------------------------------------
# PyTorch Distance Functions
# -----------------------------------------------------------------------------

def resistance_distance_pt(
    adjacency: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Computes resistance distances using PyTorch.
    The resistance distance is computed from the pseudoinverse of the Laplacian.
    """
    W = (adjacency + adjacency.T) / 2.0
    degree = torch.sum(W, axis=1)
    L = torch.diag(degree) - W
    
    pinv_L = torch.linalg.pinv(L)
    
    diag_pinv = torch.diag(pinv_L)
    
    resistance = diag_pinv.view(-1, 1) + diag_pinv.view(1, -1) - 2 * pinv_L
    resistance.clamp_(min=0)
    resistance.fill_diagonal_(0)
    
    return resistance

def heat_kernel_distance_pt(
    adjacency: torch.Tensor,
    t: float = 0.5,
    eps: float = 1e-10,
    **kwargs
) -> torch.Tensor:
    """
    Computes the heat kernel distance matrix at diffusion time t using PyTorch.
    The heat kernel distance is defined as -log(exp(-t * L)), where L is the Laplacian.
    """
    degree = torch.sum(adjacency, axis=1)
    L = torch.diag(degree) - adjacency
    
    # Use eigh for symmetric matrices
    vals, vecs = torch.linalg.eigh(L)
    
    exp_vals = torch.exp(-t * vals)
    
    kernel = (vecs * exp_vals) @ vecs.T
    
    kernel.clamp_(min=eps)
    dist = -torch.log(kernel)
            
    return dist

# -----------------------------------------------------------------------------
# PyTorch Simulation Logic
# -----------------------------------------------------------------------------

def _calculate_payoff_pytorch(
    adjacency: "torch.Tensor",
    distance_matrix: "torch.Tensor",
    alpha: float,
    beta: float,
    distance_fn: Callable,
    optimization_type: str = "global",
    **kwargs
) -> "torch.Tensor":
    """
    Calculates the payoff of the network using PyTorch operations.
    
    Args:
        optimization_type: "global" for total network payoff (scalar),
                           "nodal" for per-node payoff (vector).
    """
    # 1. Wiring Cost Term
    # For global: sum(A * D)
    # For nodal: row_sum(A * D) -> cost for each node i is sum_j(A_ij * D_ij)
    if beta != 0:
        if optimization_type == "nodal":
            # wiring_cost[i] = sum_j (A_ij * D_ij)
            wiring_cost = torch.sum(adjacency * distance_matrix, dim=1)
        else:
            wiring_cost = torch.sum(adjacency * distance_matrix)
    else:
        wiring_cost = 0.0

    # 2. Communication Cost Term
    # For global: sum(dist_matrix) / 2
    # For nodal: sum(dist_matrix, dim=1) -> cost for node i is sum_j(d_ij)
    if alpha != 0:
        dist = distance_fn(adjacency, **kwargs)
        if optimization_type == "nodal":
            comm_cost = torch.sum(dist, dim=1)
        else:
            comm_cost = torch.sum(dist) / 2.0  # Divide by 2 for undirected graphs
    else:
        comm_cost = 0.0

    # Combine
    # Note: Payoff is usually Utility - Cost. Here we are minimizing Cost, so Payoff = -Cost.
    # The original code returned "total_payoff" which was negative cost.
    
    if optimization_type == "nodal":
        # Ensure shapes match
        if isinstance(wiring_cost, float): wiring_cost = torch.zeros(adjacency.shape[0], device=adjacency.device)
        if isinstance(comm_cost, float): comm_cost = torch.zeros(adjacency.shape[0], device=adjacency.device)
        
        total_payoff = - (beta * wiring_cost + alpha * comm_cost)
    else:
        total_payoff = - (beta * wiring_cost + alpha * comm_cost)

    return total_payoff

def simulate_network_evolution_gradient_pytorch(
    distance_matrix: np.ndarray,
    n_iterations: int,
    distance_fn: Union[str, Callable] = "resistance",
    alpha: Union[float, np.ndarray] = 1.0,
    beta: Union[float, np.ndarray] = 1.0,
    learning_rate: float = 0.01,
    initial_adjacency: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
    symmetric: bool = True,
    verbose: bool = True,
    optimization_type: str = "global",
    **kwargs
) -> np.ndarray:
    """
    Simulates network evolution using gradient descent with PyTorch's autograd.
    
    Args:
        optimization_type: "global" (default) or "nodal".
                           "nodal" means each node optimizes its own connections.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Please install it to use this function.")

    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Distance function mapping
    if isinstance(distance_fn, str):
        if "resistance" in distance_fn:
            dist_fn_callable = resistance_distance_pt
        elif "heat" in distance_fn:
            dist_fn_callable = partial(heat_kernel_distance_pt, t=kwargs.get('t', 0.5))
        else:
            raise ValueError(f"Unknown distance function: {distance_fn}")
    else:
        dist_fn_callable = distance_fn

    n_nodes = distance_matrix.shape[0]

    if initial_adjacency is None:
        adj_np = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        idx = np.arange(n_nodes)
        adj_np[idx, (idx + 1) % n_nodes] = 1.0
        adj_np[(idx + 1) % n_nodes, idx] = 1.0
    else:
        adj_np = initial_adjacency.astype(np.float64).copy()

    adjacency = torch.tensor(adj_np, device=device, dtype=torch.float64, requires_grad=True)
    dist_matrix_tensor = torch.tensor(distance_matrix, device=device, dtype=torch.float64)

    optimizer = torch.optim.SGD([adjacency], lr=learning_rate)

    history = np.zeros((n_nodes, n_nodes, n_iterations))
    history[:, :, 0] = adj_np

    def get_val(param, idx):
        return param[idx] if isinstance(param, (np.ndarray, list)) else param

    iterator = tqdm(range(1, n_iterations), desc=f"Simulating ({optimization_type} Gradient PyTorch)", disable=not verbose)
    for step in iterator:
        optimizer.zero_grad()

        a_t = get_val(alpha, step)
        b_t = get_val(beta, step)

        payoff = _calculate_payoff_pytorch(
            adjacency, 
            dist_matrix_tensor, 
            a_t, 
            b_t, 
            dist_fn_callable, 
            optimization_type=optimization_type,
            **kwargs
        )
        
        if optimization_type == "global":
            loss = -payoff
            loss.backward()
        elif optimization_type == "nodal":
            # For nodal optimization, we need to compute gradients for each node's payoff
            # independently. Specifically, node i only controls row i of the adjacency matrix.
            # So we want grad_i = d(Payoff_i) / d(Adjacency_i).
            
            # We can accumulate gradients manually.
            # Warning: This is computationally expensive (N backward passes).
            # Optimization: We can use torch.autograd.grad or backward with a mask, 
            # but since Payoff_i depends on the whole matrix (due to distance calc),
            # we effectively have N separate loss functions.
            
            # However, notice that if we just sum the payoffs and backward, 
            # d(Sum Payoff_i) / d(A_jk) = sum_i d(Payoff_i) / d(A_jk).
            # This is NOT what we want. We want node k to update A_kj based ONLY on Payoff_k.
            # i.e., the gradient for A_kj should be d(Payoff_k) / d(A_kj).
            # (Assuming undirected, A_kj = A_jk, so both nodes update the link).
            
            # Actually, in a game theoretic sense for undirected graphs:
            # If node i wants to change A_ij, it optimizes Payoff_i.
            # If node j wants to change A_ji, it optimizes Payoff_j.
            # Since A_ij = A_ji, this is a cooperative or non-cooperative game.
            # Standard "gradient dynamics" usually implies:
            # dA_ij/dt = d(Payoff_i)/dA_ij + d(Payoff_j)/dA_ji
            # This is equivalent to optimizing the potential function if one exists.
            # But here we want strictly nodal gradients.
            
            # Let's implement the strict definition:
            # Gradient for A_ij comes from Node i's desire (dPi/dAij) AND Node j's desire (dPj/dAji).
            # So we can actually just sum the payoffs?
            # If we sum P = sum(P_i), then dP/dA_ij = sum_k dP_k/dA_ij.
            # This includes "externalities" (how my link affects others).
            # Pure selfish agents DO NOT care about externalities.
            # So A_ij should update by dP_i/dA_ij + dP_j/dA_ji.
            # It should NOT include dP_k/dA_ij for k != i, j.
            
            # So we cannot just backward sum(payoff).
            # We need to compute the Jacobian or do N backward passes.
            # Given N might be large, N backward passes is slow.
            # But let's do it for correctness first as requested.
            
            # To speed this up, we can use a hook or Jacobian-vector product, 
            # but let's stick to the loop for clarity and correctness first.
            
            # Actually, we can do it in one pass if we are clever?
            # No, because the dependency is complex (matrix inversion).
            
            total_grad = torch.zeros_like(adjacency)
            
            # Iterate over nodes to compute selfish gradients
            for i in range(n_nodes):
                # We only need the gradient of Payoff_i w.r.t Adjacency
                # We can retain_graph=True for all but the last one.
                
                # Optimization: We only care about row i (and col i due to symmetry).
                # But dP_i/dA_jk might be non-zero for k!=i (indirect effects).
                # But node i cannot control A_jk. Node i only controls A_i*.
                # So we only accumulate gradients into row i.
                
                # To avoid N full backward passes, maybe we can use vmap in newer pytorch?
                # For now, standard loop.
                
                grad_i = torch.autograd.grad(
                    payoff[i], 
                    adjacency, 
                    retain_graph=True, 
                    create_graph=False
                )[0]
                
                # Node i controls row i.
                # In undirected case, A_ij is shared. 
                # Usually we assume A_ij is controlled by both? 
                # Or we update A_ij based on dP_i/dA_ij + dP_j/dA_ji.
                
                # Let's add the contribution of node i to the gradient of row i.
                total_grad[i, :] += grad_i[i, :]
                
                # If symmetric, A_ji is the same variable as A_ij.
                # If we treat A as a symmetric matrix variable, 
                # then dP_i / dA_ij contributes to the update of that unique edge.
                
                # If we treat A as full matrix enforced symmetric later:
                # Node i pushes A_ij with force dP_i/dA_ij.
                # Node j pushes A_ji with force dP_j/dA_ji.
                # Since A_ij and A_ji are coupled (or will be symmetrized), 
                # we effectively sum these forces.
                
                # So: total_grad[i, :] += grad_i[i, :] is correct for the "row player" perspective.
                # And since we iterate all i, total_grad[j, :] will get added dP_j/dA_ji.
                # Finally we symmetrize the gradient or the matrix.
                
            adjacency.grad = -total_grad # We want to MAXIMIZE payoff, so gradient ascent. 
                                         # Optimizer minimizes, so we use negative gradient.
                                         # (Or passed negative payoff? No, payoff is positive utility usually.
                                         # Wait, earlier I defined payoff = -cost.
                                         # So we want to MAXIMIZE payoff (minimize cost).
                                         # SGD minimizes. So we want to minimize (-payoff).
                                         # Gradient of (-payoff) is -(gradient of payoff).
                                         # So adjacency.grad should be -total_grad.
            
            optimizer.step()
            
            # Clean up graph
            # (Not needed explicitly as we didn't create graph for optimizer step, 
            # and we re-compute forward next iter)

        with torch.no_grad():
            adjacency.fill_diagonal_(0)
            adjacency.clamp_(0, 1)
            if symmetric:
                adjacency.data = (adjacency.data + adjacency.data.T) / 2.0

        history[:, :, step] = adjacency.detach().cpu().numpy()

    return history

def find_optimal_alpha_pt(
    distance_matrix: np.ndarray,
    empirical_connectivity: np.ndarray,
    distance_fn: Union[str, Callable],
    n_iterations: int = 10000,
    beta: float = 1.0,
    alpha_range: tuple[float, float] = (1.0, 100.0),
    tolerance: float = 0.01,
    max_search_iterations: int = 20,
    random_seed: int = 11,
    symmetric: bool = True,
    verbose: bool = True,
    learning_rate: float = 0.01,
    **kwargs
) -> Dict[str, Any]:
    """
    Finds the optimal alpha value that produces a network with density closest to empirical,
    using the PyTorch-based gradient simulation.

    Args:
        distance_matrix: Precomputed distance matrix (n_nodes, n_nodes).
        empirical_connectivity: Target connectivity matrix to match density with.
        distance_fn: Distance metric function ('resistance' or 'heat_kernel').
        n_iterations: Number of iterations for each simulation.
        beta: Wiring cost parameter.
        alpha_range: Range for alpha search (min, max).
        tolerance: Acceptable difference between densities.
        max_search_iterations: Maximum number of search iterations.
        random_seed: Random seed for reproducibility.
        symmetric: If True, enforces symmetry in generated networks.
        verbose: If True, prints search progress.
        learning_rate: Learning rate for the gradient simulation.
        **kwargs: Additional arguments passed to the simulation and distance functions.

    Returns:
        Dictionary containing:
            - 'alpha': Optimal alpha value.
            - 'density': Density of the resulting network.
            - 'evolution': Full history of adjacency matrices.
    """
    n_nodes = empirical_connectivity.shape[0]
    empirical_density = np.sum(empirical_connectivity.astype(bool).astype(int)) / (n_nodes * (n_nodes - 1))
    
    def simulate_with_alpha(alpha_value):
        alpha_vec = np.full(n_iterations, alpha_value)
        network = simulate_network_evolution_gradient_pytorch(
            distance_matrix=distance_matrix,
            n_iterations=n_iterations,
            distance_fn=distance_fn,
            alpha=alpha_vec,
            beta=beta,
            random_seed=random_seed,
            symmetric=symmetric,
            verbose=False, # Disable inner loop verbose
            learning_rate=learning_rate,
            **kwargs
        )
        
        final_adj = network[:, :, -1]
        density = ut.find_density(final_adj)
        
        return density, network
    
    alpha_min, alpha_max = alpha_range
    min_density, min_net = simulate_with_alpha(alpha_min)
    max_density, max_net = simulate_with_alpha(alpha_max)
    
    if verbose:
        print(f"Initial range: alpha=[{alpha_min:.4f}, {alpha_max:.4f}], density=[{min_density:.4f}, {max_density:.4f}]")
        print(f"Target density: {empirical_density:.4f}")
    
    tested_points = [
        {'alpha': alpha_min, 'density': min_density, 'evolution': min_net},
        {'alpha': alpha_max, 'density': max_density, 'evolution': max_net}
    ]
    
    if (min_density > empirical_density and max_density > empirical_density) or \
       (min_density < empirical_density and max_density < empirical_density):
        if verbose:
            print("Warning: Initial range does not bracket the target density!")
        return min(tested_points, key=lambda p: abs(p['density'] - empirical_density))
    
    for i in range(max_search_iterations):
        if (max_density - min_density) == 0:
             alpha_mid = (alpha_min + alpha_max) / 2
        else:
            alpha_mid = alpha_min + (alpha_max - alpha_min) * \
                        (empirical_density - min_density) / (max_density - min_density)
        
        if not (alpha_min < alpha_mid < alpha_max):
            alpha_mid = (alpha_min + alpha_max) / 2
        
        if verbose:
            print(f"Iteration {i+1}/{max_search_iterations}: Testing alpha={alpha_mid:.4f}")
            
        mid_density, mid_net = simulate_with_alpha(alpha_mid)
        tested_points.append({'alpha': alpha_mid, 'density': mid_density, 'evolution': mid_net})
        
        if verbose:
            print(f"  alpha={alpha_mid:.4f} -> density={mid_density:.4f} (target diff: {mid_density - empirical_density:.4f})")
        
        if abs(mid_density - empirical_density) < tolerance:
            if verbose:
                print(f"Found optimal alpha: {alpha_mid:.4f}")
            return {'alpha': alpha_mid, 'density': mid_density, 'evolution': mid_net}
        
        if mid_density < empirical_density:
            alpha_min, min_density = alpha_mid, mid_density
        else:
            alpha_max, max_density = alpha_mid, mid_density
            
        if abs(alpha_max - alpha_min) < 1e-3:
            if verbose:
                print("Alpha range is very small, stopping search.")
            break
            
    best_result = min(tested_points, key=lambda p: abs(p['density'] - empirical_density))
    if verbose:
        print(f"Max iterations reached. Best alpha found: {best_result['alpha']:.4f} with density {best_result['density']:.4f}")
        
    return best_result
