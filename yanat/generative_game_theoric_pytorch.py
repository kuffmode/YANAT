
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
    "simulate_network_evolution_differentiable",
    "find_optimal_alpha_pt",
    "find_optimal_alpha_differentiable",
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
            # Optimized implementation using torch.func.jacrev for parallel Jacobian computation
            # This avoids the slow loop over N nodes with N backward passes.
            
            # Define functional wrapper for payoff calculation
            def functional_payoff(adj):
                return _calculate_payoff_pytorch(
                    adj, 
                    dist_matrix_tensor, 
                    a_t, 
                    b_t, 
                    dist_fn_callable, 
                    optimization_type="nodal",
                    **kwargs
                )
            
            # Compute Jacobian: shape (N_payoff, N_adj_rows, N_adj_cols)
            # jac[i, j, k] = d(Payoff_i) / d(A_jk)
            jac = torch.func.jacrev(functional_payoff)(adjacency)
            
            # We want total_grad[i, k] = d(Payoff_i) / d(A_ik)
            # This corresponds to the diagonal elements jac[i, i, k]
            # torch.diagonal(jac, dim1=0, dim2=1) gives shape (N_cols, N_nodes) -> (k, i)
            # So we transpose to get (i, k)
            total_grad = torch.diagonal(jac, dim1=0, dim2=1).T
            
            adjacency.grad = -total_grad
            
            optimizer.step()

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

def simulate_network_evolution_differentiable(
    distance_matrix: torch.Tensor,
    n_iterations: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    distance_fn: Callable,
    learning_rate: float = 0.01,
    initial_adjacency: Optional[torch.Tensor] = None,
    symmetric: bool = True,
    optimization_type: str = "global",
    optimizer_type: str = "sgd",
    **kwargs
) -> torch.Tensor:
    """
    Differentiable simulation of network evolution.
    Returns the final adjacency matrix as a tensor with grad history.
    """
    n_nodes = distance_matrix.shape[0]
    
    if initial_adjacency is None:
        adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float64, device=distance_matrix.device)
        idx = torch.arange(n_nodes, device=distance_matrix.device)
        adj[idx, (idx + 1) % n_nodes] = 1.0
        adj[(idx + 1) % n_nodes, idx] = 1.0
    else:
        adj = initial_adjacency.clone()
        
    adj.requires_grad_(True)
    current_adj = adj
    
    # We can't easily use torch.optim.Optimizer inside the loop if we want to differentiate through the *entire* path 
    # w.r.t alpha, because standard optimizers do in-place updates which might break the graph or be tricky.
    # However, for simple SGD/Adam, we can write the update rule manually to ensure graph connectivity.
    # For SGD: theta_new = theta - lr * grad
    # For Adam: we need to track moments. This is complex to do manually and differentiably w.r.t alpha if alpha affects the gradients.
    # Actually, alpha affects the gradients.
    # If we use SGD, it's fine.
    # If we want Adam for the network, we need to implement functional Adam or use a library like `torchopt` or `higher`.
    # Given the constraints, let's stick to SGD for the network evolution for now, as it's robust and simple for this inner loop.
    # We will just allow changing the learning rate.
    
    # But the user asked for "adam or stochastic gd".
    # If they meant for the ALPHA optimization, we are already using Adam.
    # If they meant for the NETWORK optimization, maybe they want momentum?
    
    # Let's stick to SGD for network evolution but allow momentum if requested (manually implemented).
    momentum = kwargs.get('momentum', 0.0)
    velocity = torch.zeros_like(current_adj) if momentum > 0 else None

    for _ in range(n_iterations):
        payoff = _calculate_payoff_pytorch(
            current_adj, 
            distance_matrix, 
            alpha, 
            beta, 
            distance_fn, 
            optimization_type=optimization_type,
            **kwargs
        )
        
        if optimization_type == "global":
            loss = -payoff
            grads = torch.autograd.grad(loss, current_adj, create_graph=True)[0]
        elif optimization_type == "nodal":
            def functional_payoff(a):
                return _calculate_payoff_pytorch(
                    a, 
                    distance_matrix, 
                    alpha, 
                    beta, 
                    distance_fn, 
                    optimization_type="nodal",
                    **kwargs
                )
            jac = torch.func.jacrev(functional_payoff)(current_adj)
            total_grad = torch.diagonal(jac, dim1=0, dim2=1).T
            grads = -total_grad
            
        if momentum > 0 and velocity is not None:
            velocity = momentum * velocity + grads
            update = velocity
        else:
            update = grads

        current_adj = current_adj - learning_rate * update
        
        if symmetric:
            current_adj = (current_adj + current_adj.T) / 2.0
            
        current_adj = torch.clamp(current_adj, 0, 1)
        current_adj = current_adj * (1 - torch.eye(n_nodes, device=current_adj.device))
        
    return current_adj

def find_optimal_alpha_differentiable(
    distance_matrix: np.ndarray,
    empirical_connectivity: np.ndarray,
    distance_fn: Union[str, Callable],
    n_iterations: int = 2000,
    beta: float = 1.0,
    initial_alpha: float = 1.0,
    learning_rate_alpha: float = 0.1,
    learning_rate_network: float = 0.01,
    alpha_search_iterations: int = 50,
    symmetric: bool = True,
    verbose: bool = True,
    optimization_type: str = "global",
    metric: str = "density",
    optimizer_class: str = "adam",
    stochastic: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Finds optimal alpha using gradient descent on the simulation output.
    
    Args:
        metric: "density" or "mse".
        optimizer_class: "adam" or "sgd".
        stochastic: If True, uses random initial adjacency for each step to avoid local minima.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_nodes = empirical_connectivity.shape[0]
    target_density = np.sum(empirical_connectivity != 0) / (n_nodes * (n_nodes - 1))
    
    dist_matrix_tensor = torch.tensor(distance_matrix, device=device, dtype=torch.float64)
    target_adj_tensor = torch.tensor(empirical_connectivity, device=device, dtype=torch.float64)
    
    # Resolve distance function
    if isinstance(distance_fn, str):
        if "resistance" in distance_fn:
            dist_fn_callable = resistance_distance_pt
        elif "heat" in distance_fn:
            def heat_dist_stable(adj, t=0.5, **kwargs):
                degree = torch.sum(adj, dim=1)
                L = torch.diag(degree) - adj
                kernel = torch.matrix_exp(-t * L)
                return -torch.log(kernel + 1e-10)
            dist_fn_callable = partial(heat_dist_stable, t=kwargs.get('t', 0.5))
        else:
            raise ValueError(f"Unknown distance function: {distance_fn}")
    else:
        dist_fn_callable = distance_fn

    alpha_param = torch.tensor(initial_alpha, device=device, dtype=torch.float64, requires_grad=True)
    beta_tensor = torch.tensor(beta, device=device, dtype=torch.float64)
    
    if optimizer_class.lower() == "adam":
        optimizer_alpha = torch.optim.Adam([alpha_param], lr=learning_rate_alpha)
    elif optimizer_class.lower() == "sgd":
        optimizer_alpha = torch.optim.SGD([alpha_param], lr=learning_rate_alpha)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_class}")
    
    history = []
    
    iterator = tqdm(range(alpha_search_iterations), desc="Optimizing Alpha", disable=not verbose)
    
    for i in iterator:
        optimizer_alpha.zero_grad()
        
        if stochastic:
            # Random initialization
            # Create a random symmetric matrix with values in [0, 1]
            rand_adj = torch.rand((n_nodes, n_nodes), device=device, dtype=torch.float64)
            rand_adj = (rand_adj + rand_adj.T) / 2.0
            rand_adj.fill_diagonal_(0)
            initial_adj = rand_adj
        else:
            initial_adj = None

        final_adj = simulate_network_evolution_differentiable(
            dist_matrix_tensor,
            n_iterations,
            alpha_param,
            beta_tensor,
            dist_fn_callable,
            learning_rate=learning_rate_network,
            symmetric=symmetric,
            optimization_type=optimization_type,
            initial_adjacency=initial_adj,
            **kwargs
        )
        
        current_density = torch.sum(final_adj) / (n_nodes * (n_nodes - 1))
        
        if metric == "density":
            loss = (current_density - target_density) ** 2
        elif metric == "mse":
            loss = torch.nn.functional.mse_loss(final_adj, target_adj_tensor)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        loss.backward()
        optimizer_alpha.step()
        
        # Enforce alpha > 0
        with torch.no_grad():
            alpha_param.clamp_(min=0.001)
            
        if verbose:
            iterator.set_postfix({
                "alpha": f"{alpha_param.item():.4f}", 
                "density": f"{current_density.item():.4f}", 
                "loss": f"{loss.item():.6f}"
            })
            
        history.append({
            'alpha': alpha_param.item(),
            'density': current_density.item(),
            'loss': loss.item()
        })
        
        if loss.item() < 1e-8:
            break
            
    return {
        'alpha': alpha_param.item(),
        'density': history[-1]['density'],
        'history': history,
        'final_adjacency': final_adj.detach().cpu().numpy()
    }

