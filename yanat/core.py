
from typing import Union, Optional

import numpy as np
from numba import njit
from typeguard import typechecked
from scipy.linalg import expm, solve
import _pickle as pk
from msapy import msa
from msapy.datastructures import ShapleyModeND

from copy import deepcopy

from yanat import utils as ut

@njit
def identity(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    The identity function. It's for the linear case and I literally stole it from Fabrizio:
        https://github.com/fabridamicelli/echoes/blob/master/echoes/utils.py

    Args:
        x (Union[float, np.ndarray]): input. can be a float or an np array.

    Returns:
        Union[float, np.ndarray]: output. will be whatever the input is!
    """
    return x

@njit
def tanh(x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the hyperbolic tangent of the input. Again, I stole this from Fabrizio:
    https://github.com/fabridamicelli/echoes/blob/master/echoes/utils.py

    Args:
        x (Union[float, int, np.ndarray]): input. can be a float or an np array.

    Returns:
        Union[float, np.ndarray]: output, squashed between -1 and 1.
    """
    return np.tanh(x)

@njit
def relu(x: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Computes the relu of the input:
    Args:
        x (Union[float, int, np.ndarray]): input. can be a float or an np array.

    Returns:
        Union[float, np.ndarray]: output, squashed between 0 and 1.
    """
    return np.maximum(0.0, x)

@njit
def simulate_dynamical_system(adjacency_matrix: np.ndarray,
                              input_matrix: np.ndarray,
                              coupling: float = 1,
                              dt: float = 0.001,
                              duration: int = 10,
                              timeconstant: float = 0.01,
                              function: callable = identity,  # type: ignore
                              ) -> np.ndarray:
    """
    Simulates a dynamical system described by the given paramteres.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix (N,N; duh)
        input_matrix (np.ndarray): Input of shape (N, T) where N is the number of nodes and T is the number of time steps.
        coupling (float, optional): The coupling strength between each node (scales the adjacency_matrix). Defaults to 1.
        dt (float, optional): The time step of the simulation. Defaults to 0.001.
        duration (int, optional): The duration of the simulation in seconds. Defaults to 10.
        timeconstant (float, optional): The time constant of the nodes, I think it's the same as the 'relaxation time'. Defaults to 0.01.
        function (Callable, optional): The activation function. Defaults to identity, which means it's linear.

    Returns:
        np.ndarray: The state of the dynamical system at each time step so again, the shape is (N, T)
    """

    N: int = input_matrix.shape[0]
    T: int = int(duration / dt)
    X: np.ndarray = np.zeros((N, T))

    decay_factor: float = dt/timeconstant
    connectivity: np.ndarray = adjacency_matrix*coupling

    for timepoint in range(1, T):

        X[:, timepoint] = ((1-decay_factor) * X[:, timepoint - 1]) + decay_factor * \
            function(connectivity @ X[:, timepoint -
                     1] + input_matrix[:, timepoint - 1])

    # TODO: Add examples and tests
    return X


@typechecked
def sar(adjacency_matrix: np.ndarray, alpha:float = 0.5, normalize: bool = False) -> np.ndarray:
    """
   Computes the analytical covariance matrix for the spatial autoregressive (SAR) model. 

    The SAR model considers each region's activity as a weighted sum of fluctuations from
    other regions, adjusted by a spatial influence factor 'alpha', plus a unit-variance Gaussian noise.
    The covariance matrix is analytically derived as the inverse of (I - alpha * A) times its transpose.
    See [1]. 

    Args:
        adjacency_matrix (np.ndarray): A square, nonnegative, and symmetric matrix representing network's structure.
        alpha (float): The spatial influence factor, should be less than the spectral radius of 'adjacency_matrix'.
        The smaller the decay rate, the quicker it assumes the walks to be subsiding.
        normalize (bool, optional): If True, normalizes the matrix using strength normalization described by [2]. Defaults to False.

    Returns:
        np.ndarray: The covariance matrix of the SAR model. Shape: (N, N)
    
    References:
    [1] https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003530
    [2] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484
    
    # TODO: Add examples and tests
    """
    if normalize: # `ut.strength_normalization` is a defined utility function, don't worry, I got you covered love.
        adjacency_matrix: np.ndarray = ut.strength_normalization(adjacency_matrix)
        
    N:int = adjacency_matrix.shape[0]
    I:np.ndarray = np.eye(N)
    # Check if the user has given a shit about the documentation.
    if adjacency_matrix.shape[1] != N:
        raise ValueError("The adjacency matrix must be square. Like Spongebob's pants.")
    
    if np.any(adjacency_matrix < 0.):
        raise ValueError("The adjacency matrix contains negative values. Unless we figure out some stuff, only nonnegative matrices are allowed.")

    # Do some magic.
    inverse_matrix:np.ndarray = solve(I - alpha * adjacency_matrix, I, assume_a='sym')
    influence_matrix:np.ndarray = inverse_matrix @ inverse_matrix.T
    return influence_matrix


@typechecked
def lam(adjacency_matrix: np.ndarray, alpha:float = 0.5, normalize: bool = False) -> np.ndarray:
    """
    Computes the influence matrix for the Linear Attenuation Model (LAM), which underpins
    the dynamics of Katz centrality and is similar to communicability, but with a linear
    discount on longer walks rather than an exponential one. The discount factor 'alpha'
    should be less than the spectral radius of the adjacency matrix. See [1] for more details.

    Note from Gorka Zamora-lopez on directed graphs: For the adjacency matrix 'A', A_{ij} = 1 indicates j --> i,
    which is the opposite of the conventional graph theory notation. If your adjacency
    matrices follow the graph theory convention, ensure to transpose it first.

    Args:
        adjacency_matrix (np.ndarray): A square and nonnegative matrix representing network's structure.
        alpha (float): the decay rate. The smaller the decay rate, the quicker it assumes the walks to be subsiding.
        normalize (bool, optional): If True, applies strength normalization [2] to the matrix. Defaults to False.

    Returns:
        np.ndarray: The influence matrix for LAM. Shape: (N, N)
        
    References:
    [1] https://arxiv.org/abs/2307.02449
    [2] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484
    
    # TODO: Add examples and tests
    """
    if normalize: # `ut.strength_normalization` is a defined utility function, don't worry suger, I got you covered.
        adjacency_matrix: np.ndarray = ut.strength_normalization(adjacency_matrix)
    
    N:int = adjacency_matrix.shape[0]
    I:np.ndarray = np.eye(N)

    # Check if the user has given a shit about the documentation.
    if adjacency_matrix.shape[1] != N:
        raise ValueError("The adjacency matrix must be square. Like Spongebob's pants.")
    
    if np.any(adjacency_matrix < 0.):
        raise ValueError("The adjacency matrix contains negative values. Unless we figure out some stuff, only nonnegative matrices are allowed.")
    
    # Do some skibidi bapbap.
    influence_matrix = solve(I - alpha * adjacency_matrix, I)
    return influence_matrix


@typechecked
def communicability(adjacency_matrix: np.ndarray, alpha:float = 1, normalize: bool = False) -> np.ndarray:
    """
    Computes the communicability of the network, with the option to be scaled by a decay rate factor 'alpha'.
    The alpha factor modulates the decay rate of walks, with smaller values leading
    to quicker subsidence. Alpha should be in the range (0, spectral radius of A).
    Works for binary, weighted, and directed graphs. See [1] for more details.

    Note from Gorka Zamora-lopez on directed graphs: For the adjacency matrix 'A', A_{ij} = 1 indicates j --> i,
    which is the opposite of the conventional graph theory notation. If your adjacency
    matrices follow the graph theory convention, ensure to transpose it first.


    Args:
        adjacency_matrix (np.ndarray): A square and nonnegative matrix representing network's structure.
        alpha (float, optional): The scaling factor. Defaults to 1, meaning that no scaling is applied.
        normalize (bool, optional): If True, applies strength normalization to the matrix [2]. Defaults to False.

    Returns:
        np.ndarray: The (scaled) communicability matrix. Shape: (N, N)
        
    References:
    [1] https://arxiv.org/abs/2307.02449
    [2] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484
    """   
    
    if normalize:
        adjacency_matrix: np.ndarray = ut.strength_normalization(adjacency_matrix)
    
    # Check if the user has given a shit about the documentation.
    if adjacency_matrix.shape[1] != adjacency_matrix.shape[0]:
        raise ValueError("The adjacency matrix must be square. Like Spongebob's pants.")
    
    if np.any(adjacency_matrix < 0.):
        raise ValueError("The adjacency matrix contains negative values. Unless we figure out some stuff, only nonnegative matrices are allowed.")    

    adjacency_matrix *= alpha
    communicability_matrix: np.ndarray = expm(adjacency_matrix)
    return communicability_matrix


@typechecked
def default_game(complements: tuple, 
                       adjacency_matrix: Union[np.ndarray,str], 
                       index: int, 
                       input_noise: np.ndarray,
                       model: callable = simulate_dynamical_system, 
                       model_params:Optional[dict]=None) -> np.ndarray:
    """
    Lesions the given nodes and simulates the dynamics of the system afterwards. Lesioning here means setting the incoming and outgoing
    connections of the node to zero

    Args:
    complements (tuple): Indices of nodes to be lesioned. Comes from MSA, don't worry about it suger.
    adjacency_matrix (Union[np.ndarray, str]): The adjacency matrix representing the system
            or a path to a pickle file containing the adjacency matrix. The pickling thing makes it faster so I recommend that.
    index (int): Index of the target node whose activity is to be returned. Also comes from MSA.
    input_noise (np.ndarray): Input noise/signal for the dynamical model. Shape (N,T).
    model (callable, optional): The dynamical system model function to simulate with. Defaults to a linear dynamical system.
    model_params (dict, optional): Additional keyword arguments to pass to the model function.

    Returns:
        np.ndarray: Resulted activity of the target node given the lesion. Shape is (T,)
    """
    model_params = model_params if model_params else {}

    if isinstance(adjacency_matrix, str):
        with open(adjacency_matrix, 'rb') as f:
            lesioned_connectivity = pk.load(f)
            
    elif isinstance(adjacency_matrix, np.ndarray):
        lesioned_connectivity = deepcopy(adjacency_matrix)
    else:
        raise ValueError("The adjacency matrix must be either a numpy array or a path to a pickle file containing the matrix.")
    
    lesioned_connectivity[:, complements] = 0.0
    lesioned_connectivity[complements, :] = 0.0

    dynamics = model(adjacency_matrix = lesioned_connectivity, input_matrix=input_noise, **model_params)
    lesioned_signal = dynamics[index]
    return lesioned_signal


@typechecked
def optimal_influence(n_elements:int, game:callable = default_game, game_kwargs:Optional[dict]=None, msa_kwargs:Optional[dict]=None) -> ShapleyModeND:
    
    game_kwargs = game_kwargs if game_kwargs else {}
    msa_kwargs = msa_kwargs if msa_kwargs else {}
    
    oi = msa.estimate_causal_influences(
    elements=list(range(n_elements)),
    objective_function=game,
    objective_function_params=game_kwargs,
    **msa_kwargs)
    
    return oi