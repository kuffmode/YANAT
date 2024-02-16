import numpy as np
from typing import Generator, Union
import pandas as pd
import _pickle as pk
from datetime import datetime 
from pathlib import Path

def find_density(adjacency_matrix: np.ndarray) -> float:
    """Finds the density of the given adjacency matrix. It's the ratio of the number of edges to the number of possible edges.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the network.

    Returns:
        float: The density of the network.
    """    
    return np.where(adjacency_matrix != 0, 1, 0).sum() / adjacency_matrix.shape[0] ** 2


def minmax_normalize(
    data: Union[pd.DataFrame, np.ndarray]
) -> Union[pd.DataFrame, np.ndarray]:
    """Normalizes data between 0 and 1.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): Data to be normalized. Can be a DataFrame or an np array but in both cases it should be at most 2D.

    Returns:
        Union[pd.DataFrame, np.ndarray]: Normalized data with the same shape as the input.
    """    
    return (data - data.min()) / (data.max() - data.min())


def log_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    """Returns the logarithm of the data (adjacency_matrix) but also takes care of the infinit values.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network. Technically can be any matrix but I did it for the adjacency matrices.

    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """    
    return np.nan_to_num(np.log(adjacency_matrix), neginf=0, posinf=0)


def log_minmax_normalize(adjacency_matrix: np.ndarray) -> np.ndarray:
    """It first takes the logarithm of the data and then normalizes it between 0 and 1. It also takes care of the infinit values and those nasty things.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network. Technically can be any matrix but I did it for the adjacency matrices.

    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """    
    lognorm_adjacency_matrix = minmax_normalize(log_normalize(adjacency_matrix))
    np.fill_diagonal(lognorm_adjacency_matrix,0.)
    return np.where(lognorm_adjacency_matrix!=1.,lognorm_adjacency_matrix,0.)


def spectral_normalization(
    target_radius: float, adjacency_matrix: np.ndarray
) -> np.ndarray:
    """Normalizes the adjacency matrix to have a spectral radius of the target_radius. Good to keep the system stable.

    Args:
        target_radius (float): A value below 1.0. It's the spectral radius that you want to achieve. But use 1.0 if you're planning to change the global coupling strength somewhere.
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.

    Returns:
        np.ndarray: Normalized adjacency matrix with the same shape as the input.
    """    
    spectral_radius = np.max(np.abs(np.linalg.eigvals(adjacency_matrix)))
    return adjacency_matrix * target_radius / spectral_radius


def strength_normalization(adjacency_matrix: np.ndarray) -> np.ndarray:
    """Normalizes the adjacency matrix to subside the effect of high-strength nodes.
    See https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484 for more details.

    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix of the network.

    Returns:
        np.ndarray: Normalized adjacency matrix with the same shape as the input.
    """
    strength: np.ndarray = adjacency_matrix.sum(1)
    normalized_strength: np.ndarray = np.power(strength, -0.5)
    diagonalized_normalized_strength: np.ndarray = np.diag(normalized_strength)
    normalized_adjacency_matrix: np.ndarray = diagonalized_normalized_strength @ adjacency_matrix @ diagonalized_normalized_strength
    return normalized_adjacency_matrix


def optimal_influence_default_values(adjacency_matrix:np.ndarray, location:str = "adjacency_matrices_for_oi", random_seed:int = 11) -> dict:
    """Returns the default values for the parameters of the optimal_influence function.

    Returns:
        dict: Default values for the parameters of the optimal_influence function.
    """
    rng:Generator = np.random.default_rng(seed=random_seed)
    NOISE_STRENGTH:float = 1
    DELTA:float = 0.01
    TAU:float = 0.02
    G:float = 0.5
    DURATION:int = 10
    input_noise:np.ndarray = rng.normal(0, NOISE_STRENGTH, (adjacency_matrix.shape[0], int(DURATION / DELTA)))
    model_params:dict = {"dt": DELTA, "timeconstant": TAU, "coupling": G, "duration": DURATION}
    timestamp:str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

      
    base_path:Path = Path(location)
    base_path.mkdir(parents=True, exist_ok=True)
    file_location:str = base_path / f"adjmat_{adjacency_matrix.shape[0]}_nodes_{timestamp}"  

    with open(f"{file_location}.pkl", 'wb') as f:
        pk.dump(adjacency_matrix, f)
    
    game_params:dict = {"adjacency_matrix": f"{file_location}.pkl", "input_noise": input_noise, "model_params": model_params}

    return game_params