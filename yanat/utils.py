from __future__ import annotations
import numpy as np
from typing import Generator, Optional, Union
import pandas as pd
import _pickle as pk
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity


def find_density(adjacency_matrix: np.ndarray) -> float:
    """Finds the density of the given adjacency matrix. It's the ratio of the number of edges to the number of possible edges.

    Args:
        adjacency_matrix (np.ndarray): The adjacency matrix of the network.

    Returns:
        float: The density of the network.
    """
    return np.where(adjacency_matrix != 0, 1, 0).sum() / adjacency_matrix.shape[0] ** 2


def minmax_normalize(
    data: Union[pd.DataFrame, np.ndarray],
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
    np.fill_diagonal(lognorm_adjacency_matrix, 0.0)
    return np.where(lognorm_adjacency_matrix != 1.0, lognorm_adjacency_matrix, 0.0)


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
    """
    Normalizes the adjacency matrix to subside the effect of high-strength (or high-degree) nodes.
    This function implements the strength normalization algorithm described in [1].
    The algorithm aims to reduce the influence of high-strength nodes in a network by scaling the adjacency matrix.

    Parameters:
        adjacency_matrix (np.ndarray): The adjacency matrix of the network. It should be a square matrix of shape (n, n), where n is the number of nodes in the network.

    Returns:
        np.ndarray: The normalized adjacency matrix with the same shape as the input.

    References:
        [1] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484
    """
    strength: np.ndarray = adjacency_matrix.sum(1)
    normalized_strength: np.ndarray = np.power(strength, -0.5)
    diagonalized_normalized_strength: np.ndarray = np.diag(normalized_strength)
    normalized_adjacency_matrix: np.ndarray = (
        diagonalized_normalized_strength
        @ adjacency_matrix
        @ diagonalized_normalized_strength
    )
    return normalized_adjacency_matrix


def optimal_influence_default_values(
    adjacency_matrix: np.ndarray,
    location: str = "adjacency_matrices_for_oi",
    random_seed: int = 11,
) -> dict:
    """
    Returns the default values for the parameters of the optimal_influence function.

    Parameters:
        adjacency_matrix (np.ndarray): The adjacency matrix representing the network structure.
        location (str, optional): The location to save the adjacency matrix file. Defaults to "adjacency_matrices_for_oi".
        random_seed (int, optional): The random seed for generating input noise. Defaults to 11.

    Returns:
        dict: Default values for the parameters of the optimal_influence function.
    """
    rng: Generator = np.random.default_rng(seed=random_seed)
    NOISE_STRENGTH: float = 1
    DELTA: float = 0.01
    TAU: float = 0.02
    G: float = 0.5
    DURATION: int = 10
    input_noise: np.ndarray = rng.normal(
        0, NOISE_STRENGTH, (adjacency_matrix.shape[0], int(DURATION / DELTA))
    )
    model_params: dict = {
        "dt": DELTA,
        "timeconstant": TAU,
        "coupling": G,
        "duration": DURATION,
    }
    timestamp: str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    base_path: Path = Path(location)
    base_path.mkdir(parents=True, exist_ok=True)
    file_location: str = (
        base_path / f"adjmat_{adjacency_matrix.shape[0]}_nodes_{timestamp}"
    )

    with open(f"{file_location}.pkl", "wb") as f:
        pk.dump(adjacency_matrix, f)

    game_params: dict = {
        "adjacency_matrix": f"{file_location}.pkl",
        "input_noise": input_noise,
        "model_params": model_params,
    }

    # TODO: allow already pickled adjacency matrices to be used as input.
    # TODO: allow the user to specify an arbitrary parameter while keeping the rest as default.
    return game_params


def simple_fit(
    parameter_space: list[ParameterGrid],
    target_matrix: np.ndarray,
    model: callable,
    model_kwargs: Optional[dict] = None,
    normalize: Union[bool, callable] = False,
) -> ParameterGrid:
    """Simple (and honestly, ugly) fitting function to find the best parameters for a (communication) model.
    Does a normal for-loop so it's not as efficient but at the moment, doesn't need to be either!

    Args:
        parameter_space (list): Parameter space to search in, as of now, it should be a list[ParameterGrid] and I gotta fix it!
        target_matrix (np.ndarray): Which matrix to compare the model's output to, usually FC or OI.
        model (callable): Which (communication) model to use.
        model_kwargs (Optional[dict], optional): Extra things that the model wants. Defaults to None.
        normalize (Union[bool, callable], optional): If the output needs to be normalized before taking correlation. Defaults to False.

    Returns:
        list: Updated copy of the parameter space with the correlation values.
    """

    model_kwargs: dict = model_kwargs if model_kwargs else {}
    results = deepcopy(parameter_space)
    for parameter in tqdm(results, total=len(parameter_space), desc="C3PO noises: "):
        estimation: np.ndarray = model(**parameter, **model_kwargs)

        if normalize:
            estimation: np.ndarray = normalize(estimation)

        r: float = _matrix_correlation(target_matrix, estimation)
        parameter.update({"correlation": r})

    # TODO: This should be parallelized and sklearn compatible.
    return results


def _matrix_correlation(one_matrix: np.ndarray, another_matrix: np.ndarray) -> float:
    """Computes the Pearson's correlation between two matrices (not just the upper-triangle).

    Args:
        one_matrix (np.ndarray): One of the matrices.
        another_matrix (np.ndarray): Guess what, the other matrix.

    Returns:
        float: Pearson's correlation between the two matrices.
    """
    return np.corrcoef(one_matrix.flatten(), another_matrix.flatten())[0, 1]

def calculate_endpoint_similarity(synthetic_matrix, empirical_matrix):
    similarities = np.zeros(synthetic_matrix.shape[0])
    for i in range(synthetic_matrix.shape[0]):
        similarities[i] = cosine_similarity(synthetic_matrix[i].reshape(1, -1),
                                            empirical_matrix[i].reshape(1, -1))
    return similarities
# TODO: add a function to create example adjacency matrices for demonstration purposes.
