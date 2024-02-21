import pytest
import numpy as np
from yanat.utils import identity, tanh, sar, lam


# The function returns the input value if it's a float or an np array.
def test_returns_input_value():
    # Arrange
    input_value = 5.0

    # Act
    result = identity(input_value)

    # Assert
    assert result == input_value


# The function returns the correct output for a np array input with a single element.
def test_returns_correct_output_for_single_element_array():
    # Arrange
    input_array = np.array([3.0])

    # Act
    result = identity(input_array)

    # Assert
    assert np.array_equal(result, input_array)



# Returns the hyperbolic tangent of a positive float or integer.
def test_positive_input():
    # Arrange
    x = 2.5

    # Act
    result = tanh(x)

    # Assert
    assert np.isclose(result, np.tanh(x))
    
    
# Computes the covariance matrix for a valid adjacency matrix with default parameters.
def test_valid_adjacency_matrix_default_parameters():
    # Arrange
    adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    # Act
    result = sar(adjacency_matrix)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)

# Raises ValueError if the adjacency matrix is not square.
def test_non_square_adjacency_matrix():
    # Arrange
    adjacency_matrix = np.array([[0, 1, 0], [1, 0, 1]])

    # Act and Assert
    with pytest.raises(ValueError):
        sar(adjacency_matrix)