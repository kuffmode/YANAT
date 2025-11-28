
import numpy as np
import pytest
from yanat.generative_game_theoric_numba import _spectral_normalization

def test_asymmetric_eigenvalues():
    # Asymmetric matrix with complex eigenvalues: [[0, -1], [1, 0]]
    # Eigenvalues are i and -i. Magnitude is 1.
    A = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    
    # This should not raise an error now
    res = _spectral_normalization(A, target_radius=2.0)
    
    # Spectral radius is 1.0. Target is 2.0. Result should be A * 2.0 / 1.0 = 2 * A
    expected = A * 2.0
    np.testing.assert_allclose(res, expected)

def test_symmetric_eigenvalues():
    # Symmetric matrix: [[1, 1], [1, 1]]
    # Eigenvalues are 2 and 0. Spectral radius is 2.
    A = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    
    res = _spectral_normalization(A, target_radius=1.0)
    
    # Expected: A * 1.0 / 2.0 = 0.5 * A
    expected = A * 0.5
    np.testing.assert_allclose(res, expected)
