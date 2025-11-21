# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp
from scipy.sparse import issparse


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None
    
    if A.ndim !=2:
        return None
    
    if A.shape[0] != A.shape[1]:
        return None

    if issparse(A):
        A = A.toarray()

    diag_elements = np.abs(np.diagonal(A))

    row_sums = np.sum(np.abs(A), axis=1) - diag_elements

    return np.all(diag_elements > row_sums)



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    if not (isinstance(A, np.ndarray) and isinstance(x, np.ndarray) and isinstance(b, np.ndarray)):
        return None
    
    if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        return None

    m, n = A.shape

    if n != x.shape[0]:
        return None
    if m != b.shape[0]:
        return None
    
    return float(np.linalg.norm(b - (A @ x)))
