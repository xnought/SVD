import numpy as np

__all__ = ["svd"]


def star(A: np.ndarray):
    """conjugate transpose
    For the notation A^* (A' in matlab)
    """
    return A.conj().T


def eigenv(A: np.ndarray):
    """computes the eigen values (not multiples) and eigen vectors of A"""
    return np.linalg.eig(A)


def svd(A: np.ndarray):
    (m, n) = A.shape

    #  A^* @ A
    if m > n:
        S_2, V = eigenv(star(A) @ A)
        S = np.sqrt(S_2)
        U = A @ V / S

    #  A @ A^*
    else:
        S_2, U = eigenv(A @ star(A))
        S = np.sqrt(S_2)
        V = star(A) @ U / S

    return S, U, V


def testing():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    S, U, V = svd(A)

    print(f"{U = }")
    print(f"{S = }")
    print(f"{V = }")
    print(f"{U @ np.diag(S) @ star(V) = }")


if __name__ == "__main__":
    testing()
