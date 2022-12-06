import numpy as np

__all__ = ["svd"]


def star(A: np.ndarray):
    return A.conj().T


def svd(A: np.ndarray):
    (m, n) = A.shape

    #  A' @ A
    if m > n:
        S_2, V = np.linalg.eig(star(A) @ A)
        S = np.sqrt(S_2)
        U = A @ V / S

    #  A @ A'
    else:
        S_2, U = np.linalg.eig(A @ star(A))
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
