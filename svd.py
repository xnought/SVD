import numpy as np

__all__ = ["svd"]


def svd(A: np.ndarray):
    S = []
    U = []
    V = []
    return S, U, V


def testing():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    S, U, V = svd(A)

    print(f"{S = }")
    print(f"{U = }")
    print(f"{V = }")


if __name__ == "__main__":
    testing()
