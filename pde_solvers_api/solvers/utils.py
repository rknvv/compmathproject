from typing import Tuple

import numpy as np


def initialize_state(grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    U = np.ones((grid_size, grid_size)) + 0.02 * np.random.random(
        (grid_size, grid_size)
    )
    V = np.zeros((grid_size, grid_size)) + 0.02 * np.random.random(
        (grid_size, grid_size)
    )

    S = max(1, grid_size // 4)
    r = S // 2
    N2 = grid_size // 2
    low = max(0, N2 - r)
    high = min(grid_size, low + S)

    U[low:high, low:high] = 0.5
    V[low:high, low:high] = 0.25

    return (U, V)


def laplace(f):
    """Метод конечных разностей по пространству."""
    up = np.roll(f, shift=-1, axis=0)
    down = np.roll(f, shift=1, axis=0)
    left = np.roll(f, shift=-1, axis=1)
    right = np.roll(f, shift=1, axis=1)
    return up + down + left + right - 4 * f
