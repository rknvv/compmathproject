from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

from schemas import GrayScottParams
from .utils import initialize_state


class BaseSolver(ABC):
    def __init__(
        self,
        params: GrayScottParams,
        U: np.ndarray | None = None,
        V: np.ndarray | None = None,
    ):
        self.params = params
        self.N = params.grid_size
        if U is None or V is None:
            U_init, V_init = initialize_state(self.N)
            self.U_flat = U_init.ravel()
            self.V_flat = V_init.ravel()
        else:
            self.U_flat = np.asarray(U).ravel()
            self.V_flat = np.asarray(V).ravel()

    @abstractmethod
    def step(self) -> None:
        pass

    def run_steps(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(n):
            self.step()
        return self.get_state()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.U_flat.reshape((self.N, self.N)), self.V_flat.reshape(
            (self.N, self.N)
        )
