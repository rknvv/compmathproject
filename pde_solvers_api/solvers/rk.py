from typing import Tuple

import numpy as np

from .utils import laplace
from .base import BaseSolver
from schemas import GrayScottParams


class RungeKuttaSolver(BaseSolver):
    def __init__(
        self, params: GrayScottParams, U: np.ndarray = None, V: np.ndarray = None
    ):
        super().__init__(params, U, V)

    def _compute_derivatives(
        self, U: np.ndarray, V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_U = laplace(U)
        d_V = laplace(V)
        reaction = U * V**2
        dU_dt = self.params.Du * d_U - reaction + self.params.F * (1.0 - U)
        dV_dt = self.params.Dv * d_V + reaction - (self.params.F + self.params.k) * V
        return dU_dt, dV_dt

    def step(self) -> None:
        U = self.U_flat.reshape((self.N, self.N))
        V = self.V_flat.reshape((self.N, self.N))
        dt = self.params.dt

        k1_U, k1_V = self._compute_derivatives(U, V)

        U_temp = U + 0.5 * dt * k1_U
        V_temp = V + 0.5 * dt * k1_V
        k2_U, k2_V = self._compute_derivatives(U_temp, V_temp)

        U_temp = U + 0.5 * dt * k2_U
        V_temp = V + 0.5 * dt * k2_V
        k3_U, k3_V = self._compute_derivatives(U_temp, V_temp)

        U_temp = U + dt * k3_U
        V_temp = V + dt * k3_V
        k4_U, k4_V = self._compute_derivatives(U_temp, V_temp)

        U_new = U + (dt / 6.0) * (k1_U + 2 * k2_U + 2 * k3_U + k4_U)
        V_new = V + (dt / 6.0) * (k1_V + 2 * k2_V + 2 * k3_V + k4_V)

        self.U_flat = U_new.ravel()
        self.V_flat = V_new.ravel()
