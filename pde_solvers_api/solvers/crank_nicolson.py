from typing import Tuple

import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

from .utils import laplace
from .base import BaseSolver
from schemas import GrayScottParams


class CrankNicolsonSolver(BaseSolver):
    def __init__(
        self, params: GrayScottParams, U: np.ndarray = None, V: np.ndarray = None
    ):
        super().__init__(params, U, V)
        self.alpha_u = (params.dt / 2) * params.Du
        self.alpha_v = (params.dt / 2) * params.Dv

    def _compute_reaction_terms(
        self, U: np.ndarray, V: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        R_U = -U * V**2 + self.params.F * (1 - U)
        R_V = U * V**2 - (self.params.F + self.params.k) * V
        return R_U, R_V

    def _compute_rhs(
        self, U_flat: np.ndarray, V_flat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        U_2d = U_flat.reshape((self.N, self.N))
        V_2d = V_flat.reshape((self.N, self.N))
        R_U, R_V = self._compute_reaction_terms(U_2d, V_2d)
        R_U_flat = R_U.ravel()
        R_V_flat = R_V.ravel()
        U_temp = (
            U_flat + self.alpha_u * laplace(U_2d).ravel() + self.params.dt * R_U_flat
        )
        V_temp = (
            V_flat + self.alpha_v * laplace(V_2d).ravel() + self.params.dt * R_V_flat
        )
        return U_temp, V_temp

    def _A_times_v(self, v: np.ndarray, alpha: float) -> np.ndarray:
        v_2d = v.reshape((self.N, self.N))
        L_v = laplace(v_2d).ravel()
        return v - alpha * L_v

    def step(self) -> None:
        U_temp, V_temp = self._compute_rhs(self.U_flat, self.V_flat)

        A_u = LinearOperator(
            dtype=np.float64,
            shape=(self.N**2, self.N**2),
            matvec=lambda v: self._A_times_v(v, self.alpha_u),
        )
        self.U_flat, _ = cg(
            A_u,
            U_temp,
            x0=self.U_flat,
            atol=1e-10,
        )

        A_v = LinearOperator(
            dtype=np.float64,
            shape=(self.N**2, self.N**2),
            matvec=lambda v: self._A_times_v(v, self.alpha_v),
        )
        self.V_flat, _ = cg(
            A_v,
            V_temp,
            x0=self.V_flat,
            atol=1e-10,
        )
