from typing import Dict, Type

import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, status

from solvers.base import BaseSolver
from solvers.utils import initialize_state
from solvers import RungeKuttaSolver, CrankNicolsonSolver
from schemas import GrayScottParams, SimulationStepInput, SimulationStateOutput


SOLVERS: Dict[str, Type[BaseSolver]] = {
    "crank_nicolson": CrankNicolsonSolver,
    "runge_kutta": RungeKuttaSolver,
}


def get_solver(method: str) -> Type[BaseSolver]:
    solver_class = SOLVERS.get(method.lower())
    if solver_class is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    return solver_class


app = FastAPI()
router = APIRouter(prefix="/gray-scott", tags=["Gray-Scott"])


@router.post("/initialize", response_model=SimulationStateOutput)
async def initialize_simulation(params: GrayScottParams):
    U, V = initialize_state(params.grid_size)
    return SimulationStateOutput(U=U.tolist(), V=V.tolist())


@router.post("/step/{method}", response_model=SimulationStateOutput)
async def run_step(method: str, input_data: SimulationStepInput):
    SolverClass = get_solver(method)
    U = np.array(input_data.U)
    V = np.array(input_data.V)

    solver = SolverClass(params=input_data.params, U=U, V=V)
    solver.step()
    new_U, new_V = solver.get_state()
    return SimulationStateOutput(U=new_U.tolist(), V=new_V.tolist())


@router.post("/solve/{method}", response_model=SimulationStateOutput)
async def solve_endpoint(
    method: str,
    params: GrayScottParams,
    steps: int = 100,
):
    SolverClass = get_solver(method)

    solver = SolverClass(params=params)
    final_U, final_V = solver.run_steps(steps)
    return SimulationStateOutput(U=final_U.tolist(), V=final_V.tolist())


app.include_router(router)
