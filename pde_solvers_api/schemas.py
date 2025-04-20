from pydantic import BaseModel
from typing import List


class GrayScottParams(BaseModel):
    Du: float
    Dv: float
    F: float
    k: float
    grid_size: int
    dt: float


class SimulationStepInput(BaseModel):
    params: GrayScottParams
    U: List[List[float]]
    V: List[List[float]]


class SimulationStateOutput(BaseModel):
    U: List[List[float]]
    V: List[List[float]]
