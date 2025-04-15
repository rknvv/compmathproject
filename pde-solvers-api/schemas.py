from pydantic import BaseModel
from typing import List

class HeatEquationInput(BaseModel):
    alpha: float           # Thermal diffusivity
    length: float          # Domain length (e.g., 1 meter)
    nx: int                # Spatial grid points (e.g., 100)
    nt: int                # Time steps (e.g., 500)
    dt: float              # Time step size
    initial_condition: List[float]  # u(x,0) as array
    boundary_conditions: dict       # E.g., {"left": 0.0, "right": 0.0}