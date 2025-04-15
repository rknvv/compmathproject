from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
from schemas import HeatEquationInput
from solvers.heat_equation import solve_heat_equation
import plotly.graph_objects as go
import base64

app = FastAPI(title="PDE Solver API")

@app.post("/solve/heat-equation/")
async def solve_heat_eq(params: HeatEquationInput): # asynchronous call to computing server
    # Convert input to NumPy
    u0 = np.array(params.initial_condition)
    
    # Solve PDE
    solution = solve_heat_equation(
        alpha=params.alpha,
        length=params.length,
        nx=params.nx,
        nt=params.nt,
        dt=params.dt,
        initial_condition=u0,
        bc=params.boundary_conditions
    )

    # Generate Plotly figure
    fig = go.Figure(data=[go.Surface(z=solution.T)])
    fig.update_layout(title="Heat Equation Solution")
    plot_html = fig.to_html(full_html=False)

    # Return solution + plot
    return {
        "solution": solution.tolist(),  # Convert to JSON-serializable
        "visualization": plot_html     # Embeddable Plotly HTML
    }