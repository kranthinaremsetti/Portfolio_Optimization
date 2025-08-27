from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import io

app = FastAPI()

class OptimizationRequest(BaseModel):
    dataset_option: str  # 'NASDAQ', 'NIFTY50', 'Crypto'
    budget: int          # number of assets to select
    risk_factor: str   # gamma value
    total_investment: float  # total investment amount

@app.post("/optimize")
def optimize_portfolio(req: OptimizationRequest):
    # Load data
    expected_returns_path = f"results/{req.dataset_option}/expected_returns.csv"
    cov_matrix_path = f"results/{req.dataset_option}/cov_matrix.csv"
    expected_returns = pd.read_csv(expected_returns_path, index_col=0)
    cov_matrix = pd.read_csv(cov_matrix_path, index_col=0)

    # Calculate Sharpe ratio
    asset_volatility = np.sqrt(np.diag(cov_matrix))
    sharpe_ratio = expected_returns.values.flatten() / asset_volatility

    # Select top assets
    N = max(req.budget, 1)
    sorted_indices = np.argsort(sharpe_ratio)[::-1][:max(N, req.budget)]
    top_assets = expected_returns.index[sorted_indices]

    mu = expected_returns.loc[top_assets].values.flatten()
    Sigma = cov_matrix.loc[top_assets, top_assets].values
    n_assets = len(top_assets)

    # Budget constraint
    budget = min(req.budget, n_assets)

    gamma_map = {
    'low': 0.1,
    'medium': 0.5,
    'high': 1.0
    }
    risk_factor = gamma_map.get(req.risk_factor)
    if risk_factor is None:
        return {"error": "Invalid risk_factor. Choose from 'low', 'medium', 'high'."}
    # QAOA setup
    qp = QuadraticProgram("Portfolio Optimization")
    for i in range(n_assets):
        qp.binary_var(name=f"x_{i}")
    linear = -mu
    quadratic = risk_factor * Sigma
    qp.minimize(linear=linear, quadratic=quadratic)
    qp.linear_constraint(
        linear={f"x_{i}": 1 for i in range(n_assets)},
        sense="==",
        rhs=budget,
        name="budget_constraint"
    )

    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    # Get selected assets
    selected_indices = [i for i, val in enumerate(result.x) if val > 0.5]
    chosen_returns = expected_returns.iloc[selected_indices].values.flatten()
    chosen_assets = top_assets[selected_indices]
    weights = chosen_returns / chosen_returns.sum() if chosen_returns.sum() > 0 else np.zeros_like(chosen_returns)
    allocation = weights * req.total_investment

    portfolio_allocation = pd.DataFrame({
        "Asset": chosen_assets,
        "Expected Return": chosen_returns,
        "Weight": weights,
        "Investment": allocation
    })

    # Convert to CSV
    csv_buffer = io.StringIO()
    portfolio_allocation.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(csv_buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=portfolio_allocation.csv"})
