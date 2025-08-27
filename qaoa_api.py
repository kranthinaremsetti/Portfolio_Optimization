from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import io
import os

app = FastAPI()

class OptimizationRequest(BaseModel):
    dataset_option: str  # 'NASDAQ', 'NIFTY50', 'Crypto'
    budget: int          # number of assets to select
    risk_factor: str   # gamma value
    total_investment: float  # total investment amount


class RebalancingRequest(BaseModel):
    dataset_option: str
    future_dataset_option: str | None = None  # defaults to dataset_option + "_FUTURE" if None
    budget: int
    risk_factor: str
    total_investment: float


def _normalize_dataset_option(name: str) -> str:
    """Normalize user-provided dataset string to match results folder names.

    Maps synonyms (e.g., 'NASDAQ100' -> 'NASDAQ', 'CRYPTO50' -> 'Crypto') and
    future variants ('_FUTURE'/'_Future'/' future' indications) to '*_Future'.
    """
    raw = (name or "").strip()
    is_future = False
    lower = raw.lower()
    # detect future suffix or word
    if lower.endswith("_future") or lower.endswith(" future") or lower.endswith("-future"):
        is_future = True
        lower = lower.replace("_future", "").replace(" future", "").replace("-future", "")

    # base dataset mapping
    base_map = {
        "nasdaq": "NASDAQ",
        "nasdaq100": "NASDAQ",
        "nifty50": "NIFTY50",
        "crypto": "Crypto",
        "crypto50": "Crypto",
    }
    base = base_map.get(lower, None)
    if base is None:
        # If exact folder exists under results, use as-is
        candidate = raw
        results_dir = os.path.join("results", candidate)
        if os.path.isdir(results_dir):
            return candidate
        # Try title-casing common forms
        if lower in base_map:
            base = base_map[lower]
        else:
            # default pass-through
            base = raw

    return f"{base}_Future" if is_future else base


def _get_results_paths(dataset_option: str):
    """Return (expected_returns_path, cov_matrix_path, error_message_or_None)."""
    folder = _normalize_dataset_option(dataset_option)
    results_dir = os.path.join("results", folder)
    er = os.path.join(results_dir, "expected_returns.csv")
    cm = os.path.join(results_dir, "cov_matrix.csv")
    if not os.path.exists(er) or not os.path.exists(cm):
        return None, None, f"Missing results for '{folder}'. Expected files: {er}, {cm}"
    return er, cm, None

@app.post("/optimize")
def optimize_portfolio(req: OptimizationRequest):
    # Load data (with normalization to match results folders)
    expected_returns_path, cov_matrix_path, err = _get_results_paths(req.dataset_option)
    if err:
        return {"error": err}
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
    chosen_returns = expected_returns.loc[top_assets].iloc[selected_indices].values.flatten()
    chosen_assets = top_assets[selected_indices]
    weights = chosen_returns / chosen_returns.sum() if chosen_returns.sum() > 0 else np.zeros_like(chosen_returns)
    allocation = weights * req.total_investment
    percentage = weights * 100.0

    portfolio = []
    for i in range(len(chosen_assets)):
        portfolio.append({
            "asset": str(chosen_assets[i]),
            "expected_return": float(chosen_returns[i]),
            "weight": float(weights[i]),
            "investment": float(allocation[i]),
            "percentage": float(percentage[i]),
        })

    response = {
        "dataset": _normalize_dataset_option(req.dataset_option),
        "budget": int(budget),
        "risk_factor": req.risk_factor,
        "gamma": float(risk_factor),
        "total_investment": float(req.total_investment),
        "objective_value": float(result.fval) if hasattr(result, 'fval') else None,
        "portfolio": portfolio,
    }
    return JSONResponse(content=response)


@app.post("/rebalance")
def rebalance_portfolio(req: RebalancingRequest):
    # Resolve dataset names
    current_opt = _normalize_dataset_option(req.dataset_option)
    future_opt = _normalize_dataset_option(req.future_dataset_option) if req.future_dataset_option else _normalize_dataset_option(f"{current_opt}_Future")

    def solve(dataset_option: str):
        # Load data with normalization
        expected_returns_path, cov_matrix_path, errp = _get_results_paths(dataset_option)
        if errp:
            return None, errp

        expected_returns = pd.read_csv(expected_returns_path, index_col=0)
        cov_matrix = pd.read_csv(cov_matrix_path, index_col=0)

        # Sharpe ratio
        asset_volatility = np.sqrt(np.diag(cov_matrix))
        sharpe_ratio = expected_returns.values.flatten() / asset_volatility

        # Select top assets (consistent with /optimize)
        N = max(req.budget, 1)
        sorted_indices = np.argsort(sharpe_ratio)[::-1][:max(N, req.budget)]
        top_assets = expected_returns.index[sorted_indices]

        mu = expected_returns.loc[top_assets].values.flatten()
        Sigma = cov_matrix.loc[top_assets, top_assets].values
        n_assets = len(top_assets)

        # Budget constraint
        budget_eff = min(req.budget, n_assets)

        gamma_map = {
            'low': 0.1,
            'medium': 0.5,
            'high': 1.0
        }
        gamma = gamma_map.get(req.risk_factor)
        if gamma is None:
            return None, "Invalid risk_factor. Choose from 'low', 'medium', 'high'."

        # QAOA setup
        qp = QuadraticProgram("Portfolio Optimization")
        for i in range(n_assets):
            qp.binary_var(name=f"x_{i}")
        linear = -mu
        quadratic = gamma * Sigma
        qp.minimize(linear=linear, quadratic=quadratic)
        qp.linear_constraint(
            linear={f"x_{i}": 1 for i in range(n_assets)},
            sense="==",
            rhs=budget_eff,
            name="budget_constraint"
        )

        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qp)

        selected_indices = [i for i, val in enumerate(result.x) if val > 0.5]
        chosen_returns = expected_returns.loc[top_assets].iloc[selected_indices].values.flatten()
        chosen_assets = top_assets[selected_indices]
        weights = chosen_returns / chosen_returns.sum() if chosen_returns.sum() > 0 else np.zeros_like(chosen_returns)
        allocation = weights * req.total_investment
        percentage = weights * 100.0

        df = pd.DataFrame({
            "Asset": chosen_assets,
            "Expected Return": chosen_returns,
            "Weight": weights,
            "Investment": allocation,
            "Percentage": percentage,
        })
        return df, None

    cur_df, err1 = solve(current_opt)
    if err1:
        return {"error": f"Current dataset error: {err1}"}
    fut_df, err2 = solve(future_opt)
    if err2:
        return {"error": f"Future dataset error: {err2}"}

    # Compare for rebalancing
    cur = cur_df.set_index("Asset")
    fut = fut_df.set_index("Asset")

    current_assets = set(cur.index)
    future_assets = set(fut.index)

    sell = sorted(list(current_assets - future_assets))
    buy = sorted(list(future_assets - current_assets))
    common = sorted(list(current_assets & future_assets))

    rows = []
    for asset in sell:
        rows.append({
            "Action": "SELL",
            "Asset": asset,
            "Current Allocation": float(cur.loc[asset, "Investment"]) if asset in cur.index else 0.0,
            "Current %": float(cur.loc[asset, "Percentage"]) if asset in cur.index else 0.0,
            "Future Allocation": 0.0,
            "Future %": 0.0,
            "Change (Allocation)": float(-cur.loc[asset, "Investment"]) if asset in cur.index else 0.0,
        })
    for asset in buy:
        rows.append({
            "Action": "BUY",
            "Asset": asset,
            "Current Allocation": 0.0,
            "Current %": 0.0,
            "Future Allocation": float(fut.loc[asset, "Investment"]) if asset in fut.index else 0.0,
            "Future %": float(fut.loc[asset, "Percentage"]) if asset in fut.index else 0.0,
            "Change (Allocation)": float(fut.loc[asset, "Investment"]) if asset in fut.index else 0.0,
        })
    for asset in common:
        cur_pct = float(cur.loc[asset, "Percentage"]) if asset in cur.index else 0.0
        fut_pct = float(fut.loc[asset, "Percentage"]) if asset in fut.index else 0.0
        cur_alloc = float(cur.loc[asset, "Investment"]) if asset in cur.index else 0.0
        fut_alloc = float(fut.loc[asset, "Investment"]) if asset in fut.index else 0.0
        pct_diff = abs(fut_pct - cur_pct)
        action = "HOLD"
        if pct_diff > 2.0:
            action = "INCREASE" if (fut_alloc - cur_alloc) > 0 else "DECREASE"
        rows.append({
            "Action": action,
            "Asset": asset,
            "Current Allocation": cur_alloc,
            "Current %": cur_pct,
            "Future Allocation": fut_alloc,
            "Future %": fut_pct,
            "Change (Allocation)": float(fut_alloc - cur_alloc),
        })

    recs_df = pd.DataFrame(rows, columns=[
        "Action", "Asset", "Current Allocation", "Current %", "Future Allocation", "Future %", "Change (Allocation)"
    ])

    # Build JSON response
    def df_to_list(df: pd.DataFrame):
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "asset": str(r["Asset"]),
                "expected_return": float(r["Expected Return"]),
                "weight": float(r["Weight"]),
                "investment": float(r.get("Investment", 0.0)),
                "percentage": float(r.get("Percentage", 0.0)),
            })
        return rows

    current_portfolio = df_to_list(cur_df)
    future_portfolio = df_to_list(fut_df)

    recommendations = []
    for _, r in recs_df.iterrows():
        recommendations.append({
            "action": str(r["Action"]),
            "asset": str(r["Asset"]),
            "current_allocation": float(r["Current Allocation"]),
            "current_pct": float(r["Current %"]),
            "future_allocation": float(r["Future Allocation"]),
            "future_pct": float(r["Future %"]),
            "change_allocation": float(r["Change (Allocation)"]),
        })

    response = {
        "dataset": current_opt,
        "future_dataset": future_opt,
        "budget": int(req.budget),
        "risk_factor": req.risk_factor,
        "total_investment": float(req.total_investment),
        "current_portfolio": current_portfolio,
        "future_portfolio": future_portfolio,
        "recommendations": recommendations,
        "summary": {
            "sell": int(len(sell)),
            "buy": int(len(buy)),
            "rebalance_or_hold": int(len(common)),
        }
    }
    return JSONResponse(content=response)
