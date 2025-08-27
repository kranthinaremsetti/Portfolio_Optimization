
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Response, Query
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from typing import Optional
import io
import os

app = FastAPI()

def get_optimized_assets(dataset_option: str) -> Optional[list]:
    """
    Loads the optimized_stocks.csv file from results/<normalized_dataset_option>/ and returns the list of assets.
    """
    folder = (dataset_option)
    assets_path = os.path.join("results", folder, "optimized_stocks.csv")
    if not os.path.exists(assets_path):
        return None
    df_assets = pd.read_csv(assets_path)
    # If the file contains only names, return as list
    if 'Asset' in df_assets.columns:
        return df_assets['Asset'].astype(str).tolist()
    # If the file contains full data, return the dataframe for further use
    return df_assets

# Quantum random number generator for uniform distribution
def qrng_uniform(n_uniforms=1, bits_per=32):
    n_qubits = bits_per
    backend = AerSimulator()
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    job = backend.run(qc, shots=n_uniforms)
    results = job.result().get_counts()
    uniforms = []
    for bitstring in results:
        x = int(bitstring, 2)
        u = (x + 0.5) / (2**bits_per)
        uniforms.extend([u] * results[bitstring])
    return np.array(uniforms[:n_uniforms])

# Stress testing logic: simulate worst-case equity
def worst_case_equity(returns, initial_equity, n_days, stress_factor=2):
    equity = initial_equity
    neg_returns = returns[returns < 0]
    pos_returns = returns[returns >= 0]
    half_days = n_days // 2

    # Randomly select half days from negative returns
    if len(neg_returns) > 0:
        neg_choices = np.random.choice(neg_returns, half_days, replace=True)
    else:
        neg_choices = np.random.choice(returns, half_days, replace=True)

    # Randomly select half days from positive returns
    if len(pos_returns) > 0:
        pos_choices = np.random.choice(pos_returns, n_days - half_days, replace=True)
    else:
        pos_choices = np.random.choice(returns, n_days - half_days, replace=True)

    # Apply stress factor to negative days
    for base_ret in neg_choices:
        equity *= (1 + base_ret * stress_factor)
    for base_ret in pos_choices:
        equity *= (1 + base_ret)
    return equity

@app.get("/simulate")
def simulate(
    dataset_option: str = Query("optimized stocks"),
    initial_equity: float = Query(200000),
    n_days: int = Query(20),
    threshold: float = Query(60)
):
    """
    Run worst-case equity simulation using returns data from optimized_stocks.csv only.
    """
    try:
        folder = (dataset_option)
        assets_path = os.path.join("results", folder, "optimized_stocks.csv")
        if not os.path.exists(assets_path):
            return Response(content="No optimized_stocks.csv found for selected dataset.", media_type="text/plain", status_code=400)
        df = pd.read_csv(assets_path)
        # If only names, error
        if df.shape[1] == 1 and 'Asset' in df.columns:
            return Response(content="optimized_stocks.csv contains only asset names, not returns data.", media_type="text/plain", status_code=400)
    except Exception as e:
        return Response(content=str(e), media_type="text/plain", status_code=400)

    final_equities = {}
    # Assume wide format: Date, Stock1, Stock2, ...
    stock_names = [col for col in df.columns if col != 'Date']
    for stock in stock_names:
        returns = df[stock].dropna().values
        if len(returns) == 0:
            continue
        final_equities[stock] = worst_case_equity(returns, initial_equity, n_days)

    if not final_equities:
        return Response(content="No equities found for selected dataset.", media_type="text/plain", status_code=400)

    plt.figure(figsize=(12,6))
    plt.bar(final_equities.keys(), final_equities.values(), color='salmon')
    plt.xticks(rotation=90)
    plt.ylabel("Equity after Worst-Case Days")
    plt.title("Stock Resilience in Worst-Case Scenario")

    ruin_level = initial_equity * (threshold / 100)
    plt.axhline(y=ruin_level, color='green', linestyle='--',
                label=f"Ruin Threshold ({threshold}%)")

    plt.ylim(0, initial_equity)  # Limit y-axis to initial equity
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")