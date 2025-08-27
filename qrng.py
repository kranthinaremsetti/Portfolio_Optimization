import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Response
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from typing import Optional
import io

app = FastAPI()


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


df = pd.read_csv("optimal_combination.csv") #harcoded
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()
df['Return'] = df.groupby("Company")['Close'].pct_change()
df = df.dropna()


def worst_case_equity(returns, initial_equity, n_days):
    equity = initial_equity
    for _ in range(n_days):
        base_ret = np.random.choice(returns)
        u = qrng_uniform(1)[0]  
        stress_factor = 3
        daily_ret = base_ret * (stress_factor if base_ret < 0 else 1)
        equity *= (1 + daily_ret)
    return equity

@app.get("/simulate")
def simulate(
    initial_equity: Optional[float] = 200000,
    n_days: Optional[int] = 20,
    threshold: Optional[float] = 60
):
    """
    Run worst-case equity simulation and return bar chart as PNG.
    Params come from frontend (defaults: equity=100000, days=20, threshold=60%).
    """
    final_equities = {}
    for company, group in df.groupby("Company"):
        returns = group['Return'].dropna().values
        final_equities[company] = worst_case_equity(returns, initial_equity, n_days)

    plt.figure(figsize=(12,6))
    plt.bar(final_equities.keys(), final_equities.values(), color='salmon')
    plt.xticks(rotation=90)
    plt.ylabel("Equity after Worst-Case Days")
    plt.title("Stock Resilience in Worst-Case Scenario")

    ruin_level = initial_equity * (threshold / 100)  
    plt.axhline(y=ruin_level, color='green', linestyle='--',
                label=f"Ruin Threshold ({threshold}%)")

    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
