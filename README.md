# Portfolio_Optimization

## FastAPI endpoints

This repo provides a small FastAPI service to optimize and rebalance portfolios using QAOA on precomputed statistics.

### 1) Setup

- Python 3.12 recommended. Create/activate your venv and install dependencies:

```powershell
pip install -r requirements.txt
```

Ensure results folders exist for your dataset(s). Each dataset needs:

```
results/<DATASET>/expected_returns.csv
results/<DATASET>/cov_matrix.csv
```

You can generate these via the streamlined workflow script (it preprocesses raw CSVs), which will also create a "Future" variant automatically:

```
results/NASDAQ/
results/NASDAQ_Future/
results/NIFTY50/
results/NIFTY50_Future/
results/Crypto/
results/Crypto_Future/
```

### 2) Start the API

```powershell
uvicorn qaoa_api:app --reload
```

Open the docs at http://127.0.0.1:8000/docs

### 3) Endpoints

#### POST /optimize

Optimizes a single dataset and returns a CSV with allocation.

Request (application/json):

```
{
	"dataset_option": "NASDAQ",   // also accepts NASDAQ100, Crypto, CRYPTO50, NIFTY50; future variants end with _Future
	"budget": 5,                   // number of assets to select
	"risk_factor": "medium",      // one of: low | medium | high
	"total_investment": 100000
}
```

Response: text/csv attachment named `portfolio_allocation.csv` with columns:
- Asset
- Expected Return
- Weight
- Investment

Errors are returned as JSON `{ "error": "..." }`.

Examples (PowerShell):

```powershell
$body = @{ dataset_option = "NASDAQ"; budget = 5; risk_factor = "medium"; total_investment = 100000 } | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:8000/optimize" -Method POST -ContentType "application/json" -Body $body -OutFile "portfolio_nasdaq.csv"
```

#### POST /rebalance

Optimizes current and future datasets and returns a combined CSV with recommendations.

Request (application/json):

```
{
	"dataset_option": "NIFTY50",          // current dataset
	"future_dataset_option": "NIFTY50_Future", // optional (defaults to <dataset>_Future)
	"budget": 5,
	"risk_factor": "medium",
	"total_investment": 100000
}
```

Response: text/csv attachment containing three sections:
1) Current Portfolio
2) Future Portfolio
3) Rebalancing Recommendations (Action, Asset, Current Allocation/%, Future Allocation/%, Change)

Example:

```powershell
$body = @{ dataset_option = "NIFTY50"; budget = 5; risk_factor = "medium"; total_investment = 100000 } | ConvertTo-Json
Invoke-WebRequest -Uri "http://127.0.0.1:8000/rebalance" -Method POST -ContentType "application/json" -Body $body -OutFile "rebalancing_nifty50.csv"
```

### 4) Dataset naming normalization

The API maps common aliases to your results folders:
- NASDAQ or NASDAQ100 -> results/NASDAQ
- NIFTY50 -> results/NIFTY50
- Crypto or CRYPTO50 -> results/Crypto
- Future variants: any of `_Future`, `_FUTURE`, ` future`, `-future` map to `<Base>_Future`

So these all work: `"NASDAQ100"`, `"NASDAQ_FUTURE"`, `"CRYPTO50_Future"`.

### 5) Troubleshooting

- 500 or error: missing results files
	- Ensure the files exist, e.g.: `results/NASDAQ/expected_returns.csv`, `results/NASDAQ/cov_matrix.csv`.
	- For future: `results/NASDAQ_Future/expected_returns.csv`, `results/NASDAQ_Future/cov_matrix.csv`.

- No assets selected / trivial solution
	- If `budget` equals the candidate pool, selection can be trivial. Use a reasonable budget given your data or increase the candidate pool upstream if needed.

- Qiskit not installed errors
	- Install dependencies from requirements.txt. The service uses local `qiskit.primitives.Sampler` and doesn’t require IBM cloud.

### 6) Preprocessing helper

Use the streamlined workflow to create the results folders from raw CSVs and optionally call the API automatically during rebalancing.

```powershell
python streamlined_rebalancing_clean.py
```

It will prompt for the dataset, optionally use the API, and save portfolios before printing SELL/BUY/HOLD recommendations.
