import pandas as pd
import numpy as np
import os
import subprocess
import sys
from datetime import datetime
import io
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

class StreamlinedRebalancingWorkflow:
    def __init__(self, initial_investment=1000000):
        """
        Streamlined portfolio rebalancing using existing notebooks
        
        Args:
            initial_investment: Total investment amount (default: 1,000,000)
        """
        self.initial_investment = initial_investment
        # Use current Python interpreter for portability
        self.python_path = sys.executable or "python"
        
        # Dataset options for user selection
        self.dataset_options = {
            '1': {'name': 'NIFTY50', 'current': 'NIFTY50_Combined.csv', 'future': 'NIFTY50_withQ1_2025.csv'},
            '2': {'name': 'NASDAQ100', 'current': 'NASDAQ100_Combined.csv', 'future': 'NASDAQ100_withQ1.csv'},
            '3': {'name': 'CRYPTO50', 'current': 'CRYPTO50_Combined.csv', 'future': 'CRYPTO50_withQ1.csv'}
        }

    def select_dataset(self):
        """
        Let user select which dataset to analyze
        """
        print("\nSELECT DATASET FOR PORTFOLIO OPTIMIZATION")
        print("=" * 50)
        print("1. NIFTY50 - Indian Stock Market")
        print("2. NASDAQ100 - US Technology Stocks") 
        print("3. CRYPTO50 - Cryptocurrency Market")
        
        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice in self.dataset_options:
                dataset_info = self.dataset_options[choice]
                print(f"\nSelected: {dataset_info['name']}")
                return dataset_info
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def run_preprocessing(self, dataset_name, csv_file):
        """
        Run the preprocessing for selected dataset
        """
        print(f"\nRunning preprocessing for {dataset_name}...")
        print(f"Processing file: {csv_file}")
        
        # Create a temporary script to run preprocessing
        temp_script = f"""
import pandas as pd
import numpy as np
import os

# Set dataset option
dataset_option = '{dataset_name}'

# Load and process data
print(f"Processing {{dataset_option}} dataset...")
df = pd.read_csv('{csv_file}')
df['Date'] = pd.to_datetime(df['Date'])

# Create pivot table
pivot_df = df.pivot(index='Date', columns='Company', values='Close')

# Remove columns with too many NaN values
threshold = len(pivot_df) * 0.5
pivot_df = pivot_df.dropna(axis=1, thresh=threshold)

# Fill NaN values
pivot_df = pivot_df.ffill().bfill()

# Calculate daily returns
daily_returns = pivot_df.pct_change().dropna()

# Calculate expected returns
expected_returns = daily_returns.mean()

# Calculate covariance matrix
cov_matrix = daily_returns.cov()

# Ensure results directory exists
results_dir = os.path.join('results', dataset_option)
os.makedirs(results_dir, exist_ok=True)

# Save results
expected_returns.to_csv(os.path.join(results_dir, 'expected_returns.csv'))
daily_returns.to_csv(os.path.join(results_dir, 'daily_returns.csv'))
cov_matrix.to_csv(os.path.join(results_dir, 'cov_matrix.csv'))

print(f"Preprocessing completed for {{dataset_option}}")
print(f"Processed {{len(expected_returns)}} companies")
"""
        
        # Write and execute temporary script
        with open('temp_preprocessing.py', 'w', encoding='utf-8') as f:
            f.write(temp_script)
        
        try:
            result = subprocess.run([self.python_path, 'temp_preprocessing.py'], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error in preprocessing: {e}")
            print(f"Error output: {e.stderr}")
            return False
        finally:
            # Clean up temp file
            if os.path.exists('temp_preprocessing.py'):
                os.remove('temp_preprocessing.py')

    def run_qaoa_optimization(self, dataset_name, budget=5, risk_factor="medium"):
        """
        Run QAOA optimization for the dataset
        """
        print(f"\nRunning QAOA optimization for {dataset_name}...")
        
        # Create a temporary script for QAOA optimization
        temp_script = f"""
import pandas as pd
import numpy as np
import os

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Load preprocessed data
dataset_option = '{dataset_name}'
results_dir = os.path.join('results', dataset_option)

expected_returns = pd.read_csv(os.path.join(results_dir, 'expected_returns.csv'), index_col=0)
cov_matrix = pd.read_csv(os.path.join(results_dir, 'cov_matrix.csv'), index_col=0)

print(f"Loaded data for {{len(expected_returns)}} companies")

# Calculate Sharpe ratio for each asset (assuming risk-free rate = 0)
asset_volatility = np.sqrt(np.diag(cov_matrix))
sharpe_ratio = expected_returns.values.flatten() / asset_volatility

# Select top assets similarly to API: N = max(budget, 1), effectively N = budget
budget = int({budget})
N = max(budget, 1)
sorted_indices = np.argsort(sharpe_ratio)[::-1][:max(N, budget)]
top_assets = expected_returns.index[sorted_indices]

# Filter expected returns and covariance matrix for top assets
mu = expected_returns.loc[top_assets].values.flatten()
Sigma = cov_matrix.loc[top_assets, top_assets].values
n_assets = len(top_assets)

print(f"Selected top {{n_assets}} assets by Sharpe ratio for QAOA optimization")

# Risk tolerance parameter mapping like API
gamma_map = {{
    'low': 0.1,
    'medium': 0.5,
    'high': 1.0
}}
risk_factor = '{risk_factor}'
if risk_factor not in gamma_map:
    raise ValueError("Invalid risk_factor. Choose from 'low', 'medium', 'high'.")
gamma = gamma_map[risk_factor]

# Budget constraint: select exactly min(budget, n_assets)
budget = min(budget, n_assets)

print(f"Budget constraint: selecting {{budget}} out of {{n_assets}} assets")

# Create QAOA optimization problem
qp = QuadraticProgram("Portfolio Optimization")

# Decision variables: x_i = 1 if asset i is chosen, 0 otherwise
for i in range(n_assets):
    qp.binary_var(name=f"x_{{i}}")

# Objective: maximize returns - gamma * risk
# Equivalent to: minimize -mu^T x + gamma * x^T Σ x
linear = -mu
quadratic = gamma * Sigma
qp.minimize(linear=linear, quadratic=quadratic)

# Budget constraint: select exactly "budget" assets
qp.linear_constraint(
    linear={{f"x_{{i}}": 1 for i in range(n_assets)}},
    sense="==",
    rhs=budget,
    name="budget_constraint"
)

print("\nSetting up QAOA quantum optimizer...")

# Set up QAOA with quantum optimization like API
sampler = Sampler()
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
optimizer = MinimumEigenOptimizer(qaoa)

print("Running QAOA optimization...")
result = optimizer.solve(qp)

print("\nQAOA Optimization completed!")
print("Optimal portfolio allocation:")
print(result)

# Extract selected assets
selected_assets = [top_assets[i] for i, x in enumerate(result.x) if x > 0.5]
print(f"\nSelected Portfolio Assets: {{selected_assets}}")

# Calculate portfolio weights based on expected returns
selected_indices = [i for i, x in enumerate(result.x) if x > 0.5]
chosen_returns = expected_returns.loc[top_assets].iloc[selected_indices].values.flatten()

# Compute proportional weights
weights = chosen_returns / chosen_returns.sum() if chosen_returns.sum() > 0 else np.zeros_like(chosen_returns)

# Total investment
total_investment = {self.initial_investment}
allocation = weights * total_investment

# Create portfolio DataFrame using API-like schema
portfolio_df = pd.DataFrame({{
    'Asset': selected_assets,
    'Expected Return': chosen_returns,
    'Weight': weights,
    'Investment': allocation,
    'Percentage': weights * 100
}})

# Save portfolio
portfolio_df.to_csv(f'portfolio_{{dataset_option.lower()}}.csv', index=False)
print(f"\nPortfolio saved as portfolio_{{dataset_option.lower()}}.csv")

# Print portfolio summary
print("\nQAOA PORTFOLIO SUMMARY:")
print("=" * 40)
for _, row in portfolio_df.iterrows():
    print(f"Asset: {{row['Asset']}}")
    print(f"   Amount: Rs.{{row['Investment']:,.0f}} ({{row['Percentage']:.1f}}%)")
    print(f"   Expected Return: {{row['Expected Return']:.4f}}")

print(f"\nTotal Objective Value: {{result.fval}}")
"""
        
        # Write and execute temporary script
        with open('temp_qaoa.py', 'w', encoding='utf-8') as f:
            f.write(temp_script)
        
        try:
            result = subprocess.run([self.python_path, 'temp_qaoa.py'], 
                                  capture_output=True, text=True, check=True)
            print(result.stdout)
            
            # Load the generated portfolio
            portfolio_file = f'portfolio_{dataset_name.lower()}.csv'
            if os.path.exists(portfolio_file):
                portfolio = pd.read_csv(portfolio_file)
                return portfolio
            else:
                print(f"Portfolio file not found: {portfolio_file}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error in QAOA optimization: {e}")
            print(f"Error output: {e.stderr}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists('temp_qaoa.py'):
                os.remove('temp_qaoa.py')

    def run_api_optimization(self, dataset_name: str, budget: int, risk_factor: str, total_investment: float,
                             api_base: str = "http://127.0.0.1:8000") -> Optional[pd.DataFrame]:
        """Call the FastAPI /optimize endpoint and return a DataFrame matching API schema.

        Returns None on failure. Requires 'requests' to be installed and API server running.
        """
        if requests is None:
            print("Requests library not available. Install it or use local optimization.")
            return None
        url = api_base.rstrip('/') + "/optimize"
        payload = {
            "dataset_option": dataset_name,
            "budget": int(budget),
            "risk_factor": risk_factor,
            "total_investment": float(total_investment)
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code != 200:
                print(f"API error {resp.status_code}: {resp.text[:300]}")
                return None
            # Expect CSV
            csv_text = resp.text
            df = pd.read_csv(io.StringIO(csv_text))
            # Ensure required columns exist
            required_cols = {"Asset", "Expected Return", "Weight", "Investment"}
            if not required_cols.issubset(set(df.columns)):
                print("API response missing expected columns.")
                return None
            # Derive Percentage if not present
            if "Percentage" not in df.columns:
                total = df["Investment"].sum()
                df["Percentage"] = (df["Investment"] / total * 100.0) if total > 0 else 0.0
            return df
        except Exception as e:
            print(f"Failed to call API: {e}")
            return None

    def compare_portfolios(self, current_portfolio, future_portfolio, dataset_name):
        """
        Compare current and future portfolios for rebalancing recommendations
        """
        print(f"\nGENERATING REBALANCING RECOMMENDATIONS")
        print("=" * 50)

        # Use API-aligned columns
        current_assets = set(current_portfolio['Asset'])
        future_assets = set(future_portfolio['Asset'])

        # Assets to sell/buy/common
        sell_assets = current_assets - future_assets
        buy_assets = future_assets - current_assets
        common_assets = current_assets & future_assets

        print("\nREBALANCING RECOMMENDATIONS:")
        print("=" * 40)
        
        # SELL recommendations
        if sell_assets:
            print("\nSELL:")
            for asset in sell_assets:
                current_data = current_portfolio[current_portfolio['Asset'] == asset].iloc[0]
                print(f"SELL {asset}")
                print(f"   Amount: Rs.{current_data['Investment']:,.0f} ({current_data['Percentage']:.1f}%)")
                print(f"   Reason: No longer optimal")
        
        # BUY recommendations
        if buy_assets:
            print("\nBUY:")
            for asset in buy_assets:
                future_data = future_portfolio[future_portfolio['Asset'] == asset].iloc[0]
                print(f"BUY {asset}")
                print(f"   Amount: Rs.{future_data['Investment']:,.0f} ({future_data['Percentage']:.1f}%)")
                print(f"   Reason: New optimal choice")
        
        # REBALANCE/HOLD recommendations
        if common_assets:
            print("\nREBALANCE/HOLD:")
            for asset in common_assets:
                current_data = current_portfolio[current_portfolio['Asset'] == asset].iloc[0]
                future_data = future_portfolio[future_portfolio['Asset'] == asset].iloc[0]

                percentage_diff = abs(future_data['Percentage'] - current_data['Percentage'])
                amount_diff = future_data['Investment'] - current_data['Investment']

                if percentage_diff > 2.0:  # Significant change
                    action = "INCREASE" if amount_diff > 0 else "DECREASE"
                    print(f"{action} {asset}")
                    print(f"   Current: {current_data['Percentage']:.1f}% -> New: {future_data['Percentage']:.1f}%")
                    print(f"   Change: Rs.{amount_diff:+,.0f}")
                else:
                    print(f"HOLD {asset}")
                    print(f"   Amount: Rs.{current_data['Investment']:,.0f} ({current_data['Percentage']:.1f}%)")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"   Assets to Sell: {len(sell_assets)}")
        print(f"   Assets to Buy: {len(buy_assets)}")
        print(f"   Assets to Rebalance/Hold: {len(common_assets)}")

    def run_complete_workflow(self):
        """
        Run the complete streamlined workflow
        """
        print("STREAMLINED PORTFOLIO REBALANCING WORKFLOW")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Investment Amount: Rs.{self.initial_investment:,}")
        
        # Step 1: Let user select dataset
        dataset_info = self.select_dataset()
        dataset_name = dataset_info['name']
        current_file = dataset_info['current']
        future_file = dataset_info['future']
        
        # Check if files exist
        if not os.path.exists(current_file):
            print(f"Error: {current_file} not found!")
            return
        
        if not os.path.exists(future_file):
            print(f"Error: {future_file} not found!")
            return
        
        try:
            # Step 2: Process current dataset
            print(f"\nSTEP 1: Processing CURRENT {dataset_name} data...")
            if not self.run_preprocessing(dataset_name, current_file):
                print("Failed to process current dataset")
                return
            
            # Step 2: Optimization parameters and mode
            use_api = False
            mode = input("Use API for optimization? (y/N): ").strip().lower()
            if mode == 'y':
                if requests is None:
                    print("'requests' not installed; falling back to local optimization.")
                else:
                    use_api = True
            # Ask optimization parameters once
            print("\nEnter optimization parameters (like API):")
            while True:
                try:
                    budget = int(input("Budget (number of assets to select): ").strip())
                    break
                except ValueError:
                    print("Please enter a valid integer for budget.")
            risk_factor = input("Risk factor (low/medium/high) [medium]: ").strip().lower() or "medium"

            print(f"\nSTEP 2: Running optimization for CURRENT data...")
            if use_api:
                current_portfolio = self.run_api_optimization(dataset_name, budget, risk_factor, self.initial_investment)
                if current_portfolio is not None:
                    current_file_out = f'portfolio_{dataset_name.lower()}.csv'
                    current_portfolio.to_csv(current_file_out, index=False)
                    print(f"Saved current portfolio from API to {current_file_out}")
            else:
                current_portfolio = self.run_qaoa_optimization(dataset_name, budget=budget, risk_factor=risk_factor)
            if current_portfolio is None:
                print("Failed to optimize current portfolio")
                return
            
            # Step 4: Process future dataset
            print(f"\nSTEP 3: Processing FUTURE {dataset_name} data...")
            future_dataset_name = f"{dataset_name}_FUTURE"
            if not self.run_preprocessing(future_dataset_name, future_file):
                print("Failed to process future dataset")
                return
            
            # Step 4: Run optimization for FUTURE data
            print(f"\nSTEP 4: Running optimization for FUTURE data...")
            if use_api:
                future_portfolio = self.run_api_optimization(future_dataset_name, budget, risk_factor, self.initial_investment)
                if future_portfolio is not None:
                    future_file_out = f'portfolio_{future_dataset_name.lower()}.csv'
                    future_portfolio.to_csv(future_file_out, index=False)
                    print(f"Saved future portfolio from API to {future_file_out}")
            else:
                future_portfolio = self.run_qaoa_optimization(future_dataset_name, budget=budget, risk_factor=risk_factor)
            if future_portfolio is None:
                print("Failed to optimize future portfolio")
                return
            
            # Step 6: Compare portfolios and generate recommendations
            print(f"\nSTEP 5: Comparing portfolios...")
            self.compare_portfolios(current_portfolio, future_portfolio, dataset_name)
            
            print(f"\nWORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"Results saved in portfolio CSV files")
            
        except Exception as e:
            print(f"Error in workflow: {e}")
            import traceback
            traceback.print_exc()

# Main execution
if __name__ == "__main__":
    # Create workflow instance
    workflow = StreamlinedRebalancingWorkflow(initial_investment=1000000)
    
    # Run the complete workflow
    workflow.run_complete_workflow()
