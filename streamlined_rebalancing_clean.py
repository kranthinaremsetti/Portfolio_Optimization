import pandas as pd
import numpy as np
import os
import subprocess
import sys
from datetime import datetime

class StreamlinedRebalancingWorkflow:
    def __init__(self, initial_investment=1000000):
        """
        Streamlined portfolio rebalancing using existing notebooks
        
        Args:
            initial_investment: Total investment amount (default: 1,000,000)
        """
        self.initial_investment = initial_investment
        self.python_path = r"C:\Users\krant\OneDrive\Desktop\Quantum_hack\venv\Scripts\python.exe"
        
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

    def run_qaoa_optimization(self, dataset_name):
        """
        Run QAOA optimization for the dataset
        """
        print(f"\nRunning QAOA optimization for {dataset_name}...")
        
        # Create a temporary script for QAOA optimization
        temp_script = f"""
import pandas as pd
import numpy as np
import os

try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms import QAOA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    
    # Try newer Qiskit versions first
    try:
        from qiskit.primitives import StatevectorSampler as Sampler
    except ImportError:
        try:
            from qiskit.primitives import Sampler
        except ImportError:
            from qiskit_aer.primitives import Sampler
    
    QISKIT_AVAILABLE = True
    print("Qiskit quantum libraries loaded successfully")
    
except ImportError as e:
    print(f"Qiskit import error: {{e}}")
    print("Falling back to classical optimization only")
    QISKIT_AVAILABLE = False

# Load preprocessed data
dataset_option = '{dataset_name}'
results_dir = os.path.join('results', dataset_option)

expected_returns = pd.read_csv(os.path.join(results_dir, 'expected_returns.csv'), index_col=0)
cov_matrix = pd.read_csv(os.path.join(results_dir, 'cov_matrix.csv'), index_col=0)

print(f"Loaded data for {{len(expected_returns)}} companies")

# Calculate Sharpe ratio for each asset (assuming risk-free rate = 0)
asset_volatility = np.sqrt(np.diag(cov_matrix))
sharpe_ratio = expected_returns.values.flatten() / asset_volatility

# Get indices of top 15 assets by Sharpe ratio
N = 15
sorted_indices = np.argsort(sharpe_ratio)[::-1][:N]
top_assets = expected_returns.index[sorted_indices]

# Filter expected returns and covariance matrix for top assets
mu = expected_returns.loc[top_assets].values.flatten()
Sigma = cov_matrix.loc[top_assets, top_assets].values
n_assets = N

print(f"Selected top {{N}} assets by Sharpe ratio for QAOA optimization")

# Risk tolerance parameter
gamma = 0.5  # Medium risk tolerance

# Budget constraint: select 5 assets
budget = 5

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

print("\\nSetting up QAOA quantum optimizer...")

try:
    # Set up QAOA with quantum optimization
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)
    
    print("Running QAOA optimization...")
    result = optimizer.solve(qp)
    
    print("\\nQAOA Optimization completed!")
    print("Optimal portfolio allocation:")
    print(result)
    
    # Extract selected assets
    selected_assets = [top_assets[i] for i, x in enumerate(result.x) if x > 0.5]
    print(f"\\nSelected Portfolio Assets: {{selected_assets}}")
    
    # Calculate portfolio weights based on expected returns
    selected_indices = [i for i, x in enumerate(result.x) if x > 0.5]
    chosen_returns = expected_returns.loc[top_assets].iloc[selected_indices].values.flatten()
    
    # Compute proportional weights
    weights = chosen_returns / chosen_returns.sum()
    
    # Total investment
    total_investment = {self.initial_investment}
    allocation = weights * total_investment
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({{
        'Company': selected_assets,
        'Expected_Return': chosen_returns,
        'Weight': weights,
        'Allocation': allocation,
        'Percentage': weights * 100
    }})
    
    # Save portfolio
    portfolio_df.to_csv(f'portfolio_{{dataset_option.lower()}}.csv', index=False)
    print(f"\\nPortfolio saved as portfolio_{{dataset_option.lower()}}.csv")
    
    # Print portfolio summary
    print("\\nQAOA PORTFOLIO SUMMARY:")
    print("=" * 40)
    for _, row in portfolio_df.iterrows():
        print(f"Company: {{row['Company']}}")
        print(f"   Amount: Rs.{{row['Allocation']:,.0f}} ({{row['Percentage']:.1f}}%)")
        print(f"   Expected Return: {{row['Expected_Return']:.4f}}")
    
    print(f"\\nTotal Objective Value: {{result.fval}}")

except Exception as e:
    print(f"\\nQAOA optimization failed: {{e}}")
    print("Falling back to classical optimization...")
    
    # Fallback: Classical optimization - select top 5 by Sharpe ratio
    top_5_indices = sorted_indices[:5]
    selected_assets = expected_returns.index[top_5_indices].tolist()
    selected_returns = expected_returns.iloc[top_5_indices].values.flatten()
    
    # Calculate weights based on expected returns
    positive_returns = np.maximum(selected_returns, 0.001)
    weights = positive_returns / positive_returns.sum()
    
    # Calculate allocations
    total_investment = {self.initial_investment}
    allocations = weights * total_investment
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame({{
        'Company': selected_assets,
        'Expected_Return': selected_returns,
        'Weight': weights,
        'Allocation': allocations,
        'Percentage': weights * 100
    }})
    
    # Save portfolio
    portfolio_df.to_csv(f'portfolio_{{dataset_option.lower()}}.csv', index=False)
    print(f"\\nClassical portfolio saved as portfolio_{{dataset_option.lower()}}.csv")
    
    # Print portfolio summary
    print("\\nCLASSICAL PORTFOLIO SUMMARY:")
    print("=" * 40)
    for _, row in portfolio_df.iterrows():
        print(f"Company: {{row['Company']}}")
        print(f"   Amount: Rs.{{row['Allocation']:,.0f}} ({{row['Percentage']:.1f}}%)")
        print(f"   Expected Return: {{row['Expected_Return']:.4f}}")
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

    def compare_portfolios(self, current_portfolio, future_portfolio, dataset_name):
        """
        Compare current and future portfolios for rebalancing recommendations
        """
        print(f"\nGENERATING REBALANCING RECOMMENDATIONS")
        print("=" * 50)
        
        current_companies = set(current_portfolio['Company'])
        future_companies = set(future_portfolio['Company'])
        
        # Companies to sell
        sell_companies = current_companies - future_companies
        # Companies to buy  
        buy_companies = future_companies - current_companies
        # Companies in both
        common_companies = current_companies & future_companies
        
        print("\nREBALANCING RECOMMENDATIONS:")
        print("=" * 40)
        
        # SELL recommendations
        if sell_companies:
            print("\nSELL:")
            for company in sell_companies:
                current_data = current_portfolio[current_portfolio['Company'] == company].iloc[0]
                print(f"SELL {company}")
                print(f"   Amount: Rs.{current_data['Allocation']:,.0f} ({current_data['Percentage']:.1f}%)")
                print(f"   Reason: No longer optimal")
        
        # BUY recommendations
        if buy_companies:
            print("\nBUY:")
            for company in buy_companies:
                future_data = future_portfolio[future_portfolio['Company'] == company].iloc[0]
                print(f"BUY {company}")
                print(f"   Amount: Rs.{future_data['Allocation']:,.0f} ({future_data['Percentage']:.1f}%)")
                print(f"   Reason: New optimal choice")
        
        # REBALANCE/HOLD recommendations
        if common_companies:
            print("\nREBALANCE/HOLD:")
            for company in common_companies:
                current_data = current_portfolio[current_portfolio['Company'] == company].iloc[0]
                future_data = future_portfolio[future_portfolio['Company'] == company].iloc[0]
                
                percentage_diff = abs(future_data['Percentage'] - current_data['Percentage'])
                amount_diff = future_data['Allocation'] - current_data['Allocation']
                
                if percentage_diff > 2.0:  # Significant change
                    action = "INCREASE" if amount_diff > 0 else "DECREASE"
                    print(f"{action} {company}")
                    print(f"   Current: {current_data['Percentage']:.1f}% -> New: {future_data['Percentage']:.1f}%")
                    print(f"   Change: Rs.{amount_diff:+,.0f}")
                else:
                    print(f"HOLD {company}")
                    print(f"   Amount: Rs.{current_data['Allocation']:,.0f} ({current_data['Percentage']:.1f}%)")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"   Stocks to Sell: {len(sell_companies)}")
        print(f"   Stocks to Buy: {len(buy_companies)}")
        print(f"   Stocks to Rebalance/Hold: {len(common_companies)}")

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
            
            # Step 3: Run QAOA for current data
            print(f"\nSTEP 2: Running optimization for CURRENT data...")
            current_portfolio = self.run_qaoa_optimization(dataset_name)
            if current_portfolio is None:
                print("Failed to optimize current portfolio")
                return
            
            # Step 4: Process future dataset
            print(f"\nSTEP 3: Processing FUTURE {dataset_name} data...")
            future_dataset_name = f"{dataset_name}_FUTURE"
            if not self.run_preprocessing(future_dataset_name, future_file):
                print("Failed to process future dataset")
                return
            
            # Step 5: Run QAOA for future data
            print(f"\nSTEP 4: Running optimization for FUTURE data...")
            future_portfolio = self.run_qaoa_optimization(future_dataset_name)
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
