import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen
import joblib

class VECM_Module:
    """
    Implements VECM methodology: Stationarity -> Cointegration -> Estimation -> Feature Extraction.
    Source: 
    """
    def __init__(self, data):
        self.data = data
        self.model_fit = None

    def check_stationarity(self):
        """Checks unit roots using ADF (Augmented Dickey-Fuller). Source: """
        print("\n--- Stationarity Check (ADF Test) ---")
        results = {}
        for col in self.data.columns:
            res = adfuller(self.data[col])
            print(f"{col}: p-value = {res[1]:.4f}")
            results[col] = res[1]
        return results

    def run_cointegration(self):
        """Uses Johansen test to find rank. Source: """
        print("\n--- Johansen Cointegration Test ---")
        johansen = coint_johansen(self.data, det_order=-1, k_ar_diff=1)
        # Simplified rank determination: count how many trace stats > critical value (95%)
        rank = sum(johansen.lr1 > johansen.cvt[:, 1])
        print(f"Estimated Cointegration Rank: {rank}")
        return rank

    def fit(self, rank=1):
        """Fits VECM and extracts residuals. Source: """
        # Lag selection using AIC 
        lag_order = select_order(self.data, maxlags=5, deterministic="ci")
        selected_lags = lag_order.aic
        print(f"Selected Lag Order (AIC): {selected_lags}")

        model = VECM(self.data, k_ar_diff=selected_lags, coint_rank=rank, deterministic='ci')
        self.model_fit = model.fit()
        print("VECM Model Fitted Successfully.")
        return self.model_fit

    def get_residuals(self):
        """Extracts residuals as input for LSTM. Source: """
        if not self.model_fit:
            raise ValueError("Fit model first.")
        
        # Get residuals and align index (drop first k_ar rows used for lags)
        resid = pd.DataFrame(
            self.model_fit.resid,
            columns=[f"vecm_resid_{c}" for c in self.data.columns],
            index=self.data.index[self.model_fit.k_ar:]
        )
        return resid

# --- 1. GENERATE SYNTHETIC DATA (Instance Input) ---
# Creates data mimicking Stock Price, Interest Rate, and GDP
np.random.seed(42)
dates = pd.date_range(start="2015-01-01", periods=1000, freq="B")
common_trend = np.cumsum(np.random.normal(0, 1, 1000))

data = pd.DataFrame(index=dates)
data['Stock_Close'] = 100 + 2*common_trend + np.random.normal(0, 5, 1000) # Stock
data['Interest_Rate'] = 5 - 0.5*common_trend + np.random.normal(0, 0.5, 1000) # Macro 1
data['GDP_Proxy'] = 50 + 1.5*common_trend + np.random.normal(0, 2, 1000) # Macro 2

print("Instance Input Data Head:\n", data.head())
data.to_csv("raw_data.csv") # Save for next steps

# --- 2. RUN VECM ---
vecm = VECM_Module(data)
vecm.check_stationarity()
rank = vecm.run_cointegration()
if rank == 0: rank = 1 # Force rank 1 for demo purposes if synthetic data is weak
vecm.fit(rank)
vecm_residuals = vecm.get_residuals()

# Save VECM residuals
vecm_residuals.to_csv("vecm_residuals.csv")
print("\nVECM Residuals saved to 'vecm_residuals.csv'")
