# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Generate sample data for demonstration (this could be your historical crypto price data)
np.random.seed(42)
data = {
    'date': pd.date_range(start='2023-01-01', periods=300),  # Date range
    'price': 100 + np.cumsum(np.random.normal(0, 1, 300))  # Simulated price data with random walk
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(df['price'], label='Original Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Cryptocurrency Prices')
plt.legend()
plt.show()

# Fit an ARIMA model
model = ARIMA(df['price'], order=(5, 1, 0))  # ARIMA(p=5, d=1, q=0) - Adjust p, d, q based on data characteristics
model_fit = model.fit()

# Make predictions
forecast = model_fit.forecast(steps=30)  # Forecast for 30 future days

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(df['price'], label='Original Data')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ARIMA Forecast for Cryptocurrency Prices')
plt.legend()
plt.show()

# Print forecasted values
print(forecast)
