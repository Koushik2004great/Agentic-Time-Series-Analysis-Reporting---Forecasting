## Time Series Analysis Report: Technology Sales Forecasting

**Objective:** The aim of this analysis is to develop a robust model for forecasting monthly sales in the Technology category, using historical data from the superstore_sales.csv dataset.

**Methodology:**

1.  **Data Preparation:** The analysis focused on the 'Sales' column, filtered by the 'Technology' category, with 'Order Date' as the time index.
2.  **Stationarity Testing:** The Augmented Dickey-Fuller (ADF) test was employed to assess the stationarity of the time series. The series was found to be stationary (p-value = 7.88e-05 <= 0.05), indicating no need for differencing.
3.  **Decomposition:** Time series decomposition was performed, revealing a multiplicative seasonality.
4.  **Model Selection:** Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots identified significant lags at 12 and 12, 16 respectively. A Seasonal ARIMA (SARIMA) model was selected. Model order optimization was performed using AIC to determine the best model parameters.

**Model Details:**

The selected SARIMA model is SARIMA(0, 1, 1)(0, 1, 1, 12), with an AIC score of 718.01.

**Conclusion:**

The Ljung-Box test (p-value = 0.45) and Jarque-Bera test (p-value = 0.53) indicate that the model residuals are uncorrelated and normally distributed, respectively. These results suggest that the chosen SARIMA model is well-fitted and reliable for forecasting future sales in the Technology category. Forecasts for the next 12 months, along with confidence intervals, have been generated.
