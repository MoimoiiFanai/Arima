import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Connect to the MongoDB client
client = MongoClient('mongodb+srv://CE:project1@mzuceproject.kyvn8sq.mongodb.net/')
db = client['Data']  # Database name
collection = db['Fin']  # Collection name

# Fetch the data from MongoDB
data = list(collection.find())

# Convert the fetched data to a DataFrame
df = pd.DataFrame(data)

# Print the columns and first few rows to debug
print("Columns in the DataFrame:", df.columns)
print("First few rows of the DataFrame:\n", df.head())

# Rename columns to fit ARIMA's requirements
df.rename(columns={'DATE': 'ds', 'PRICE': 'y'}, inplace=True)

# Ensure the 'ds' column is datetime
df['ds'] = pd.to_datetime(df['ds'])

# Print the first few rows after renaming and date conversion
print("First few rows after renaming and date conversion:\n", df.head())

# Exclude the time between 9 PM to 6 AM
df2 = df[(df['ds'].dt.hour > 6) & (df['ds'].dt.hour < 19)]

# Print the first few rows after filtering
print("First few rows after filtering:\n", df2.head())

# Set 'ds' as the index
df2.set_index('ds', inplace=True)

# Fit the ARIMA model on the dataset
model = ARIMA(df2['y'], order=(5, 1, 0))  # Adjust order (p, d, q) as needed
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())

# Create a dataframe with future dates for forecasting
future_dates = pd.date_range(start=df2.index[-1], periods=2*365*24, freq='H')
future_dates = future_dates[(future_dates.hour > 6) & (future_dates.hour < 19)]

# Forecast the future values
forecast = model_fit.get_forecast(steps=len(future_dates))
forecast_index = future_dates
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({
    'ds': forecast_index,
    'yhat': forecast_values,
    'yhat_lower': forecast_conf_int.iloc[:, 0],
    'yhat_upper': forecast_conf_int.iloc[:, 1]
})

# Display the forecast
print("Forecast tail:\n", forecast_df.tail())

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(df2.index, df2['y'], label='Observed')
plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='k', alpha=0.1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()