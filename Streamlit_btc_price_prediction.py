# importing libraries

import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Fetch Bitcoin historical data
btc_data = yf.download('BTC-USD', start='2015-01-01', end='2024-09-01')
btc_data.reset_index(inplace=True)


# Calculate the moving averages and returns
btc_data['MA_30'] = btc_data['Close'].rolling(window=30).mean()
btc_data['Returns'] = btc_data['Close'].pct_change()

# Drop missing values
btc_data.dropna(inplace=True)


# Prepare the feature set and target
X = btc_data[['MA_30', 'Returns']]
y = btc_data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Streamlit app title
st.title('Bitcoin Price Prediction')

# Sidebar for date input
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-09-01'))

# Filter the data based on the selected dates
filtered_data = btc_data[(btc_data['Date'] >= str(start_date)) & (btc_data['Date'] <= str(end_date))]

# Plot the actual vs predicted prices
st.subheader('Price Data')
st.line_chart(filtered_data['Close'])


# Calculate MAE, MSE, RMSE, and R² score
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Display the results in the Streamlit app
st.subheader('Model Evaluation')
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")


# Display predictions
st.subheader('Predictions')

plt.figure(figsize=(10,6))
plt.plot(filtered_data['Date'][-len(predictions):], y_test, label='Actual')  # Use Date for x-axis
plt.plot(filtered_data['Date'][-len(predictions):], predictions, label='Predicted')  # Use Date for x-axis
plt.xlabel('Date')  # Label the x-axis as "Date"
plt.ylabel('Bitcoin Price (USD)')  # Label the y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
st.pyplot(plt)








