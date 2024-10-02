import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Caching the data loading and model training
@st.cache_data
def load_data():
    bitcoin_df = pd.read_csv('Data/coin_Bitcoin.csv')
    ethereum_df = pd.read_csv('Data/coin_Ethereum.csv')
    binancecoin_df = pd.read_csv('Data/coin_BinanceCoin.csv')
    
    bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'], format="%d-%m-%Y")
    ethereum_df['Date'] = pd.to_datetime(ethereum_df['Date'], format="%d-%m-%Y")
    binancecoin_df['Date'] = pd.to_datetime(binancecoin_df['Date'], format="%d-%m-%Y")
    
    merged_df = bitcoin_df[['Date', 'Close']].merge(
        ethereum_df[['Date', 'Close']], on='Date', suffixes=('_bitcoin', '_ethereum')
    ).merge(binancecoin_df[['Date', 'Close']], on='Date')
    merged_df.rename(columns={'Close': 'Close_binancecoin'}, inplace=True)
    
    return merged_df

@st.cache_resource
def train_model(merged_df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_df[['Close_bitcoin', 'Close_ethereum', 'Close_binancecoin']])
    
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data)-time_step-1):
            X.append(data[i:(i+time_step), :])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=25, validation_split=0.2)
    
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(
        np.concatenate([predicted_prices, np.zeros((predicted_prices.shape[0], 2))], axis=1)
    )[:, 0]
    
    prediction_df = pd.DataFrame({
        'Date': merged_df['Date'][time_step+1:],
        'Actual_Bitcoin': merged_df['Close_bitcoin'][time_step+1:],
        'Predicted_Bitcoin': predicted_prices
    })
    
    metrics = calculate_metrics(prediction_df)
    
    return prediction_df.sort_values('Date', ascending=False), metrics


def calculate_metrics(prediction_df):
    mse = mean_squared_error(prediction_df['Actual_Bitcoin'], prediction_df['Predicted_Bitcoin'])
    rmse = mse**0.5
    mae = mean_absolute_error(prediction_df['Actual_Bitcoin'], prediction_df['Predicted_Bitcoin'])
    r2 = r2_score(prediction_df['Actual_Bitcoin'], prediction_df['Predicted_Bitcoin'])
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

def main():
    st.title('Cryptocurrency Price Prediction')
    
    # Add retrain model button
    if st.button('Retrain Model'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # Add loading spinner
    with st.spinner('Loading data and training model...'):
        # Load data and train model (cached)
        merged_df = load_data()
        prediction_df_sorted, metrics = train_model(merged_df)


    st.header('Bitcoin Price Prediction')
    
    # Date range selector
    st.sidebar.markdown('Select Date Range:')
    min_date = prediction_df_sorted['Date'].min()
    max_date = prediction_df_sorted['Date'].max()
    
    # Add date inputs for the user to select a range
    start_date = st.sidebar.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input('End date', min_value=start_date, max_value=max_date, value=max_date)
    
    # Filter the DataFrame based on the selected date range
    filtered_df = prediction_df_sorted[
        (prediction_df_sorted['Date'] >= pd.to_datetime(start_date)) &
        (prediction_df_sorted['Date'] <= pd.to_datetime(end_date))
    ]
    
    # Create and display the line chart
    line_chart_df = filtered_df[['Date', 'Actual_Bitcoin', 'Predicted_Bitcoin']].set_index('Date')
    st.line_chart(line_chart_df)
    
    # Display metrics
    st.subheader('Model Evaluation')
    for metric, value in metrics.items():
        st.write(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()