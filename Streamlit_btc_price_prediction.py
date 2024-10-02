import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Initialize session state if not already done
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
    st.session_state['prediction_df'] = None
    st.session_state['train_metrics'] = None
    st.session_state['test_metrics'] = None
    st.session_state['min_date'] = None
    st.session_state['max_date'] = None

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
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=25, validation_split=0.2)
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    train_predict = scaler.inverse_transform(
        np.concatenate([train_predict, np.zeros((train_predict.shape[0], 2))], axis=1)
    )[:, 0]
    test_predict = scaler.inverse_transform(
        np.concatenate([test_predict, np.zeros((test_predict.shape[0], 2))], axis=1)
    )[:, 0]
    
    train_dates = merged_df['Date'][time_step+1:time_step+1+len(train_predict)]
    test_dates = merged_df['Date'][time_step+1+len(train_predict):time_step+1+len(train_predict)+len(test_predict)]
    
    train_df = pd.DataFrame({
        'Date': train_dates,
        'Actual_Bitcoin': merged_df['Close_bitcoin'][time_step+1:time_step+1+len(train_predict)].values,
        'Predicted_Bitcoin': train_predict,
        'Dataset': 'Train'
    })
    
    test_df = pd.DataFrame({
        'Date': test_dates,
        'Actual_Bitcoin': merged_df['Close_bitcoin'][time_step+1+len(train_predict):time_step+1+len(train_predict)+len(test_predict)].values,
        'Predicted_Bitcoin': test_predict,
        'Dataset': 'Test'
    })
    
    prediction_df = pd.concat([train_df, test_df])
    
    train_metrics = calculate_metrics(train_df)
    test_metrics = calculate_metrics(test_df)
    
    return prediction_df.sort_values('Date', ascending=False), train_metrics, test_metrics

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

def create_plot(filtered_df):
    fig = go.Figure()

    # Add actual prices
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['Actual_Bitcoin'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add predicted prices with color differentiation
    for dataset, color in [('Train', 'green'), ('Test', 'red')]:
        mask = filtered_df['Dataset'] == dataset
        fig.add_trace(go.Scatter(
            x=filtered_df[mask]['Date'],
            y=filtered_df[mask]['Predicted_Bitcoin'],
            name=f'Predicted ({dataset})',
            line=dict(color=color)
        ))

    fig.update_layout(
        title='Bitcoin Price - Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified'
    )
    
    return fig

def main():
    st.title('Cryptocurrency Price Prediction')
    
    # Train model only if not already trained
    if not st.session_state['model_trained']:
        with st.spinner('Loading data and training model... This may take a few minutes.'):
            merged_df = load_data()
            prediction_df_sorted, train_metrics, test_metrics = train_model(merged_df)
            
            # Store results in session state
            st.session_state['prediction_df'] = prediction_df_sorted
            st.session_state['train_metrics'] = train_metrics
            st.session_state['test_metrics'] = test_metrics
            st.session_state['min_date'] = prediction_df_sorted['Date'].min()
            st.session_state['max_date'] = prediction_df_sorted['Date'].max()
            st.session_state['model_trained'] = True

    # Add retrain model button
    if st.button('Retrain Model'):
        st.session_state['model_trained'] = False
        st.rerun()

    if st.session_state['model_trained']:
        st.header('Bitcoin Price Prediction')
        
        # Date range selector
        st.sidebar.markdown('Select Date Range:')
        start_date = st.sidebar.date_input('Start date', 
                                          min_value=st.session_state['min_date'],
                                          max_value=st.session_state['max_date'],
                                          value=st.session_state['min_date'])
        end_date = st.sidebar.date_input('End date', 
                                        min_value=start_date,
                                        max_value=st.session_state['max_date'],
                                        value=st.session_state['max_date'])
        
        # Filter data based on selected dates
        filtered_df = st.session_state['prediction_df'][
            (st.session_state['prediction_df']['Date'] >= pd.to_datetime(start_date)) &
            (st.session_state['prediction_df']['Date'] <= pd.to_datetime(end_date))
        ]

        # Create and display plot
        fig = create_plot(filtered_df)
        st.plotly_chart(fig)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Training Metrics')
            for metric, value in st.session_state['train_metrics'].items():
                st.write(f"{metric}: {value:.2f}")
        
        with col2:
            st.subheader('Testing Metrics')
            for metric, value in st.session_state['test_metrics'].items():
                st.write(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()