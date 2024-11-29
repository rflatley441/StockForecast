import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#Fetch stock data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

#Calculate daily returns (observed state)
def calculate_returns(prices):
    returns = prices.pct_change().dropna()  # daily returns (percentage change)
    return returns

# Train the HMM model
def train_hmm_model(data, n_components=3):
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
    # Reshaping data for HMM model (HMM expects 2D data, with each time step as a row)
    returns_reshaped = data.values.reshape(-1, 1)  # reshape to 2D array (time, feature)
    model.fit(returns_reshaped)
    return model

# Predict hidden states (Bull, Bear, Neutral)
def predict_states(model, data):
    returns_reshaped = data.values.reshape(-1, 1)
    hidden_states = model.predict(returns_reshaped)
    return hidden_states

# Predict future state of the market
def predict_future_state(model, last_data_point):
    last_return_reshaped = np.array([last_data_point]).reshape(-1, 1)  # reshape for prediction
    future_state = model.predict(last_return_reshaped)
    return future_state[0]

# Step 6: Visualize the results
def plot_results(prices, hidden_states):
    # Ensure hidden_states and prices have the same length (length is always one off, idk why)
    if len(prices) > len(hidden_states):
        prices = prices[:len(hidden_states)]
    elif len(hidden_states) > len(prices):
        hidden_states = hidden_states[:len(prices)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices.index, prices, label='Stock Price')
    plt.scatter(prices.index, prices, c=hidden_states, cmap='viridis', marker='o')
    plt.title('Hidden States over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    today = datetime.today()
    last_month = today - timedelta(days=30)
    
    ticker = 'SPY'  # Could use multiple stocks for better analysis
    start_date_train = '2015-01-01'
    end_date_train = last_month.strftime('%Y-%m-%d')  
    start_date_test = last_month.strftime('%Y-%m-%d')
    end_date_test = today.strftime('%Y-%m-%d') 
    
    # Fetch training and testing stock data
    stock_data_train = fetch_data(ticker, start_date_train, end_date_train)
    stock_data_test = fetch_data(ticker, start_date_test, end_date_test)
    
    # Calculate returns (observed state)
    returns_train = calculate_returns(stock_data_train)
    returns_test = calculate_returns(stock_data_test)
    
    # Train HMM model
    hmm_model = train_hmm_model(returns_train)
    
    # Predict hidden states (Bull, Bear, Neutral) for the training data
    hidden_states_train = predict_states(hmm_model, returns_train)
    
    # Map states to readable labels (for example: 0 - Bull, 1 - Bear, 2 - Neutral)
    state_labels = ['Bull', 'Bear', 'Neutral']
    predicted_labels_train = [state_labels[state] for state in hidden_states_train]
    
    # Visualize the results for the training period
    plot_results(stock_data_train, hidden_states_train)
    
    # Predict future market state based on the last observed return in the training data
    last_return_train = returns_train.iloc[-1]  # Last return value of the training period
    future_state_train = predict_future_state(hmm_model, last_return_train)
    
    print(f"The predicted market state for the next period based on the training data is: {state_labels[future_state_train]}")
    
    # Now evaluate with the test data (this month's stock data)
    hidden_states_test = predict_states(hmm_model, returns_test)
    
    # Visualize the results for the test period
    plot_results(stock_data_test, hidden_states_test)
    
    # Compare the prediction with the real stock returns (optional evaluation metric)
    last_return_test = returns_test.iloc[-1]  # Last return value of the test period
    future_state_test = predict_future_state(hmm_model, last_return_test)
    
    print(f"The predicted market state for this month is: {state_labels[future_state_test]}")
