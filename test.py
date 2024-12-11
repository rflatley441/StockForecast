import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.dates as mdates


def fetch_data(ticker, start_date, end_date):
    
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

def train_test_split(ticker):
    today = datetime.today()
    last_month = today - timedelta(days=120)

    start_date_train = '2015-01-01'
    end_date_train = last_month.strftime('%Y-%m-%d')
    start_date_test = last_month.strftime('%Y-%m-%d')
    end_date_test = today.strftime('%Y-%m-%d')

    stock_data_train = fetch_data(ticker, start_date_train, end_date_train)
    stock_data_test = fetch_data(ticker, start_date_test, end_date_test)

    return stock_data_train, stock_data_test


def calculate_returns(prices):
    returns = prices.pct_change().dropna() * 100
    return returns

def train_hmm_model(data, n_components=3, random_state=42):
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=10000, random_state=random_state)
    returns_reshaped = data.values.reshape(-1, 1)
    model.fit(returns_reshaped)
    return model

def predict_states(model, data):
    returns_reshaped = data.values.reshape(-1, 1)
    hidden_states = model.predict(returns_reshaped)
    return hidden_states

def plot_results(prices, hidden_states, state_label):
    # Ensure the length of prices and hidden_states match
    if len(prices) > len(hidden_states):
        prices = prices[:len(hidden_states)]
    elif len(hidden_states) > len(prices):
        hidden_states = hidden_states[:len(prices)]

    # Create the plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(prices.index, prices, c=hidden_states, cmap='viridis', marker='o')
    
    # Add the legend with the mapped state labels
    plt.legend(handles=scatter.legend_elements()[0], labels=state_label, title="Market States")
    
    # Set plot title and labels
    plt.title('Hidden States over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def get_state_labels_from_model(model):
    means = model.means_.flatten()  # Mean of each component (state)

    # Print the means of each state
    print("Means of the hidden states:")
    print(means)

    # Assign labels based on the relative order (min, middle, max)
    state_labels = ['Bull (Stocks Rising)', 'Neutral', 'Bear (Stocks Falling)']

    # Initialize the state mapping dictionary
    state_mapping = {}

    # Get the indices that sort the means array (ascending)
    sorted_indices = np.argsort(means)

    # Map the sorted indices to the state labels (min -> Bear, middle -> Neutral, max -> Bull)
    state_mapping[sorted_indices[0]] = state_labels[2]  # Min -> Bear
    state_mapping[sorted_indices[1]] = state_labels[1]  # Middle -> Neutral
    state_mapping[sorted_indices[2]] = state_labels[0]  # Max -> Bull

    # Print the mapping of components to market states
    print("State Mapping (component -> market state):")
    print(state_mapping)

    # Ensure the state labels correspond to the sorted indices (but keeping original index intact)
    state_actual_labels = [state_mapping[i] for i in range(len(means))]
    print("State Actual Labels:")
    print(state_actual_labels)

    return state_labels

def predict_future_state(model, last_data_point):
    last_return_reshaped = np.array([last_data_point]).reshape(-1, 1)
    future_state = model.predict(last_return_reshaped)
    return future_state[0]

def calculate_period_price_change(data):
    data = data.reset_index()

    results = []
    start_index = 0

    for i in range(1, len(data)):
        if data.loc[i, "Market_State"] != data.loc[start_index, "Market_State"]:
            # Check if the period has fewer than 2 data points
            if i - start_index < 3:
                start_index = i
                continue  # Skip this period if it has fewer than 2 data points

            start_date = data.loc[start_index, "Date"]  
            end_date = data.loc[i - 1, "Date"]
            start_price = data.loc[start_index, "SPY"]
            end_price = data.loc[i - 1, "SPY"]
            price_change = end_price - start_price
            percentage_change = (price_change / start_price) * 100

            results.append({
                "Market_State": data.loc[start_index, "Market_State"],
                "start_date": start_date,
                "end_date": end_date,
                "start_price": start_price,
                "end_price": end_price,
                "price_change": price_change,
                "percentage_change": percentage_change
            })

            start_index = i

    # Handle the last period
    if len(data) - start_index >= 2:  # Only add the last period if it has at least 2 data points
        start_date = data.loc[start_index, "Date"]
        end_date = data.loc[len(data) - 1, "Date"]
        start_price = data.loc[start_index, "SPY"]
        end_price = data.loc[len(data) - 1, "SPY"]
        price_change = end_price - start_price
        percentage_change = (price_change / start_price) * 100

        results.append({
            "Market_State": data.loc[start_index, "Market_State"],
            "start_date": start_date,
            "end_date": end_date,
            "start_price": start_price,
            "end_price": end_price,
            "price_change": price_change,
            "percentage_change": percentage_change
        })

    return pd.DataFrame(results)


def plot_results_with_period_lines(prices, hidden_states, state_label, period_changes):
    if len(prices) > len(hidden_states):
        prices = prices[:len(hidden_states)]
    elif len(hidden_states) > len(prices):
        hidden_states = hidden_states[:len(prices)]

    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(prices.index, prices, c=hidden_states, cmap='viridis', marker='o', label='Price')

    plt.legend(handles=scatter.legend_elements()[0], labels=state_label, title="Market States")

    for _, row in period_changes.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        start_price = row['start_price']
        end_price = row['end_price']
        
        plt.plot([start_date, end_date], [start_price, end_price], color='red', linewidth=2)

    plt.title('Hidden States with Period Lines')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


# Define parameters
ticker = 'SPY'
stock_data_train, stock_data_test = train_test_split(ticker)

# Compute returns
returns_train = calculate_returns(stock_data_train)
returns_test = calculate_returns(stock_data_test)

hmm_model_SPY = train_hmm_model(returns_train, random_state=10)

hidden_states_train = predict_states(hmm_model_SPY, returns_train)

state_labels_SPY = get_state_labels_from_model(hmm_model_SPY)

plot_results(stock_data_train, hidden_states_train, state_labels_SPY)


# Predict the future state for training data
last_return_train = returns_train.iloc[-1]
future_state_train = predict_future_state(hmm_model_SPY, last_return_train)

print(f"The predicted market state for the next period based on training data is: {state_labels_SPY[future_state_train]}")

# Predict hidden states for testing data
hidden_states_test = predict_states(hmm_model_SPY, returns_test)
stock_data_test = stock_data_test.iloc[-len(hidden_states_test):]

# Visualize results for test data
plot_results(stock_data_test, hidden_states_test, state_labels_SPY)


# Predict the future state for test data
last_return_test = returns_test.iloc[-1]
future_state_test = predict_future_state(hmm_model_SPY, last_return_test)

print(f"The predicted market state for this next period is: {state_labels_SPY[future_state_test]}")


hidden_states_df = pd.DataFrame(hidden_states_test, index=stock_data_test.index, columns=["Hidden_State"])

merged = pd.concat([stock_data_test, hidden_states_df], axis=1)

merged["Market_State"] = merged["Hidden_State"].map(lambda x: state_labels_SPY[x])


period_changes = calculate_period_price_change(merged)

print(period_changes)

plot_results_with_period_lines(stock_data_test, hidden_states_test, state_labels_SPY, period_changes)


# Ensure lengths match by trimming stock_data_train
stock_data_train_trimmed = stock_data_train.iloc[-len(hidden_states_train):]

# Create a DataFrame for hidden states using training data
hidden_states_df_train = pd.DataFrame(hidden_states_train, index=stock_data_train_trimmed.index, columns=["Hidden_State"])

# Merge the stock data and hidden states for training data
merged_train = pd.concat([stock_data_train_trimmed, hidden_states_df_train], axis=1)

# Map the numerical states to human-readable labels for training data
merged_train["Market_State"] = merged_train["Hidden_State"].map(lambda x: state_labels_SPY[x])

# Calculate the price change for each period in the training data
period_changes_train = calculate_period_price_change(merged_train)

# Print the period changes for training data
print("Period Changes for Training Data:")
print(period_changes_train)

# Plot the results with period lines for training data
plot_results_with_period_lines(stock_data_train_trimmed, hidden_states_train, state_labels_SPY, period_changes_train)

plt.figure(figsize=(15, 6))

# Map market states to colors
colors = period_changes_train["Market_State"].map({
    "Bull (Stocks Rising)": "green",
    "Bear (Stocks Falling)": "red",
    "Neutral": "blue"
})

# Map market states to colors
state_color_map = {
    "Bull (Stocks Rising)": "green",
    "Bear (Stocks Falling)": "red",
    "Neutral": "blue"
}
colors = period_changes_train["Market_State"].map(state_color_map)

# Create a bar chart
plt.figure(figsize=(15, 8))
plt.bar(period_changes_train.index, period_changes_train["percentage_change"], color=colors, alpha=0.7)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal line at 0%

# Add a legend
legend_labels = list(state_color_map.keys())
legend_colors = [state_color_map[label] for label in legend_labels]
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors],
           labels=legend_labels, title="Market States")

# Set titles and labels
plt.title("Percentage Change per Market Period")
plt.xlabel("Periods")
plt.ylabel("Percentage Change")

plt.show()