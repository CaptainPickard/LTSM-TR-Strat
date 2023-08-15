import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample price data for short-term and long-term trends
# Replace this with your actual data
time_short = [1, 2, 3, 4, 5]
price_short = [10, 12, 14, 16, 18]

time_long = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
price_long = [8, 10, 11, 14, 16, 18, 20, 22, 25, 28]

# Create linear regression models
model_short = LinearRegression()
model_long = LinearRegression()

# Reshape the data to fit the model's requirements
time_short = np.array(time_short).reshape(-1, 1)
time_long = np.array(time_long).reshape(-1, 1)

# Fit the models
model_short.fit(time_short, price_short)
model_long.fit(time_long, price_long)

# Predictions for short-term and long-term trends
price_pred_short = model_short.predict(time_short)
price_pred_long = model_long.predict(time_long)

# Create a figure and axis
plt.figure(figsize=(10, 6))
plt.title("Forex Trading Pair Price Trends")
plt.xlabel("Time")
plt.ylabel("Price")

# Plot the actual data and regression lines
plt.scatter(time_short, price_short, color='blue', label='Short-term Data')
plt.scatter(time_long, price_long, color='orange', label='Long-term Data')
plt.plot(time_short, price_pred_short, color='blue', linestyle='dashed', label='Short-term Trend')
plt.plot(time_long, price_pred_long, color='orange', linestyle='dashed', label='Long-term Trend')

# Add legend
plt.legend()

# Show the plot
plt.show()