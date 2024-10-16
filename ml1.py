import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = {'Size': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
        'Price': [250000, 260000, 270000, 280000, 290000, 300000, 310000, 320000, 330000, 340000]}

df = pd.DataFrame(data)

# Features and target
X = df['Size'].values.reshape(-1, 1)
y = df['Price'].values

# Linear regression using gradient descent
def linear_regression(X, y, learning_rate=0.00000001, iterations=1000):
    m = 0  # Initialize slope
    c = 0  # Initialize intercept
    n = float(len(X))  # Number of data points

    for i in range(iterations):
        y_pred = m * X + c  # Predicted value of y
        D_m = (-2/n) * sum(X * (y - y_pred))  # Derivative of m
        D_c = (-2/n) * sum(y - y_pred)        # Derivative of c
        m = m - learning_rate * D_m  # Update m
        c = c - learning_rate * D_c  # Update c

    return m, c

# Train the model
m, c = linear_regression(X, y)
print(f"Optimal slope (m): {m}")
print(f"Optimal intercept (c): {c}")

# Make predictions
y_pred = m * X + c

# Plot the original data and regression line
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Size (sq. ft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction using Linear Regression')
plt.legend()
plt.show()

# Evaluate the model
mse = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error: {mse}")

ss_tot = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"R-squared: {r2}")
