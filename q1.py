import pandas as pd
import numpy as np
df= pd.read_csv('ml1\CarPrice_Assignment.csv')
df=df.drop(['car_ID',	'symboling',	'CarName',	'fueltype'	,'aspiration'	,'doornumber'	,'carbody'	,'drivewheel',	'enginelocation','enginetype',	'cylindernumber','fuelsystem'],axis=1)

print(df)
X = df.iloc[:,:].values.reshape(-1, 1)
y = df['price'].values

# Linear regression using gradient descent
def linear_regression(X, y, learning_rate=0.00000001, iterations=1000) :
    w= 0  # Initialize slope
    b = 0  # Initialize intercept
    n = float(len(X))  # Number of data points

    for i in range(iterations):
        y_pred = w* X + b  # Predicted value of y
        D_w = (-2/n) * sum(X * (y - y_pred))  # Derivative of m
        D_b = (-2/n) * sum(y - y_pred)        # Derivative of c
        w = w - learning_rate * D_w  # Update m
        b = b - learning_rate * D_b  # Update c

    return w, b

# Train the model
w, b = linear_regression(X, y)
print(f"Optimal slope (w): {np.round(w,2)}")
print(f"Optimal intercept (b): {np.round(b,2)}")

# Make predictions
y_pred = w * X + b

print(np.round(y_pred,-1))
