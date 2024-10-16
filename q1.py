import pandas as pd
import numpy as np
df= pd.read_csv('ml1\CarPrice_Assignment.csv')
df=df.drop(['car_ID',	'symboling',	'CarName',	'fueltype'	,'aspiration'	,'doornumber'	,'carbody'	,'drivewheel',	'enginelocation','enginetype',	'cylindernumber','fuelsystem'],axis=1)

print(df)
X = df.iloc[:,:].values.reshape(-1, 1)
y = df['price'].values


def linear_regression(X, y, learning_rate=0.00000001, iterations=1000) :
    w= 0  
    b = 0  
    n = float(len(X)) 

    for i in range(iterations):
        y_pred = w* X + b  
        D_w = (-2/n) * sum(X * (y - y_pred)) 
        D_b = (-2/n) * sum(y - y_pred)       
        w = w - learning_rate * D_w 
        b = b - learning_rate * D_b  

    return w, b


w, b = linear_regression(X, y)
print(f"Optimal slope (w): {np.round(w,2)}")
print(f"Optimal intercept (b): {np.round(b,2)}")


y_pred = w * X + b

print(np.round(y_pred,-1))
