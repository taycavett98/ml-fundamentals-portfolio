# want to implement a simple linear regression model

# state our problem
# who : F1 team - want to optimize cars
# what : one number predicts another, in this case we want to. look at engine horsepower to predict lap time (in seconds)
# why  : easy to visualize, we can understand the relationship between horsepower and lap time
# problem satement: "if i increase my enginer power by 50HP, how much faster will my car go per lap?"

# make some fake data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('supervised_learning/linear-regression/f1_data.csv')
print(data.head())
x = data['engine_power_hp'].values
y = data['lap_time_seconds'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# reshape the data
# x_train = x_train.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)

# y = mx+b
m = np.sum((x_train - np.mean(x_train)) * (y_train - np.mean(y_train))) / np.sum((x_train - np.mean(x_train)) ** 2)
b = np.mean(y_train) - m * np.mean(x_train)

print(f"Model: lap_time = {m:.4f} * engine_power_hp + {b:.4f}")

y_pred = m * x_test + b

print("Predicted lap times:", y_pred)

# evaluate our model

