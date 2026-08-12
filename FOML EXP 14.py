# Name: Hemakshitha
# Program 14: House Price Prediction

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Name: Hemakshitha")
print("House Price Prediction")
print("-" * 40)

# House dataset
# Features: Area, Bedrooms, Age
X = np.array([
    [1000, 2, 10],
    [1200, 2, 8],
    [1500, 3, 5],
    [1800, 3, 4],
    [2000, 4, 3],
    [2200, 4, 2],
    [2500, 4, 1],
    [2800, 5, 1],
    [1100, 2, 9],
    [1400, 3, 6],
    [1700, 3, 5],
    [1900, 4, 4],
    [2100, 4, 3],
    [2400, 4, 2],
    [2700, 5, 1]
])

# House prices
y = np.array([
    3000000,
    3500000,
    4500000,
    5500000,
    6500000,
    7200000,
    8500000,
    9500000,
    3200000,
    4200000,
    5000000,
    6000000,
    7000000,
    8200000,
    9200000
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# New house details
new_house = np.array([[2000, 3, 3]])

# Predict house price
predicted_price = model.predict(new_house)

print("\nNew House Details:")
print("Area     : 2000 sq.ft")
print("Bedrooms : 3")
print("Age      : 3 years")

print("\nPredicted House Price: ₹", round(predicted_price[0], 2))
