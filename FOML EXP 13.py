# Name: Hemakshitha
# Program 13: Car Price Prediction Model

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Name: Hemakshitha")
print("Car Price Prediction Model")
print("-" * 40)

# Car dataset
X = np.array([
    [2015, 80000, 1200],
    [2016, 70000, 1300],
    [2017, 60000, 1500],
    [2018, 50000, 1600],
    [2019, 40000, 1800],
    [2020, 30000, 1500],
    [2021, 20000, 1600],
    [2022, 10000, 2000],
    [2015, 90000, 1200],
    [2017, 65000, 1500],
    [2018, 55000, 1600],
    [2019, 45000, 1800],
    [2020, 35000, 1500],
    [2021, 25000, 1600],
    [2022, 15000, 2000]
])

# Car prices
y = np.array([
    350000,
    400000,
    500000,
    600000,
    750000,
    850000,
    1000000,
    1200000,
    320000,
    480000,
    580000,
    700000,
    820000,
    950000,
    1150000
])

# Split dataset
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

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# New car details
new_car = np.array([[2022, 12000, 1600]])

# Predict price
predicted_price = model.predict(new_car)

print("\nNew Car Details:")
print("Year       : 2022")
print("Kilometers : 12,000")
print("Engine     : 1600 cc")

print("\nPredicted Car Price: ₹", round(predicted_price[0], 2))
