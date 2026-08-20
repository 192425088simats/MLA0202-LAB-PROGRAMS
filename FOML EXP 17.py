import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("Name: Hemakshitha")
print("Mobile Price Prediction")
print("-----------------------")

data = {
    "RAM": [2, 3, 4, 4, 6, 8, 8, 12, 12, 16],
    "Storage": [32, 32, 64, 128, 128, 128, 256, 256, 512, 512],
    "Battery": [3000, 3500, 4000, 4500, 4500, 5000, 5000, 5500, 6000, 6000],
    "Camera": [8, 12, 16, 20, 24, 32, 48, 50, 64, 108],
    "Price": [8000, 10000, 13000, 16000, 20000, 25000, 30000, 40000, 55000, 70000]
}

df = pd.DataFrame(data)

print("\nINPUT:")
print(df)

X = df[["RAM", "Storage", "Battery", "Camera"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nOUTPUT:")
print("Actual Prices:")
print(list(y_test))

print("\nPredicted Prices:")
print([round(x, 2) for x in y_pred])

mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:")
print(round(mse, 2))

new_mobile = pd.DataFrame(
    [[8, 256, 5000, 50]],
    columns=["RAM", "Storage", "Battery", "Camera"]
)

prediction = model.predict(new_mobile)

print("\nNew Mobile Input:")
print("RAM = 8 GB")
print("Storage = 256 GB")
print("Battery = 5000 mAh")
print("Camera = 50 MP")

print("\nPredicted Mobile Price:")
print("Rs.", round(prediction[0], 2))
