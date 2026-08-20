import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Name: Hemakshitha")
print("Future Sales Prediction")
print("----------------------")

data = {
    "Month": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    ],

    "Sales": [
        120, 135, 150, 165, 180,
        195, 210, 225, 240, 255,
        270, 285, 300, 315, 330,
        345, 360, 375, 390, 405
    ]
}

df = pd.DataFrame(data)

print("\nINPUT:")
print(df)

X = df[["Month"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nOUTPUT:")

print("Actual Sales:")
print(list(y_test))

print("\nPredicted Sales:")
print([round(x, 2) for x in y_pred])

print("\nMean Squared Error:")
print(round(mse, 2))

print("\nR2 Score:")
print(round(r2, 2))

future_months = pd.DataFrame({
    "Month": [21, 22, 23, 24, 25]
})

future_sales = model.predict(future_months)

print("\nFuture Sales Input:")
print("Months: 21, 22, 23, 24, 25")

print("\nFuture Sales Prediction:")

for month, sales in zip(
    future_months["Month"],
    future_sales
):
    print(
        "Month", month,
        "-> Predicted Sales =", round(sales, 2)
    )
