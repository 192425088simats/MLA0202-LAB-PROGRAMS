# Name: Hemakshitha
# Program 11: Credit Score Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Name: Hemakshitha")
print("Credit Score Classification")
print("-" * 40)

# Create dataset
data = {
    "Income": [25000, 45000, 60000, 30000, 80000, 55000, 20000, 90000,
               70000, 35000, 50000, 65000, 28000, 75000, 40000],
    
    "LoanAmount": [15000, 20000, 10000, 25000, 15000, 12000, 18000, 10000,
                   14000, 22000, 16000, 11000, 24000, 13000, 19000],
    
    "CreditHistory": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0,
                      1, 1, 0, 1, 0],
    
    "Age": [25, 35, 45, 28, 50, 40, 23, 52, 46, 30,
            38, 43, 27, 48, 32],
    
    "CreditScore": [
        "Good", "Good", "Good", "Poor", "Good",
        "Good", "Poor", "Good", "Good", "Poor",
        "Good", "Good", "Poor", "Good", "Poor"
    ]
}

df = pd.DataFrame(data)

# Convert Good/Poor into numbers
encoder = LabelEncoder()
df["CreditScore"] = encoder.fit_transform(df["CreditScore"])

# Features and target
X = df[["Income", "LoanAmount", "CreditHistory", "Age"]]
y = df["CreditScore"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# New customer
new_customer = pd.DataFrame(
    [[60000, 12000, 1, 40]],
    columns=["Income", "LoanAmount", "CreditHistory", "Age"]
)

# Predict
prediction = model.predict(new_customer)

result = encoder.inverse_transform(prediction)

print("\nNew Customer Details:")
print("Income: ₹60,000")
print("Loan Amount: ₹12,000")
print("Credit History: Good")
print("Age: 40")

print("\nPredicted Credit Score:", result[0])
