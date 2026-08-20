import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

print("Name: Hemakshitha")
print("Naive Bayes classification for Bank Loan prediction")
print("---------------------------------------------------")

data = {
    "Age": [
        25, 30, 35, 40, 45,
        50, 28, 32, 38, 42,
        48, 55, 27, 36, 44,
        52, 29, 34, 41, 47
    ],

    "Income": [
        25000, 30000, 40000, 50000, 60000,
        70000, 28000, 35000, 45000, 55000,
        65000, 80000, 27000, 42000, 52000,
        75000, 32000, 38000, 58000, 68000
    ],

    "LoanAmount": [
        100000, 120000, 150000, 180000, 200000,
        250000, 110000, 140000, 160000, 190000,
        220000, 300000, 100000, 155000, 185000,
        270000, 125000, 145000, 210000, 230000
    ],

    "CreditScore": [
        650, 680, 700, 720, 750,
        780, 660, 690, 710, 730,
        760, 800, 640, 705, 725,
        790, 670, 695, 740, 765
    ],

    "LoanApproved": [
        "No", "No", "Yes", "Yes", "Yes",
        "Yes", "No", "Yes", "Yes", "Yes",
        "Yes", "Yes", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "Yes", "Yes"
    ]
}

df = pd.DataFrame(data)

print("\nINPUT:")
print(df)

X = df[
    ["Age", "Income", "LoanAmount", "CreditScore"]
]

y = df["LoanApproved"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nOUTPUT:")
print("Actual Loan Status:")
print(y_test)

print("\nPredicted Loan Status:")
print(y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_customer = pd.DataFrame(
    [[35, 45000, 160000, 710]],
    columns=["Age", "Income", "LoanAmount", "CreditScore"]
)

prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print("\nNew Customer Input:")
print("Age = 35")
print("Income = Rs.45000")
print("Loan Amount = Rs.160000")
print("Credit Score = 710")

print("\nPredicted Loan Status:")
print(encoder.inverse_transform(prediction)[0])

print("\nPrediction Probability:")
print(round(max(probability[0]) * 100, 2), "%")
