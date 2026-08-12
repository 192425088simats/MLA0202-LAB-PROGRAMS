# Name: Hemakshitha
# Program 16: Comparison of Classification Algorithms

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

print("Name: Hemakshitha")
print("Comparison of Classification Algorithms")
print("-" * 50)

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling for Logistic Regression and KNN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

print("\nAlgorithm Performance:")
print("-" * 50)

# Logistic Regression
models["Logistic Regression"].fit(X_train_scaled, y_train)
prediction = models["Logistic Regression"].predict(X_test_scaled)
accuracy = accuracy_score(y_test, prediction)

print("Logistic Regression :", round(accuracy * 100, 2), "%")

# KNN
models["KNN"].fit(X_train_scaled, y_train)
prediction = models["KNN"].predict(X_test_scaled)
accuracy = accuracy_score(y_test, prediction)

print("KNN                 :", round(accuracy * 100, 2), "%")

# Decision Tree
models["Decision Tree"].fit(X_train, y_train)
prediction = models["Decision Tree"].predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Decision Tree       :", round(accuracy * 100, 2), "%")

# Naive Bayes
models["Naive Bayes"].fit(X_train, y_train)
prediction = models["Naive Bayes"].predict(X_test)
accuracy = accuracy_score(y_test, prediction)

print("Naive Bayes         :", round(accuracy * 100, 2), "%")

print("\nComparison completed successfully.")
