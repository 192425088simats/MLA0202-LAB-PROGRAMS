# Name: Hemakshitha
# Program 15: Iris Flower Classification using Naive Bayes

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

print("Name: Hemakshitha")
print("Iris Flower Classification using Naive Bayes")
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

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# New flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict new flower
prediction = model.predict(new_flower)

print("\nNew Flower Details:")
print("Sepal Length : 5.1 cm")
print("Sepal Width  : 3.5 cm")
print("Petal Length : 1.4 cm")
print("Petal Width  : 0.2 cm")

print("\nPredicted Flower:", iris.target_names[prediction[0]])
