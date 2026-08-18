
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

print("Perceptron Based IRIS Classification")
print("------------------------------------")

print("\nIris Dataset Loaded Successfully")
print("Number of Samples:", len(X))
print("Number of Features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nActual Classes:")
print(y_test)

print("\nPredicted Classes:")
print(y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\nNew Iris Flower:")
print("Sepal Length = 5.1 cm")
print("Sepal Width  = 3.5 cm")
print("Petal Length = 1.4 cm")
print("Petal Width  = 0.2 cm")

print("\nPredicted Iris Class:")
print(iris.target_names[prediction[0]])
