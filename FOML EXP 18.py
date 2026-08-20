from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

print("Name: Hemakshitha")
print("Perceptron based IRIS classification")
print("-------------------------------------")

iris = load_iris()

X = iris.data
y = iris.target

print("\nINPUT:")
print("Iris Dataset")
print("Total Samples:", len(X))

print("\nFeatures:")
print("Sepal Length")
print("Sepal Width")
print("Petal Length")
print("Petal Width")

print("\nSample Input:")
print("5.1  3.5  1.4  0.2")
print("6.2  3.4  5.4  2.3")
print("5.9  3.0  5.1  1.8")

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

print("\nOUTPUT:")
print("Actual Classes:")
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

print("\nNew Iris Flower Input:")
print("Sepal Length = 5.1 cm")
print("Sepal Width  = 3.5 cm")
print("Petal Length = 1.4 cm")
print("Petal Width  = 0.2 cm")

print("\nPredicted Class:")
print(iris.target_names[prediction[0]])
