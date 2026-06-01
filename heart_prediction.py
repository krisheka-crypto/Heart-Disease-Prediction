import pandas as pd

df = pd.read_csv("heart.csv")

X = df.drop("condition", axis=1)
y = df["condition"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

import joblib

joblib.dump(model, "heart_model.pkl")
print("Model saved!")