import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

np.random.seed(42)

n = 500
rooms = np.random.randint(1, 6, n)
size = np.random.randint(500, 3000, n)
age = np.random.randint(1, 50, n)

price = rooms * 50000 + size * 30 - age * 1000 + np.random.randint(10000, 50000, n)

df = pd.DataFrame({
    "rooms": rooms,
    "size": size,
    "age": age,
    "price": price
})

X = df[["rooms", "size", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")