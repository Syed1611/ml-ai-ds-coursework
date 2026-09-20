import numpy as np

def relu(x):
  return (x > 0) * x


def relu2deriv(output):
  return output>0

x = np.array([
    [0.1, 0.2],
    [0.2, 0.3],
    [0.4, 0.1],
    [0.3, 0.1],
    [0.2, 0.2]
])

y = np.array([
    [0],
    [0],
    [1],
    [1],
    [1]
])

alpha = 0.1
hidden_size = 3

weights_0_1 = np.array([
    [1.0, -1.0, 0.5],
    [0.5,  1.0, -1.0]
])

weights_1_2 = np.array([
    [0.5],
    [1.0],
    [-0.5]
])

for iteration in range(3):
  print("epoch: ", iteration+1)
  layer_0 = x
  layer_1 = relu(np.dot(layer_0,weights_0_1))
  print("layer_1: ", layer_1)
  layer_2 = np.dot(layer_1,weights_1_2)
  print("layer_2: ", layer_2)
  layer_2_delta = (layer_2 - y)
  print("layer_2_delta: ", layer_2_delta)
  layer_1_delta = np.dot(layer_2_delta, weights_1_2.T)*relu2deriv(layer_1)
  print("layer_1_delta: ", layer_1_delta)
  weights_1_2 -= alpha * np.dot(layer_1.T, layer_2_delta)
  print("weights_1_2: ", weights_1_2)
  weights_0_1 -= alpha * np.dot(layer_0.T, layer_1_delta)
  print("weights_0_1: ", weights_0_1)
  print("\n\n")

import numpy as np

def relu(x):
  return (x > 0) * x


def relu2deriv(output):
  return output>0

x = np.array( [[ 1, 0, 1 ],
 [ 0, 1, 1 ],
 [ 0, 0, 1 ],
 [ 1, 1, 1 ] ] )

y = np.array([[ 1, 1, 0, 0]]).T

alpha = 0.1
hidden_size_1 = 4
hidden_size_2 = 3
weights_0_1 = np.random.random((3, hidden_size_1))
weights_1_2 = np.random.random((hidden_size_1, hidden_size_2))
weights_2_3 = np.random.random((hidden_size_2, 1))

for iteration in range(100):
  layer_3_error = 0
  for i in range(len(x)):
    layer_0 = x[i:i+1]
    layer_1 = relu(np.dot(layer_0,weights_0_1))
    layer_2 = relu(np.dot(layer_1,weights_1_2))
    layer_3 = np.dot(layer_2,weights_2_3)
    layer_3_error += np.sum((layer_3 - y[i:i+1]) ** 2)
    layer_3_delta = (layer_3 - y[i:i+1])
    layer_2_delta = np.dot(layer_3_delta, weights_2_3.T)*relu2deriv(layer_2)
    layer_1_delta = np.dot(layer_2_delta, weights_1_2.T)*relu2deriv(layer_1)
    weights_2_3 -= alpha * np.dot(layer_2.T, layer_3_delta)
    weights_1_2 -= alpha * np.dot(layer_1.T, layer_2_delta)
    weights_0_1 -= alpha * np.dot(layer_0.T, layer_1_delta)
    if(iteration % 10 == 9):
      print("Error:" + str(layer_3_error))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def relu(x):
  return (x > 0) * x


def relu2deriv(output):
  return output>0

df = pd.read_csv("winequality-total.csv", sep=";")

# print(df.head())

X = df.drop(columns=["quality"]).values
y = df["quality"].values.reshape(-1, 1)

num_samples, num_features = X.shape

print(num_features)

# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Standardize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# x = np.array( [[ 1, 0, 1 ],
#  [ 0, 1, 1 ],
#  [ 0, 0, 1 ],
#  [ 1, 1, 1 ] ] )

# y = np.array([[ 1, 1, 0, 0]]).T

alpha = 0.1
hidden_size_1 = 4
hidden_size_2 = 3
weights_0_1 = np.random.random((11, hidden_size_1))
weights_1_2 = np.random.random((hidden_size_1, hidden_size_2))
weights_2_3 = np.random.random((hidden_size_2, 1))

for iteration in range(100):
  layer_3_error = 0
  for i in range(len(X_train)):
    layer_0 = X_train[i:i+1]
    layer_1 = relu(np.dot(layer_0,weights_0_1))
    layer_2 = relu(np.dot(layer_1,weights_1_2))
    layer_3 = np.dot(layer_2,weights_2_3)
    layer_3_error += np.sum((layer_3 - y_train[i:i+1]) ** 2)
    layer_3_delta = (layer_3 - y_train[i:i+1])
    layer_2_delta = np.dot(layer_3_delta, weights_2_3.T)*relu2deriv(layer_2)
    layer_1_delta = np.dot(layer_2_delta, weights_1_2.T)*relu2deriv(layer_1)
    weights_2_3 -= alpha * np.dot(layer_2.T, layer_3_delta)
    weights_1_2 -= alpha * np.dot(layer_1.T, layer_2_delta)
    weights_0_1 -= alpha * np.dot(layer_0.T, layer_1_delta)
    if(iteration % 10 == 9):
      print("Error:" + str(layer_3_error))

