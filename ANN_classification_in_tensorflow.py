# -*- coding: utf-8 -*-
# ANN_classification_in_tensorflow.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
import random
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import fashion_mnist

print(tf.__version__)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in)

    if model.output_shape[-1] > 1:
        print("doing multiclass classification...")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# Generate data
X, y = make_circles(n_samples=1000, noise=0.03, random_state=42)
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})

# Model 1
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model_1.compile(loss="binary_crossentropy", optimizer="SGD", metrics=["accuracy"])
model_1.fit(X, y, epochs=5)
plot_decision_boundary(model_1, X, y)

# Model 2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])
model_2.compile(loss="binary_crossentropy", optimizer="SGD", metrics=["accuracy"])
model_2.fit(X, y, epochs=100, verbose=0)
model_2.evaluate(X, y)
plot_decision_boundary(model_2, X, y)

# Model 3 (deeper)
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
model_3.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_3.fit(X, y, epochs=100)
plot_decision_boundary(model_3, X, y)

# Model 6 - non-linearity
model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1)
])
model_6.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), metrics=["accuracy"])
history = model_6.fit(X, y, epochs=100)
model_6.evaluate(X, y)
plot_decision_boundary(model_6, X, y)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

# Model 7 - with sigmoid output
model_7 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model_7.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), metrics=["accuracy"])
history1 = model_7.fit(X, y, epochs=100)
model_7.evaluate(X, y)
pd.DataFrame(history1.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()
plot_decision_boundary(model_7, X, y)

# Train-test split
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Model 8 - evaluation
model_8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model_8.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
history = model_8.fit(X_train, y_train, epochs=20)
loss, accuracy = model_8.evaluate(X_test, y_test)
print(f"Model loss on the test set: {loss}")
print(f"Model accuracy on the test set: {100 * accuracy:.2f}%")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_8, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_8, X_test, y_test)
plt.show()

# Loss curves
pd.DataFrame(history.history).plot()
plt.title("Model_8 training curves")
plt.show()

# Fashion MNIST - Multiclass classification
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualization
plt.figure(figsize=(5, 5))
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)
plt.show()

# Model with raw data
model_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model_1.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
non_norm_history = model_1.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# Normalize data
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# Model with normalized data
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model_2.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
norm_history = model_2.fit(train_data_norm, train_labels, epochs=10, validation_data=(test_data_norm, test_labels))

# Compare curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
pd.DataFrame(non_norm_history.history).plot(ax=axes[0], title="Non-normalized Data")
pd.DataFrame(norm_history.history).plot(ax=axes[1], title="Normalized Data")
plt.tight_layout()
plt.show()

# Final model
model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model_3.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
history = model_3.fit(train_data, train_labels, epochs=20, validation_data=(test_data, test_labels))

# Predictions
y_probs = model_3.predict(test_data)
y_preds = y_probs.argmax(axis=1)

# Confusion matrix
print(confusion_matrix(y_true=test_labels, y_pred=y_preds))

# Model summary and structure
model_3.summary()
plot_model(model_3, show_shapes=True)
