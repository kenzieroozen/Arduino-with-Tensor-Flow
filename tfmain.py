import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

normal_data = np.random.normal(loc=0.0, scale=0.5, size=(128, 3))

anomalous_data = np.random.normal(loc=3.0, scale=0.5, size=(50, 10))


train_data = normal_data[:800]
test_data = np.vstack([normal_data[800:], anomalous_data])

#normalize
train_data = train_data.astype("float32")
test_data = test_data.astype("float32")

#autoencoder
input_dim = train_data.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(3, activation="relu"),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="linear")
])

model.compile(optimizer="adam", loss="mse")

#normal data training
history = model.fit(train_data, train_data,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

# test phase
reconstructions = model.predict(test_data)
mse = np.mean(np.square(test_data - reconstructions), axis=1)

# threshold for anomaly
threshold = np.mean(history.history['loss']) + 3*np.std(history.history['loss'])

print("Reconstruction error threshold:", threshold)

#flagging anomaly
anomalies = mse > threshold
print("Detected anomalies:", np.sum(anomalies), "out of", len(test_data))

# error histogram
plt.hist(mse, bins=50)
plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
plt.legend()
plt.show()
