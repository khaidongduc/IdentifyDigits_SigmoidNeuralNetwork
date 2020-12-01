from keras.datasets import mnist
import numpy as np
import pickle
from matplotlib import pyplot as plt

from modules import neural_network

# extract data
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

filename = "trained_weights"
infile = open(filename, "rb")
theta1, theta2 = pickle.load(infile)
infile.close()

# constant
input_layer_size = 784  # 28 x 28

# process_raw_data
num_test = len(X_test_raw)
X_test = np.zeros((num_test, input_layer_size))
for i in range(num_test):
    X_test[i] = X_test_raw[i].ravel()
X_test /= 255.0

# evaluation

print("Evaluating...")

predictions = neural_network.predict(theta1, theta2, X_test)
num_right = 0
for i in range(num_test):
    num_right += predictions[i] == y_test[i]
print("Accuracy: {}%".format((num_right / num_test) * 100.0))

# print the first 25 predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    img = X_test_raw[i]
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap="gray")
    plt.xlabel("E: {}, A: {}".format(y_test[i], predictions[i]))
plt.show()