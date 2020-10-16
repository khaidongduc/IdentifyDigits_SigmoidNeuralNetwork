from keras.datasets import mnist
from modules import neural_network
import numpy as np
import pickle
from matplotlib import pyplot as plt


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
show_predictions = False
for i in range(num_test):
    num_right += predictions[i] == y_test[i]
    if show_predictions:
        img = X_test_raw[i]
        plt.imshow(img, cmap="gray")
        plt.title("expected: {}, actual: {}".format(y_test[i], predictions[i]))
        plt.show()

print("Accuracy: {}%".format((num_right / num_test) * 100.0))
