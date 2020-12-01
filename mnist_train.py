from keras.datasets import mnist
from modules import neural_network
import numpy as np
from scipy import optimize as op
import pickle

(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

# constant

input_layer_size = 784  # 28 x 28
hidden_layer_size = 25
num_labels = 10
epsilon = 0.12
_lambda = 1.0

# process raw data
num_data = len(X_train_raw)
X_train = np.zeros((num_data, input_layer_size))
for i in range(num_data):
    X_train[i] = X_train_raw[i].ravel()
X_train /= 255.0

# calc coefficients
cost_function = lambda theta: (neural_network.nn_cost_function(theta,
                                                               input_layer_size, hidden_layer_size,
                                                               num_labels,
                                                               X_train, y_train, _lambda))

gradient = lambda theta: (neural_network.nn_gradient(theta,
                                                     input_layer_size, hidden_layer_size,
                                                     num_labels,
                                                     X_train, y_train, _lambda))

initial_Theta1 = neural_network.rand_initial_weights(input_layer_size, hidden_layer_size, epsilon)
initial_Theta2 = neural_network.rand_initial_weights(hidden_layer_size, num_labels, epsilon)

initial_nn_params = np.append(initial_Theta1.ravel(), initial_Theta2.ravel())

print("Training...")
result = op.fmin_cg(cost_function, initial_nn_params, fprime=gradient)
res_theta1, res_theta2 = neural_network.translate_nn_params(result,
                                                            input_layer_size, hidden_layer_size,
                                                            num_labels)
print("Writing to file...")
filename = "trained_weights"
outfile = open(filename, "wb")
pickle.dump((res_theta1, res_theta2), outfile)
outfile.close()
print("Done.")

