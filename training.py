import numpy as np
from modules import neural_network
import scipy.optimize as op
import pickle

infile = open("data/test_data", 'rb')
features_train, labels_train = pickle.load(infile)

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
epsilon = 0.12
_lambda = 1.0

# calc coefficients
cost_function = lambda theta: (neural_network.nn_cost_function(theta,
                                                               input_layer_size, hidden_layer_size,
                                                               num_labels,
                                                               features_train, labels_train, _lambda))

gradient = lambda theta: (neural_network.nn_gradient(theta,
                                                     input_layer_size, hidden_layer_size,
                                                     num_labels,
                                                     features_train, labels_train, _lambda))


initial_Theta1 = neural_network.rand_initial_weights(input_layer_size, hidden_layer_size, epsilon)
initial_Theta2 = neural_network.rand_initial_weights(hidden_layer_size, num_labels, epsilon)

initial_nn_params = np.append(initial_Theta1.ravel(), initial_Theta2.ravel())


print("Training...")
result = op.fmin_cg(cost_function, initial_nn_params, fprime=gradient)

res_theta1, res_theta2 = neural_network.translate_nn_params(result,
                                                            input_layer_size, hidden_layer_size,
                                                            num_labels)
print("Done.")
print("Writing to file...")
filename = "data/trained_weights"
outfile = open(filename, "wb")
pickle.dump((res_theta1, res_theta2), outfile)
outfile.close()
print("Done.")
