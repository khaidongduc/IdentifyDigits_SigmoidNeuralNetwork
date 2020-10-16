import numpy as np
from numpy import exp, log


def sigmoid(z):
    """
    calculate sigmoid(z) = 1/(1 + e^(-z))
    :param z: the variable
    :return: the ans
    """
    return 1.0 / (1 + exp(-z))


def sigmoid_grad(z):
    """
    calculate the gradient of sigmoid(z)
    :param z: the variable
    :return: the ans
    """
    return sigmoid(z) * (1 - sigmoid(z))


def predict(theta1, theta2, data):
    """
    predict the answer given the weights
    :param theta1: the weight of the first layer
    :param theta2: the weight of the second layer
    :param data: the data we want to predict
    :return: the prediction
    """
    num_data = len(data)

    z1 = np.insert(data, 0, np.ones((1, num_data)), axis=1)
    h1 = sigmoid(np.dot(z1, theta1.T))

    z2 = np.insert(h1, 0, np.ones((1, num_data)), axis=1)
    h2 = sigmoid(np.dot(z2, theta2.T))

    return np.argmax(h2, axis=1)


def rand_initial_weights(L_in, L_out, epsilon):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections
    :param L_in: number of incoming connections
    :param L_out: number of outgoing connections
    :param epsilon: margin of error
    :return: a matrix where its elements is randomly initialized between (-epsilon, epsilon)
    Note that the ans should be set to a matrix of size(L_out, 1 + L_in) as
      the first column of W handles the "bias" terms
    """
    return np.random.random((L_out, L_in + 1)) * 2 * epsilon - epsilon


def nn_cost_function(nn_params,
                     input_layer_size, hidden_layer_size,
                     num_labels,
                     X, y, _lambda):
    """
    calculate the cost of the function with given variables
    :param nn_params: the weights of the neural network rolled out into a list
    :param input_layer_size: the size of the input layer
    :param hidden_layer_size: the size of the hidden layer
    :param num_labels: the number of available labels
    :param X: given features (data)
    :param y: given labels (data)
    :param _lambda: the regularization term
    :return: the cost of the neural network
    """
    # extract data
    num_data = len(X)
    theta1, theta2 = translate_nn_params(nn_params,
                                         input_layer_size, hidden_layer_size,
                                         num_labels)
    # process raw data
    X = np.insert(np.array(X), 0, np.ones(num_data), axis=1)
    temp = y
    y = np.zeros((num_labels, num_data))
    for i, label in enumerate(temp):
        y[label][i] = 1

    # forward propagation
    a1 = X
    a2 = np.insert(sigmoid(np.dot(theta1, a1.T)).T, 0, np.ones(num_data), axis=1)
    z3 = sigmoid(np.dot(theta2, a2.T))
    h_theta = z3
    cost = (1 / num_data) * sum(sum((- y) * log(h_theta) - (1 - y) * log(1 - h_theta)))
    # regularization
    t1 = np.delete(theta1, 0, axis=1)
    t2 = np.delete(theta2, 0, axis=1)
    reg = _lambda * (sum(sum(t1 ** 2)) + sum(sum(t2 ** 2))) / (2 * num_data)
    cost = cost + reg
    return cost


def nn_gradient(nn_params,
                input_layer_size, hidden_layer_size,
                num_labels,
                X, y, _lambda):
    """
    calculate the gradient of the cost of the function with given variables
    :param nn_params: the weights of the neural network rolled out into a list
    :param input_layer_size: the size of the input layer
    :param hidden_layer_size: the size of the hidden layer
    :param num_labels: the number of available labels
    :param X: given features (data)
    :param y: given labels (data)
    :param _lambda: the regularization term
    :return: the gradient of the cost of the neural network
    :return:
    """
    # extract data
    num_data = len(X)
    theta1, theta2 = translate_nn_params(nn_params,
                                         input_layer_size, hidden_layer_size,
                                         num_labels)
    # process raw data
    X = np.insert(np.array(X), 0, np.ones(num_data), axis=1)
    temp = y
    y = np.zeros((num_labels, num_data))
    for i, label in enumerate(temp):
        y[label][i] = 1

    # start
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    for i in range(num_data):
        a1 = np.c_[X[i]].T
        z2 = np.dot(theta1, a1.T)

        a2 = np.insert(sigmoid(z2), 0, 1, axis=0)

        z3 = np.dot(theta2, a2)
        a3 = sigmoid(z3)

        # backward propagation
        delta_3 = a3 - np.c_[y[:, i]]
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias

        delta_2 = np.dot(theta2.T, delta_3) * sigmoid_grad(z2)
        delta_2 = np.delete(delta_2, 0, axis=0)

        theta1_grad = theta1_grad + delta_2.dot(a1)
        theta2_grad = theta2_grad + delta_3.dot(a2.T)
    # regularization
    theta1_grad[:, 0] = theta1_grad[:, 0] / num_data
    selector = [col for col in range(1, len(theta1_grad[0]))]
    theta1_grad[:, selector] = theta1_grad[:, selector] / num_data
    theta1_grad[:, selector] = theta1_grad[:, selector] + (_lambda / num_data) * theta1_grad[:, selector]

    theta2_grad[:, 0] = theta2_grad[:, 0] / num_data
    selector = [col for col in range(1, len(theta2_grad[0]))]
    theta2_grad[:, selector] = theta2_grad[:, selector] / num_data
    theta2_grad[:, selector] = theta2_grad[:, selector] + (_lambda / num_data) * theta2_grad[:, selector]

    # unroll gradients
    grad = np.append(theta1_grad.ravel(), (theta2_grad.ravel()))
    return grad


def translate_nn_params(nn_params,
                        input_layer_size, hidden_layer_size,
                        num_labels):
    """
    translate the list of coefficients into theta1 and theta2
    :param nn_params: the list
    :param input_layer_size: the input layer size
    :param hidden_layer_size: the hidden layer size
    :param num_labels: the number of available labels
    :return: theta1, theta2
    """
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))
    return theta1, theta2
