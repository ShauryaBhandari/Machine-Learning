import numpy as np
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T
print("Input for XOR gate")
print(x)
print("\nDesirable output")
print(y)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidgrad(z):
    x = sigmoid(z)
    return x * (1 - x)


p = x.shape[0]  # training examples
q = x.shape[1]  # features/input
hiddenlayers = 2  # hidden layer size
learningrate = 1  # learning rate for backpropagation

# Initialising weights for our model
theta1 = (np.random.random((q + 1, hiddenlayers)))
theta2 = (np.random.random((hiddenlayers + 1, 1)))


def forward_prop(x, theta1, theta2):
    # adding bias column to input
    a1 = np.c_[np.ones(x.shape[0]), x]
    # weights * input
    z1 = a1.dot(theta1)
    # the input of the second layer is the output of the first layer, passed through the activation function and column of biases is added
    a2 = np.c_[np.ones(x.shape[0]), sigmoid(z1)]
    # weinghts * input of layer 2
    z3 = a2.dot(theta2)
    h3 = sigmoid(z3)
    return a1, z1, a2, z3, h3


for i in range(1500):
    a1, z1, a2, z3, hyp = forward_prop(x, theta1, theta2)
    del_2 = y - hyp
    # the error of the previous layer is found by computing the dot product of the error of the previous layer and the weights of the second layer,without the column for biases.
    # this matrix is made to undergo element-wise multiplication with the output of the first layer(taking into account the activation function)
    del_1 = del_2.dot(theta2[1:, :].T)

    # the error of the second layer is multiplied element wise by the sigmoid gradient of the output of the second layer
    delta2 = del_2

    # the error of the first layer is multiplied element wise by the sigmoid gradient of the output of the first layer
    delta1 = del_1 * sigmoidgrad(z1)

    # the parameters are updated using gradient descent
    theta2 += learningrate * a2.T.dot(delta2)
    theta1 += learningrate * a1.T.dot(delta1)

a1, z1, a2, z3, hyp = forward_prop(x, theta1, theta2)
print("\nPredicted Output")
print(hyp)
