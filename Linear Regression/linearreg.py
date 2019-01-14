from numpy import *
import matplotlib.pyplot as plt


def cost_function(b, m, points):
    cost = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        cost = cost + (y - (m * x - b))**2
        return cost / 2 * float(len(points))


def step_gradient(current_b, current_m, points, learning_rate):
    # actual gradient descent; most imp fn; constant fn
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient = b_gradient - (2 / N) * (y - (current_m * x) + current_b)
        m_gradient = m_gradient - (2 / N) * x * \
            (y - (current_m * x) + current_b)
    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)
    return[new_b, new_m]


def gradient_desc_runner(points, starting_b, starting_m, learning_rate, numof_iterations):
    b = starting_b
    m = starting_m
    for i in range(numof_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        return [b, m]


X = 0
y = 0
numof_iterations = 1000


def run():
    points = loadtxt(open("data.csv", "rb"), delimiter=",")
    X = c_[ones(points.shape[0]), points[:, 0]]
    y = c_[points[:, 1]]
    learning_rate = 0.0001  # hyperparameters; value is set beforehand
    # y = mx + c
    initial_b = 0
    initial_m = 0
    numof_iterations = 1000
    [b, m] = gradient_desc_runner(
        points, initial_b, initial_m, learning_rate, numof_iterations)
    print(b)
    print(m)


if __name__ == '__main__':
    run()
    points = loadtxt(open("data.csv", "rb"), delimiter=",")
    X = c_[ones(points.shape[0]), points[:, 0]]
    y = c_[points[:, 1]]
    plt.scatter(X[:, 1], y)
    plt.xlabel("Marks obtained")
    plt.ylabel("Hours studied")
    plt.show()
