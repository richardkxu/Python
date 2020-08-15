"""
Implementing logistic regression for classification problem
Helpful resources:
Coursera ML course
https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


def cost_function(a, y):
    """
    hinge loss
    """
    return (-y * np.log(a) - (1 - y) * np.log(1 - a)).mean()


def forward(X, w, b):
    """
    forward pass
        If b is None, then no bias term.
    """
    z = np.dot(X, w)
    if b is not None:
        z += b
    a = sigmoid_function(z)
    return a


def backprop(alpha, w, b, a, y):
    dz = a - y
    dw = np.dot(X.T, dz) / y.size
    db = None
    w = w - alpha * dw  # updating the weights
    if b is not None:
        db = np.sum(dz) / y.size
        b = b - alpha * db
    return w, b


# here alpha is the learning rate, X is the feature matrix,y is the target matrix
def logistic_reg(alpha, X, y, bias=True, max_iterations=70000):
    """
    logistic regression trained with gradient descent
    """
    w = np.zeros(X.shape[1])  # without bias para
    b = None
    if bias:
        b = np.zeros(1)

    for iterations in range(max_iterations):
        a = forward(X, w, b)
        w, b = backprop(alpha, w, b, a, y)
        a = forward(X, w, b)
        J = cost_function(a, y)
        if iterations % 100 == 0:
            print(f"loss: {J}")  # printing the loss after every 100 iterations
    return w, b


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    print("shape of X: {}".format(X.shape))
    print("shape of y: {}".format(y.shape))

    alpha = 0.1
    w, b = logistic_reg(alpha, X, y, bias=True, max_iterations=70000)
    print("w: ", w)  # printing the w i.e our weights vector
    if b is not None:
        print("b: ", b)

    def predict_prob(X):
        return sigmoid_function(
            np.dot(X, w)
        )  # predicting the value of probability from the logistic regression algorithm

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="b", label="0")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="r", label="1")
    (x1_min, x1_max) = (X[:, 0].min(), X[:, 0].max())
    (x2_min, x2_max) = (X[:, 1].min(), X[:, 1].max())
    (xx1, xx2) = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = forward(grid, w, b).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors="black")

    plt.legend()
    plt.show()
