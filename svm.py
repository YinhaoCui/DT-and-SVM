import numpy as np
from sklearn.utils import shuffle


class SVM():
    def __init__(self):
        self.regularization_strength = 10000
        self.learning_rate = 0.001
        self.W = 0

    #cost function, using to computer the hinger loss for each turn.
    def compute_cost(self, W, X, Y):
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, W))
        distances[distances < 0] = 0
        hinge_loss = self.regularization_strength * (np.sum(distances) / N)

        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])

        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))

        for ind, d in enumerate(distance):
            if max(0, d) == 0:
                di = W
            else:
                di = W - (self.regularization_strength * Y_batch[ind] * X_batch[ind])
            dw += di

        return dw / len(Y_batch)

    def fit(self, _X, _Y):
        max_epochs = 5000
        weights = np.zeros(_X.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.01

        for epoch in range(1, max_epochs):
            X, Y = shuffle(_X, _Y)
            for ind, x in enumerate(X):
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                weights = weights - (self.learning_rate * ascent)

            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.compute_cost(weights, _X, _Y)

                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    self.W = weights
                    return
                prev_cost = cost
                nth += 1
        self.W = weights

    def predict(self, X_test):
        result = np.array([])
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(X_test.to_numpy()[i], self.W))
            result = np.append(result, yp)
        return result
