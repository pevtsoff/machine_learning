import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Adaline perceptron use gradient descent method to update the weights for ML equation"""



class AdalinePerceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.w_ = np.zeros(1 + X.shape[1])

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        self.cost_ = []
        print(f'before update self.w_={self.w_}')

        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # the cost estimation
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        print(f'weights after update self.w_={self.w_}')

        return self

    def net_input(self, x_row):
        """Calculate net input"""
        return x_row.dot(self.w_[1:]) + self.w_[0]

    def classify(self, x_row):
        """Return class label after unit step"""
        return np.where(self.net_input(x_row) >= 0.0, 1, -1)




df = pd.read_csv('./iris.data', header=None)
# select setosa and versicolor
y = df.iloc[0:100, 4].values
# setting expected class marks
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# using standard deviation to standardize data
# This allows to run model on a high eta speed = 0.01
# instead of 0.0001 when using initial X data
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
print(f"X.shape={ X_std.shape}")

# plot data
plt.scatter(X_std[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X_std[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# training perceptron - high speed non-standard data
ppn = AdalinePerceptron(eta=0.01, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker='o')
print(f'errors={ppn.cost_}')
plt.xlabel('Iteration number')
plt.ylabel('Squared errors sum for eta=0.01')
plt.tight_layout()
plt.show()
print(f'perceptron weights after the first training={ppn.w_}')


# training perceptron - low speed and non-standard data (
# this run requires a lot of iterations
ppn = AdalinePerceptron(eta=0.00001, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker='o')
print(f'errors={ppn.cost_}')
plt.xlabel('Iteration number')
plt.ylabel('Squared errors sum for eta=0.00001')
plt.tight_layout()
plt.show()
print(f'perceptron weights after the second training={ppn.w_}')


# training perceptron - high speed standard data
# This run requires much less iterations
ppn = AdalinePerceptron(eta=0.01, n_iter=20)
ppn.fit(X_std, y)
plt.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker='o')
print(f'errors={ppn.cost_}')
plt.xlabel('Iteration number')
plt.ylabel('Squared errors sum for eta=0.01 and standardized data (training run)')
plt.tight_layout()
plt.show()
print(f'perceptron weights after the first training={ppn.w_}')


# real run
ppn.fit(X_std, y)
plt.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker='o')
print(f'errors={ppn.cost_}')
plt.xlabel('Iteration number')
plt.ylabel('Squared errors sum for eta=0.01 and standardized data (real run)')
plt.tight_layout()
plt.show()
print(f'perceptron weights after the real run training={ppn.w_}')


