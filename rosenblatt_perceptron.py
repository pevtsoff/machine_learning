import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Rosenblatt perceptron uses quantizer function to update the weights in ML equation"""



class Perceptron(object):
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
        self.w_ = [30,20,11]

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
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = self.w_ if any(self.w_) else rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        print(f'before update self.w_={self.w_}')

        for _ in range(self.n_iter):
            errors = 0
            for x_row, target in zip(X, y):
                update = self.eta * (target - self.classify(x_row))
                self.w_[1:] += update * x_row
                self.w_[0] += update
                errors += int(update != 0.0)

            print(f'on iteration{_} the self.w_={self.w_}')
            self.errors_.append(errors)

        print(f'after update self.w_={self.w_}')
        return self

    def net_input(self, x_row):
        """Calculate net input"""
        return np.dot(x_row, self.w_[1:]) + self.w_[0]

    def classify(self, x_row):
        """Return class label after unit step"""
        return np.where(self.net_input(x_row) >= 0.0, 1, -1)



df = pd.read_csv('./iris.data', header=None)
# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
print(f"X.shape={ X.shape}")

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# training perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
print(f'errors={ppn.errors_}')
plt.xlabel('Iteration number')
plt.ylabel('Number of wrong classifications')
plt.tight_layout()
plt.show()
print(f'perceptron weights after the first cycle={ppn.w_}')

# Real run on the same data_set
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
print(f'errors={ppn.errors_}')
plt.xlabel('Iteration number')
plt.ylabel('Number of wrong classifications in real run')
plt.tight_layout()
plt.show()