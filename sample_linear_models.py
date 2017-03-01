"""
Let’s look at an example of the linear model in a classification context.
"""
import matplotlib.pyplot as plt

from LinearModels import *

# Parameters
n_classes = 2
plot_colors = "rby"
plot_step = 0.02

# Load data
data = np.genfromtxt('data.csv', delimiter=';')

# Models
clf = [LeastSquaresClassification(),
       NearestNeighborClassification(1),
       NearestNeighborClassification(15),
]
name_clf = [
    'LeastSquares',
    'kNN (k=1)',
    'kNN (k=15)',
]

# Plot
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3]]):

    # We only take the two corresponding features
    X = data[:, 0:2]
    y = data[:, 2]

    # Train
    clf[pairidx].fit(X, y)

    # Plot the decision boundary
    plt.subplot(1, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf[pairidx].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(name_clf[pairidx])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label="class %d" % i,
                    cmap=plt.cm.Paired, alpha=0.8)

plt.suptitle("Classification")
plt.legend()
plt.show()