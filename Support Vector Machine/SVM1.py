import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

# Creating 40 separable points
X, y = make_blobs(n_samples=500, centers=2, random_state=20)

clf = svm.SVC(kernel="linear", C=1)
clf.fit(X, y)

# Visualising data
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# plt.show()

# Giving it new data
newData = [[3, 4], [5, 6]]
print(clf.predict(newData))

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Evaluating model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
plt.show()
