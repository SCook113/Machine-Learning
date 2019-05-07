import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
iris = datasets.load_iris()
X = iris
X = iris["data"][:, (2, 3)]

############################
# Show data
############################
# plt.plot(X[:,0], X[:,1], "o", label="SGD")
# plt.show()

y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
# Linear classification
svm_clf = Pipeline((("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge")),))
svm_clf.fit(X, y)
print("yes" if svm_clf.predict([[5.5, 1.7]]) == 1.0 else "no")

# Add a pypeline using polynomial features
polynomial_svm_clf = Pipeline((("poly_features", PolynomialFeatures(degree=3)), ("scaler", StandardScaler()),
                               ("svm_clf", LinearSVC(C=10, loss="hinge"))))
polynomial_svm_clf.fit(X, y)
print("yes" if polynomial_svm_clf.predict([[5.5, 1.7]]) == 1.0 else "no")

poly_kernel_svm_clf = Pipeline((("scaler", StandardScaler()), ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))))
poly_kernel_svm_clf.fit(X, y)

# It is also possible to handle regression tasks
# with a SVM. Let's try:
# Linear / SVR is the regression equivalent of the SVC class
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)

############################
# Experiment with PCA
############################
from sklearn.decomposition import PCA

# Plot 2D and 3D against each other
copy = datasets.load_iris()
X = copy["data"]
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot3D(X[:, 0], X[:, 1], X[:, 2], 'o')
ax2 = fig.add_subplot(122)
ax2.plot(X2D[:, 0], X2D[:, 1], 'o')
plt.show()

# Find the dimension in which 95% of the
# variance is preserved
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
d = np.argmax(cumsum >= 0.95) + 1
# OR
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(d)

############################
# Use grid search to find best kernel and gamma value
# for PCA
############################
clf = Pipeline([("kpca", KernelPCA(n_components=2)), ("log_reg", LogisticRegression())])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
