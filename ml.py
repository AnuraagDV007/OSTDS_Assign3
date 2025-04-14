import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

model = LinearRegression()
model.fit(X, y)

print("Linear Regression Coefficients:")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)

# Generate synthetic data
X_multi = np.random.rand(100, 3)
y_multi = 3 + 2*X_multi[:, 0] + 4*X_multi[:, 1] + 5*X_multi[:, 2] + np.random.randn(100)

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("\nMultiple Regression Coefficients:")
print("Intercept:", model_multi.intercept_)
print("Coefficients:", model_multi.coef_)

poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

print("\nPolynomial Regression Coefficients:")
print(poly_model.coef_)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_multi, y_multi)

print("\nRidge Regression Coefficients:")
print("Intercept:", ridge_model.intercept_)
print("Coefficients:", ridge_model.coef_)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_multi, y_multi)

print("\nLasso Regression Coefficients:")
print("Intercept:", lasso_model.intercept_)
print("Coefficients:", lasso_model.coef_)

# Simulate high-dimensional data
X_pca = np.random.rand(100, 5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Optional: Plot
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("PCA - First 2 Principal Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# Generate synthetic data for clustering
from sklearn.datasets import make_blobs
X_blob, y_blob = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(X_blob)

# Plot the clusters
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()
