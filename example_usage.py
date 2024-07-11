from cefi.core import cluster_enhanced_feature_importance
from cefi.evaluation import evaluate_synthetic, evaluate_clustering
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
importance = cluster_enhanced_feature_importance(X, n_clusters=3)
print("Feature Importance:", importance)