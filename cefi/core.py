import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

def cluster_enhanced_feature_importance(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    feature_importance = np.zeros(X.shape[1])

    for cluster in range(n_clusters):
        y_binary = (cluster_labels == cluster).astype(int)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y_binary)
        importance = rf.feature_importances_
        cluster_size = np.sum(y_binary)
        feature_importance += importance * cluster_size

    feature_importance /= np.sum(feature_importance)
    return feature_importance