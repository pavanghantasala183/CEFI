from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_synthetic(true_importance, estimated_importance):
    correlation, _ = spearmanr(true_importance, estimated_importance)
    mae = np.mean(np.abs(true_importance - estimated_importance))
    return correlation, mae

def evaluate_clustering(X, feature_importance, n_clusters):
    top_features = np.argsort(feature_importance)[::-1][:5]
    X_top = X[:, top_features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_top)
    silhouette = silhouette_score(X_top, labels)
    calinski = calinski_harabasz_score(X_top, labels)
    return silhouette, calinski