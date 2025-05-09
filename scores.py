import numpy as np
from sklearn.metrics import silhouette_score

def sil_score_func(data, labels):
    """
    Silhouette Score without noise points.
    """
    cluster = labels != -1  # -1 are noise points
    cluster_data = data[cluster]
    cluster_labels = labels[cluster]

    if len(np.unique(cluster_labels)) > 1:  #we need at least two clusters
        score = silhouette_score(cluster_data, cluster_labels)
        return score
    else:
        return None