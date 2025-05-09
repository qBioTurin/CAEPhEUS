'''
sys.argv[1] data
sys.argv[2] min_cluster_size parameter value
sys.argv[3] cluster_selection_method parameter value
sys.argv[4] min_samples parameter value
sys.argv[5] original data for the later interpretation
'''
import pandas as pd
import hdbscan
import sys
import numpy as np

sys.path.append('...') 
import data_for_dim_red 

sys.path.append('...') 
import scores

#data
path_data = sys.argv[1]
data = pd.read_csv(filepath_or_buffer=path_data)

#HDBSCAN clustering
min_cluster_size_param = sys.argv[2]
cluster_selection_method_param = sys.argv[3]
min_samples_param = sys.argv[4]
clustering = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size_param), cluster_selection_method=cluster_selection_method_param, min_samples = int(min_samples_param))
clustering.fit(data)

#clustering labels
labels = clustering.labels_

#Silhouette scores
silhouette = scores.sil_score_func(data, labels)

#number of clusters
unique_labels = np.unique(labels)
n_clusters = len(unique_labels[unique_labels != -1])

with open("metriche_clusters.txt", "w") as my_file:
    my_file.write(f"silhouette {silhouette} \n")
    my_file.write(f"number of cluster {n_clusters} \n")


path_data_no_dim_red  = sys.argv[5]
X_scaled, X_split = data_for_dim_red.data_transformation(path_data_no_dim_red )
X_split['Cluster']= labels

filtered_df = X_split[['MRN', 'Cluster']]
filtered_df.to_csv(f'cluster_finali_HDBSCAN-{min_cluster_size_param}-{cluster_selection_method_param}.csv', index=False)