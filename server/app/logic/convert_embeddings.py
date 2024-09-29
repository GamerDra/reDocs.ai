# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score

# reshaped_embeddings_list = []

# def convert_embeddings(lst):
#     for embedding in lst:
#         reshaped_embedding = embedding.reshape(-1)
#         reshaped_embeddings_list.append(reshaped_embedding)

#     return reshaped_embeddings_list

# def clustering(lst):
#     silhouette_scores = []

#     for n_clusters in range(2, len(lst)):
#         clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', metric='cosine').fit(lst)
#         arr = clustering.labels_
#         silhouette_scores.append(silhouette_score(lst, arr, metric='cosine'))

#     max_cluster = np.argmax(silhouette_scores) + 2
#     clustering = AgglomerativeClustering(n_clusters=max_cluster, linkage='complete', metric='cosine').fit(lst)
#     arr = clustering.labels_
#     unique_values = np.unique(arr)
#     indices_list = []

#     for val in unique_values:
#         indices = np.where(arr == val)[0]
#         indices_list.append(indices)

#     return indices_list

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

reshaped_embeddings_list = []
def convert_embeddings(embedding_list):
    reshaped_embeddings_list = [embedding.reshape(-1) for embedding in embedding_list]
    return reshaped_embeddings_list


def clustering(list1, analytics,  plot_dendrogram=False, **kwargs):
    #truncate_mode="level", p=3
    clustering = AgglomerativeClustering(n_clusters=len(list1)//2+1, linkage='complete', metric='cosine', compute_distances=True).fit(list1)
    arr=clustering.labels_
    unique_values = np.unique(arr)
    indices_list = []
    for val in unique_values:
        indices = np.where(arr == val)[0]
        indices_list.append(indices)
    #for i in range(len(indices_list)):
        #print("Bro i called clustering: ",list(indices_list[i]))
    if plot_dendrogram:
        filenames = [file[0] for file in analytics]
        labels = [f.split('\\')[-1] for f in filenames]
        counts = np.zeros(clustering.children_.shape[0])
        n_samples = len(clustering.labels_)
        for i, merge in enumerate(clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [clustering.children_, clustering.distances_, counts]
        ).astype(float)
        plt.figure(figsize=(15, 7))
        plt.title("Hierarchical Clustering Dendrogram")
        dendrogram(linkage_matrix,labels=labels,**kwargs)
        plt.xlabel("Dendrogramm.")

    return indices_list
    
 
    
if __name__  == "__main__":
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

    np.random.seed(42)
    X = np.random.rand(10, 4)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(X)
    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()