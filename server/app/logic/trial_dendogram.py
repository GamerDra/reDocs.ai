from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# Initialize BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to tokenize and get embeddings
def code(codes):
    code_tokens = tokenizer.tokenize(codes)
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    return context_embeddings

# Function to generate embeddings for long texts
def embedding(text):
    t = 0
    window = 500
    overlap = 200
    step = 0
    size = len(text)
    b = np.array([0])
    while t < size:
        a = code(text[t:t + window]).detach().numpy().reshape(-1)
        size_a = a.size
        size_b = b.size
        diff = size_b - size_a
        if diff > 0:
            a = np.pad(a, (0, diff), 'constant')
        else:
            b = np.pad(b, (0, abs(diff)), 'constant')
        b = b + a
        t = t + window - overlap
        step += 1
        if t >= size:
            t = size
    b = b / step
    b = b.reshape(int(b.size / 768), 768)
    return b

# Function to reshape embeddings and pad them to the same length
def convert_embeddings(embeddings_list):
    reshaped_embeddings_list = []
    max_len = max(embedding.shape[0] for embedding in embeddings_list)
    for embedding in embeddings_list:
        reshaped_embedding = embedding.reshape(-1)
        reshaped_embedding = np.pad(reshaped_embedding, (0, max_len * 768 - reshaped_embedding.size), 'constant')
        reshaped_embeddings_list.append(reshaped_embedding)
    return reshaped_embeddings_list

# Function to perform clustering
def clustering(reshaped_embeddings_list):
    clustering_model = AgglomerativeClustering(n_clusters=len(reshaped_embeddings_list) // 2 + 1, linkage='complete', metric='cosine')
    clustering_model.fit(reshaped_embeddings_list)
    labels = clustering_model.labels_
    unique_labels = np.unique(labels)
    indices_list = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        indices_list.append(indices)
    return indices_list

# Function to plot dendrogram with labels
def plot_dendrogram(model, labels, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=labels, **kwargs)

# Generate some example text embeddings
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast dark-colored fox leaped over a sleepy dog.",
    "A red ball bounces on the green grass.",
    "Children are playing in the park.",
    "The sun sets over the horizon, painting the sky with vibrant colors.",
    "A beautiful sunset can be seen from the hilltop.",
    "Birds are singing in the morning light.",
    "The cat sleeps peacefully on the warm windowsill.",
    "A gentle breeze rustles the leaves in the forest.",
    "Raindrops patter softly on the roof."
]

# Short labels for each text
short_labels = [
    "Fox jumps lazy dog",
    "Fox leaped sleepy dog",
    "Red ball on grass",
    "Children playing park",
    "Sunset vibrant colors",
    "Sunset hilltop",
    "Birds singing morning",
    "Cat sleeps windowsill",
    "Breeze in forest",
    "Raindrops on roof"
]



embeddings = [embedding(text) for text in texts]

# Convert embeddings
reshaped_embeddings_list = convert_embeddings(embeddings)

# Debugging: Print lengths of embeddings and short_labels
print(f"Number of embeddings: {len(reshaped_embeddings_list)}")
print(f"Number of short labels: {len(short_labels)}")

# Print shapes of embeddings
for i, embedding in enumerate(reshaped_embeddings_list):
    print(f"Embedding {i} shape: {embedding.shape}")

# Perform clustering
clusters = clustering(reshaped_embeddings_list)

# Print cluster indices
for i, indices in enumerate(clusters):
    print(f"Cluster {i}: {indices}")

# Plot dendrogram with labels
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(reshaped_embeddings_list)

plt.figure(figsize=(15, 7))
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, labels=short_labels, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.xticks(rotation=90)
plt.show()
