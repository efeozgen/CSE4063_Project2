from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import numpy as np
import random
import matplotlib.pyplot as plt

class AGNESClustering:
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
    
    def fit(self, data):
        self.model.fit(data)
        return self.model.labels_

def load_json_data(json_file_path):
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data

def convert_to_numeric(data):
    numeric_data = []
    for entry in data:
        numeric_entry = []
        for key in entry.keys():
            try:
                numeric_entry.append(float(entry[key]))
            except ValueError:
                numeric_entry.append(0.0)  # or handle non-numeric values as needed
        numeric_data.append(numeric_entry)
    return np.array(numeric_data)

def plot_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title("AGNES Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Example usage
if __name__ == "__main__":
    json_file_path = 'final_clean_data.json'
    data = load_json_data(json_file_path)
    
    if len(data) > 10000:
        data = random.sample(data, 10000)
    
    data_matrix = convert_to_numeric(data)
    
    # Scale the data
    scaler = StandardScaler()
    data_matrix = scaler.fit_transform(data_matrix)
    
    agnes = AGNESClustering(n_clusters=3)
    labels = agnes.fit(data_matrix)
    
    print("AGNES Clustering Labels:")
    print(labels)
    
    plot_clusters(data_matrix, labels)
