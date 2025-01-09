import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class DBSCANClusterer:
    def __init__(self, json_file_path, batch_size=10000, eps=0.05, min_samples=5):
       
        self.json_file_path = json_file_path
        self.batch_size = batch_size
        self.eps = eps
        self.min_samples = min_samples

    def load_data(self):

        with open(self.json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def process_data(self):
        
        data = self.load_data()
        all_points = []
        all_labels = []

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]

            # Extract prices
            prices = [float(item['saleEstimate_currentPrice']) for item in batch if item.get('saleEstimate_currentPrice')]
            prices = np.array(prices).reshape(-1, 1)

            # Normalize prices
            scaler = MinMaxScaler()
            prices_normalized = scaler.fit_transform(prices)
            all_points.append(prices_normalized)

            # Apply DBSCAN
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = model.fit_predict(prices_normalized)
            all_labels.append(labels)

        # Combine all points and labels
        all_points = np.vstack(all_points)
        all_labels = np.concatenate(all_labels)

        return all_points, all_labels

    def plot_clusters(self, all_points, all_labels):
        
        unique_labels = set(all_labels)
        for label in unique_labels:
            if label == -1:  # Mark noise points
                color = 'red'
                marker = 'x'
            else:
                color = 'blue'
                marker = 'o'

            cluster_points = all_points[all_labels == label]
            plt.scatter(cluster_points, np.zeros_like(cluster_points), c=color, label=f'Cluster {label}', marker=marker)

        plt.title('DBSCAN Clustering on Prices (All Batches)')
        plt.xlabel('Normalized Sale Price')
        plt.legend()
        plt.show()

    def run(self):
      
        all_points, all_labels = self.process_data()
        self.plot_clusters(all_points, all_labels)
        return all_points, all_labels

if __name__ == "__main__":
    # Example usage
    json_file = "final_clean_data.json"
    dbscan_clusterer = DBSCANClusterer(json_file)
    all_points, all_labels = dbscan_clusterer.run()
