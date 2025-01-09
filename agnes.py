import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

class AGNESClustering:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        self.df = None

    def load_json_data(self, json_file_path):
        self.df = pd.read_json(json_file_path)
        print(f"Data loaded. DataFrame shape: {self.df.shape}")
        return self.df

    def preprocess_data(self):
        # Optimize data types
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = self.df[col].astype('float32')
        for col in self.df.select_dtypes(include=['int64']).columns:
            self.df[col] = self.df[col].astype('int32')

        # Fill missing values
        self.df.fillna(0, inplace=True)

        # One-hot encode categorical columns
        categorical_columns = ['tenure', 'propertyType', 'saleEstimate_confidenceLevel']
        for col in categorical_columns:
            one_hot = pd.get_dummies(self.df[col], prefix=col)
            self.df = pd.concat([self.df, one_hot], axis=1)
            self.df.drop(col, axis=1, inplace=True)

        print("Data preprocessed with One-Hot Encoding.")
        print(f"Encoded Data Shape: {self.df.shape}")
        return self.df

    def scale_data(self, data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)

    def reduce_dimensions(self, data, n_components=2):
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        print(f"Reduced data shape: {reduced_data.shape}")
        print(f"Explained Variance Ratio: {sum(pca.explained_variance_ratio_)}")
        return reduced_data

    def plot_dendrogram(self, data):
        plt.figure(figsize=(10, 7))
        dendrogram = sch.dendrogram(sch.linkage(data, method=self.linkage))
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

    def fit(self, data):
        self.model.fit(data)
        return self.model.labels_

    def cluster_and_visualize(self):
        # Take a subset of 10,000 rows for clustering
        subset_df = self.df.sample(n=10000, random_state=42)
        print(f"Subset Data Shape: {subset_df.shape}")

        # Scale data
        scaled_data = self.scale_data(subset_df.values)

        # Reduce dimensions for visualization
        reduced_data = self.reduce_dimensions(scaled_data, n_components=2)

        # Plot dendrogram
        self.plot_dendrogram(scaled_data)

        # Cluster
        labels = self.fit(reduced_data)
        print("Clustering completed.")

        # Visualization (Using first 2 PCA components for simplicity)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
        plt.title("AGNES Clustering Visualization (First 2 PCA Components)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Cluster')
        plt.show()

        # Analyze each cluster
        subset_df['Cluster'] = labels
        cluster_summary = subset_df.groupby('Cluster').mean()
        print("Cluster Summary:")
        print(cluster_summary)

        return labels

# Usage example
if __name__ == "__main__":
    json_file_path = 'final_clean_data.json'

    # Initialize clustering class
    agnes = AGNESClustering(n_clusters=4, linkage='ward')

    # Load and preprocess data
    df = agnes.load_json_data(json_file_path)
    df = agnes.preprocess_data()

    # Perform clustering and visualization
    labels = agnes.cluster_and_visualize()
