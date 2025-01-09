import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def optimize_data_types(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def one_hot_encode_columns(df, columns):
    for col in columns:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, one_hot], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df

def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def reduce_dimensions(data, n_components=10):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    print(f"Reduced data shape: {reduced_data.shape}")
    print(f"Explained Variance Ratio for {n_components} components: {sum(pca.explained_variance_ratio_)}")
    return reduced_data

def inspect_pca_components(data, n_components=10):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("PCA Components (contributions of each feature):")
    print(pd.DataFrame(pca.components_, columns=[f"Feature {i}" for i in range(data.shape[1])]))

def determine_optimal_clusters(data, max_k=10):
    distortions = []
    for k in range(2, max_k+1):
        model = MiniBatchKMeans(n_clusters=k, random_state=42)
        model.fit(data)
        distortions.append(model.inertia_)
    
    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k+1), distortions, marker='o', label='Distortion')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.legend()
    plt.show()

class OptimizedKMeansClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=2000, random_state=42)
        self.df = None

    def load_json_data(self, json_file_path):
        self.df = pd.read_json(json_file_path)
        print(f"Data loaded. DataFrame shape: {self.df.shape}")
        return self.df

    def preprocess_data(self):
        # Optimize data types
        self.df = optimize_data_types(self.df)
        
        # Fill missing values
        self.df.fillna(0, inplace=True)
        
        # One-hot encode categorical columns
        categorical_columns = ['tenure', 'propertyType', 'saleEstimate_confidenceLevel']
        self.df = one_hot_encode_columns(self.df, categorical_columns)
        
        print("Data preprocessed with One-Hot Encoding.")
        print(f"Encoded Data Shape: {self.df.shape}")
        return self.df

    def fit(self, data):
        self.model.fit(data)
        return self.model.labels_

    def reduce_and_cluster(self):
        # Take a subset of 10,000 rows for clustering
        subset_df = self.df.sample(n=10000, random_state=42)
        print(f"Subset Data Shape: {subset_df.shape}")

        # Scale data
        scaled_data = scale_data(subset_df.values)

        # Inspect PCA components
        inspect_pca_components(scaled_data, n_components=10)

        # Reduce dimensions
        reduced_data = reduce_dimensions(scaled_data, n_components=10)

        # Determine optimal number of clusters using Elbow Method
        determine_optimal_clusters(reduced_data, max_k=10)

        # Cluster
        labels = self.fit(reduced_data)
        print("Clustering completed.")

        # Visualization (Using first 2 PCA components for simplicity)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
        plt.title("KMeans Clustering Visualization (First 2 PCA Components)")
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
    kmeans = OptimizedKMeansClustering(n_clusters=5)  # Set optimal number of clusters
    
    # Load and preprocess data
    df = kmeans.load_json_data(json_file_path)
    df = kmeans.preprocess_data()
    
    # Perform clustering on a 10,000-row subset
    labels = kmeans.reduce_and_cluster()
