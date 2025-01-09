import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DBSCANClustering:
    def __init__(self):
        self.df = None

    def load_data(self, file_name):
        # JSON verisini oku
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"Loaded Data Shape: {df.shape}")  # Verinin şekline bakın
        
        # Sayısal verilere dönüştür
        df = df.apply(pd.to_numeric, errors='coerce')  # Dönüştürürken hata verirse NaN yapar
        
        # Kategorik sütunları geçici olarak çıkar
        df = df.drop(columns=['tenure', 'propertyType', 'saleEstimate_confidenceLevel'])
        
        # Eksik verileri at
        df = df.dropna()  # NaN olan satırları sil
        
        print(f"Cleaned Data Shape: {df.shape}")  # Temizlenmiş verinin şekli
        return df

    def scale_data(self, data):
        # Veriyi standartlaştır
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def reduce_dimensions(self, data, n_components=2):
        # Veriyi PCA ile 2 boyuta indir
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        return reduced_data

    def fit(self, data):
        # DBSCAN algoritması ile kümeleme yap
        dbscan = DBSCAN(eps=0.4, min_samples=10)  # DBSCAN parametrelerini buradan ayarlayabilirsiniz
        labels = dbscan.fit_predict(data)
        return labels

    def cluster_and_visualize(self, file_name):
        # Veriyi yükle
        self.df = self.load_data(file_name)

        # Verinin büyüklüğünü kontrol et
        if len(self.df) == 0:
            raise ValueError("Dataframe is empty. Check the data loading process.")
        
        # Yeni özellikler seç (rentEstimate_currentPrice, saleEstimate_currentPrice, floorAreaSqM)
        features = ['rentEstimate_currentPrice', 'saleEstimate_currentPrice', 'floorAreaSqM']
        subset_df = self.df[features]
        
        # Subset al (ilk 10000 satır)
        subset_df = subset_df.sample(n=min(10000, len(subset_df)), random_state=42)
        print(f"Subset Data Shape: {subset_df.shape}")

        # Veriyi ölçeklendir
        scaled_data = self.scale_data(subset_df.values)

        # Boyutları indir
        reduced_data = self.reduce_dimensions(scaled_data, n_components=2)

        # DBSCAN ile kümeleme yap
        labels = self.fit(scaled_data)
        print("Clustering completed.")

        # DBSCAN sonucunu görselleştir
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
        plt.title("DBSCAN Clustering Visualization (First 2 PCA Components)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label='Cluster')
        plt.show()

        # Her küme için istatistiksel analiz
        subset_df['Cluster'] = labels
        cluster_summary = subset_df.groupby('Cluster').mean()
        print("Cluster Summary:")
        print(cluster_summary)

        return labels

# Uygulama kısmı
if __name__ == "__main__":
    visualization = DBSCANClustering()
    labels = visualization.cluster_and_visualize('final_clean_data.json')
