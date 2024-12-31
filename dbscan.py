from sklearn.neighbors import NearestNeighbors
import numpy as np

class DBSCAN:
    def __init__(self, eps, min_samples):
        """
        DBSCAN algoritması için başlangıç.
        :param eps: Komşuluk yarıçapı.
        :param min_samples: Minimum nokta sayısı.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def fit(self, data):
        """
        DBSCAN algoritmasını çalıştırır.
        :param data: Numpy array veya liste veri kümesi.
        """
        num_points = len(data)
        self.labels = np.full(num_points, -1)  # Gürültü noktalar başlangıçta -1 ile etiketlenir
        cluster_id = 0
        
        # Tüm noktalar için komşu noktaları hesapla
        neighbors_model = NearestNeighbors(radius=self.eps).fit(data)
        distances, neighbors = neighbors_model.radius_neighbors(data)
        
        for point_idx in range(num_points):
            if self.labels[point_idx] != -1:
                continue  # Zaten işlenmiş
            
            # Komşuları belirle
            neighbor_indices = neighbors[point_idx]
            if len(neighbor_indices) < self.min_samples:
                self.labels[point_idx] = -1  # Gürültü
            else:
                # Yeni bir küme başlat
                self._expand_cluster(data, neighbors, point_idx, cluster_id, neighbor_indices)
                cluster_id += 1
    
    def _expand_cluster(self, data, neighbors, point_idx, cluster_id, neighbor_indices):
        """
        Küme genişletme işlemi.
        """
        self.labels[point_idx] = cluster_id
        queue = list(neighbor_indices)
        
        while queue:
            current_point = queue.pop(0)
            if self.labels[current_point] == -1:  # Gürültüyse kümele
                self.labels[current_point] = cluster_id
            
            elif self.labels[current_point] == -1:
                self.labels[current_point] = cluster_id
                current_neighbors = neighbors[current_point]
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
    
    def get_labels(self):
        """
        Nokta etiketlerini döndür.
        """
        return self.labels
