import itertools

class ECLAT:
    def __init__(self, dataset):
        """
        ECLAT algoritması için başlangıç.
        :param dataset: Listelerden oluşan veri kümesi.
        """
        self.dataset = dataset
        self.itemsets = {}
    
    def fit(self, min_support):
        """
        ECLAT algoritmasını çalıştırır.
        :param min_support: Minimum destek değeri.
        """
        # Tüm öğeleri içeren bir liste oluştur
        items = {}
        for transaction_id, transaction in enumerate(self.dataset):
            for item in transaction:
                if item in items:
                    items[item].add(transaction_id)
                else:
                    items[item] = {transaction_id}
        
        # Tüm öğe kümelerini hesapla
        self.itemsets = {frozenset([item]): transactions 
                         for item, transactions in items.items() 
                         if len(transactions) >= min_support}
        
        k = 2
        while True:
            new_combinations = self._generate_combinations(k)
            if not new_combinations:
                break
            k += 1
    
    def _generate_combinations(self, k):
        """
        Yeni öğe kombinasyonları oluştur.
        """
        new_combinations = {}
        keys = list(self.itemsets.keys())
        
        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1:]:
                union_set = key1.union(key2)
                if len(union_set) == k:
                    transactions = self.itemsets[key1].intersection(self.itemsets[key2])
                    if len(transactions) >= min_support:
                        new_combinations[union_set] = transactions
        
        if new_combinations:
            self.itemsets.update(new_combinations)
        return new_combinations
    
    def get_frequent_itemsets(self):
        """
        Sık öğe kümelerini al.
        """
        return {item: len(transactions) for item, transactions in self.itemsets.items()}
