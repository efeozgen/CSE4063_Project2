import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json

class ECLAT:
    def __init__(self, json_file_path, min_support):
       
        self.json_file_path = json_file_path
        self.min_support = min_support

    def load_transactions(self):
       
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        transactions = []
        for entry in data:
            transaction = []
            if entry.get('bedrooms'):
                transaction.append(f"{entry['bedrooms']} bedrooms")
            if entry.get('tenure'):
                transaction.append(entry['tenure'])
            if entry.get('propertyType'):
                transaction.append(entry['propertyType'])
            transactions.append(transaction)

        return transactions

    def get_frequent_itemsets(self, transactions):
       
        itemsets = defaultdict(set)

        # Step 1: Create an item-to-transaction mapping
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                itemsets[frozenset([item])].add(tid)

        # Step 2: Recursive function to find larger itemsets
        def find_frequent_itemsets(prefix, candidates):
            results = {}
            for i, (itemset, tids) in enumerate(candidates.items()):
                support = len(tids)
                if support >= self.min_support:
                    results[itemset] = support
                    new_candidates = {}

                    for other_itemset, other_tids in list(candidates.items())[i + 1:]:
                        union_itemset = itemset | other_itemset
                        union_tids = tids & other_tids

                        if len(union_tids) >= self.min_support:
                            new_candidates[union_itemset] = union_tids

                    results.update(find_frequent_itemsets(itemset, new_candidates))

            return results

        # Step 3: Find all frequent itemsets
        return find_frequent_itemsets(frozenset(), itemsets)

    def plot_top_frequent_itemsets(self, frequent_itemsets, top_n=20):
        
        # Convert frequent itemsets to a sorted list of tuples
        sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Prepare data for plotting
        labels = [" + ".join(map(str, itemset)) for itemset, _ in sorted_itemsets]
        support_values = [support for _, support in sorted_itemsets]

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=support_values, y=labels, palette="viridis")
        plt.title(f"Top {top_n} Frequent Itemsets", fontsize=14)
        plt.xlabel("Support", fontsize=12)
        plt.ylabel("Itemsets", fontsize=12)
        plt.tight_layout()
        plt.show()

    def run(self):
       
        transactions = self.load_transactions()
        frequent_itemsets = self.get_frequent_itemsets(transactions)
        
        # Call the plot function to visualize the results
        self.plot_top_frequent_itemsets(frequent_itemsets)

        return frequent_itemsets

if __name__ == "__main__":
    # Example usage
    json_file = "final_clean_data.json"
    min_support = 50

    eclat = ECLAT(json_file, min_support)
    frequent_itemsets = eclat.run()

    print("Frequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"{set(itemset)}: {support}")
