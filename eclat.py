import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class Eclat:
    def __init__(self, json_file_path, min_support=0.01):
        self.json_file_path = json_file_path
        self.min_support = min_support

    def load_transactions(self):
        with open(self.json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        transactions = []
        for entry in data:
            transaction = []
            if entry.get("bedrooms"):
                transaction.append(f"{entry['bedrooms']} bedrooms")
            if entry.get("tenure"):
                transaction.append(entry["tenure"])
            if entry.get("propertyType"):
                transaction.append(entry["propertyType"])
            transactions.append(transaction)

        return transactions

    def get_frequent_itemsets(self, transactions):
        itemsets = defaultdict(set)

        # Creating an item-to-transaction mapping
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                itemsets[item].add(tid)

        # Generating frequent itemsets
        frequent_itemsets = {
            frozenset([item]): len(tids)
            for item, tids in itemsets.items()
            if len(tids) / len(transactions) >= self.min_support
        }

        return frequent_itemsets

    def plot_top_frequent_itemsets(self, frequent_itemsets, top_n=20):

        # Convert frequent itemsets to a sorted list of tuples
        sorted_itemsets = sorted(
            frequent_itemsets.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        # Prepare data for plotting
        labels = [" + ".join(map(str, itemset)) for itemset, _ in sorted_itemsets]
        support_values = [support for _, support in sorted_itemsets]

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=support_values, y=labels, palette="viridis")
        plt.title(f"Top {top_n} Frequent Itemsets", fontsize=14)
        plt.xlabel("Support")
        plt.ylabel("Itemsets")
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_support(self, frequent_itemsets, transactions):
        items = list(frequent_itemsets.keys())
        supports = [frequent_itemsets[item] / len(transactions) for item in items]

        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(items)), supports, tick_label=[list(item)[0] for item in items]
        )
        plt.xlabel("Itemsets")
        plt.ylabel("Support")
        plt.title("Support of Frequent Itemsets")
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show()

    def run(self):
        transactions = self.load_transactions()
        frequent_itemsets = self.get_frequent_itemsets(transactions)
        return frequent_itemsets, transactions


# Example usage
if __name__ == "__main__":
    json_file = "C:/Users/HP/Desktop/Marun/Dönem 7/Data Mining/Projects/CSE4063_Project2/final_clean_data.json"
    min_support = 0.01

    eclat = Eclat(json_file, min_support)
    frequent_itemsets, transactions = eclat.run()

    print("Frequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"{set(itemset)}: {support}")

    # Plot frequent itemsets
    eclat.plot_top_frequent_itemsets(frequent_itemsets)

    # Plot support values
    eclat.plot_support(frequent_itemsets, transactions)
