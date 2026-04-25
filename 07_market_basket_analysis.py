import pandas as pd
import mlxtend.frequent_patterns  # type: ignore
# Load data
df = pd.read_csv('online_retail_cleaned.csv')

# Create transaction-product matrix
basket = df.groupby(["InvoiceNo", "Description"])["Quantity"] \
           .sum().unstack().fillna(0)

# Convert to boolean
basket = basket > 0

# Frequent itemsets
frequent_items = mlxtend.frequent_patterns.apriori(basket, min_support=0.02, use_colnames=True)

# Association rules
rules = mlxtend.frequent_patterns.association_rules(frequent_items, metric="lift", min_threshold=1)

# Display top rules
print(
    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    .sort_values(by="lift", ascending=False)
    .head(10)
)