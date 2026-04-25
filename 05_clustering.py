import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("features_data.csv")

# Customer behavior summary
customer = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Quantity": "sum",
    "SalesValue": "sum",
    "Description": "nunique"
}).reset_index()

customer.columns = [
    "CustomerID",
    "TransactionCount",
    "TotalQuantity",
    "TotalSpend",
    "ProductDiversity"
]

# Scale
features = customer[[
    "TransactionCount",
    "TotalQuantity",
    "TotalSpend",
    "ProductDiversity"
]]

scaled = StandardScaler().fit_transform(features)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
customer["Cluster"] = kmeans.fit_predict(scaled)

# Visualize
plt.scatter(customer["TotalSpend"], customer["TransactionCount"], c=customer["Cluster"])
plt.xlabel("Total Spend")
plt.ylabel("Transaction Count")
plt.title("Customer Segments")
plt.show()

print(customer.head())