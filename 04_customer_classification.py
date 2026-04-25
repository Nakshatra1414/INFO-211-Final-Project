import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("features_data.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Customer-level features
customer = df.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "Quantity": "sum",
    "UnitPrice": "mean",
    "Description": "nunique",
    "SalesValue": "sum"
}).reset_index()

customer.columns = [
    "CustomerID",
    "TransactionCount",
    "TotalQuantity",
    "AvgUnitPrice",
    "ProductDiversity",
    "TotalSpend"
]

# Create target tiers
customer["Tier"] = pd.qcut(customer["TotalSpend"], q=3, labels=[0,1,2])

# Features WITHOUT TotalSpend
X = customer[[
    "TransactionCount",
    "TotalQuantity",
    "AvgUnitPrice",
    "ProductDiversity"
]]
y = customer["Tier"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))