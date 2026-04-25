import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("features_data.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Define reference date
snapshot_date = df["InvoiceDate"].max()

# Customer-level metrics
customer = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # recency
    "InvoiceNo": "nunique",                                   # frequency
    "SalesValue": "sum",                                      # monetary
    "Description": "nunique"                                  # diversity
}).reset_index()

customer.columns = [
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "ProductDiversity"
]

# Create churn label
# Customers inactive for more than 90 days = churned
customer["Churn"] = customer["Recency"].apply(lambda x: 1 if x > 90 else 0)

# Features
X = customer[["Frequency", "Monetary", "ProductDiversity"]]
y = customer["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))