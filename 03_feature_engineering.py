import pandas as pd

df = pd.read_csv('online_retail_cleaned.csv')

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Extract time features
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour

# Save updated dataset
df.to_csv("features_data.csv", index=False)

print("Feature engineering completed")