import pandas as pd

df = pd.read_csv('online_retail_cleaned.csv')

df["SalesValue"] = df["Quantity"] * df["UnitPrice"]

print("Top Products:")
print(df.groupby("Description")["SalesValue"].sum().sort_values(ascending=False).head(10))

print("Top Countries:")
print(df.groupby("Country")["SalesValue"].sum().sort_values(ascending=False).head(10))

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["Month"] = df["InvoiceDate"].dt.month_name()
df["Weekday"] = df["InvoiceDate"].dt.day_name()

busiest_months = df["Month"].value_counts()
busiest_days = df["Weekday"].value_counts()

print(busiest_months.head())
print(busiest_days)