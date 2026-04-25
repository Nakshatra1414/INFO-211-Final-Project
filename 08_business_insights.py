import pandas as pd

df = pd.read_csv('online_retail_cleaned.csv')

df["SalesValue"] = df["Quantity"] * df["UnitPrice"]

print("Top Products:")
print(df.groupby("Description")["SalesValue"].sum().sort_values(ascending=False).head(10))

print("Top Countries:")
print(df.groupby("Country")["SalesValue"].sum().sort_values(ascending=False).head(10))