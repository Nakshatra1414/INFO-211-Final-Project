import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('online_retail_cleaned.csv')

# Top products by revenue
top_products = df.groupby("Description")["SalesValue"].sum().sort_values(ascending=False).head(10)

print(top_products)

plt.figure()
top_products.plot(kind="bar")
plt.title("Top 10 Products by Revenue")
plt.ylabel("Revenue")
plt.show()

# Top countries
top_countries = df.groupby("Country")["SalesValue"].sum().sort_values(ascending=False).head(10)

plt.figure()
top_countries.plot(kind="bar")
plt.title("Top Countries by Revenue")
plt.ylabel("Revenue")
plt.show()