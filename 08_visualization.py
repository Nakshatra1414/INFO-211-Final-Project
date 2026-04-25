import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('online_retail_cleaned.csv')

# Revenue by product
top_products = df.groupby("Description")["SalesValue"].sum().sort_values(ascending=False).head(10)

plt.figure()
top_products.plot(kind="bar")
plt.title("Top Products")
plt.show()

# Revenue by country
top_countries = df.groupby("Country")["SalesValue"].sum().head(10)

plt.figure()
top_countries.plot(kind="bar")
plt.title("Top Countries")
plt.show()