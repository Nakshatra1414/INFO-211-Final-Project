# 09_visualization.py
# Refined Professional Business Intelligence Visualizations
# Red-free, muted, high-level presentation aesthetics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Global Style Configuration
# -----------------------------
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

# Carefully chosen muted palettes (no reds / greens)
palette_products = sns.color_palette("Blues_d", 10)
palette_countries = sns.color_palette("Purples_d", 10)
palette_hist = sns.color_palette("PuBu", 8)
palette_months = sns.color_palette("cividis", 12)
palette_weekdays = sns.color_palette("GnBu", 7)
scatter_color = "#4C6A92"   # muted steel blue

# -----------------------------
# Load and Prepare Data
# -----------------------------
df = pd.read_csv('online_retail_cleaned.csv')

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])

if "TotalSpend" not in df.columns:
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

# -----------------------------
# Visualization 1: Top Products
# -----------------------------
top_products = (
    df.groupby("Description")["TotalSpend"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
sns.barplot(
    x=top_products.values,
    y=top_products.index,
    palette=palette_products
)
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue Generated")
plt.ylabel("Product Description")
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 2: Top Countries
# -----------------------------
top_countries = (
    df.groupby("Country")["TotalSpend"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
sns.barplot(
    x=top_countries.values,
    y=top_countries.index,
    palette=palette_countries
)
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue Generated")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 3: Customer Spending Distribution
# -----------------------------
customer_spend = df.groupby("CustomerID")["TotalSpend"].sum()

plt.figure()
sns.histplot(
    customer_spend,
    bins=40,
    kde=True,
    color=palette_hist[4]
)
plt.title("Distribution of Customer Spending")
plt.xlabel("Total Spending per Customer")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 4: Monthly Trends
# -----------------------------
df["Month"] = df["InvoiceDate"].dt.month_name()

month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

monthly_sales = (
    df.groupby("Month")["TotalSpend"]
    .sum()
    .reindex(month_order)
)

plt.figure()
sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o",
    linewidth=3,
    color=palette_months[6]
)
plt.xticks(rotation=45)
plt.title("Monthly Revenue Trends")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 5: Weekday Revenue
# -----------------------------
df["Weekday"] = df["InvoiceDate"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    df.groupby("Weekday")["TotalSpend"]
    .sum()
    .reindex(weekday_order)
)

plt.figure()
sns.barplot(
    x=weekday_sales.index,
    y=weekday_sales.values,
    palette=palette_weekdays
)
plt.title("Busiest Weekdays by Revenue")
plt.xlabel("Day of Week")
plt.ylabel("Revenue Generated")
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 6: Quantity vs Revenue
# -----------------------------
plt.figure()
sns.scatterplot(
    data=df,
    x="Quantity",
    y="TotalSpend",
    alpha=0.6,
    color=scatter_color
)
plt.title("Order Quantity vs Revenue")
plt.xlabel("Quantity Purchased")
plt.ylabel("Revenue Generated")
plt.tight_layout()
plt.show()