import pandas as pd

# Load dataset from workspace
file_path = 'online_retail.csv'
df = pd.read_csv(file_path)

# Display basic structure of the dataset
print(df.head())

# Remove rows with missing customer IDs because they are needed for behavior analysis
df = df.dropna(subset=['CustomerID'])

# Remove rows where quantity or price is zero/negative
# These may represent returns, errors, or invalid transactions
# Keeping only valid purchases improves analysis quality
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Create a new column for total transaction value
df['SalesValue'] = df['Quantity'] * df['UnitPrice']

# Save cleaned dataset
cleaned_path = 'online_retail_cleaned.csv'
df.to_csv(cleaned_path, index=False)

print('Data cleaned and saved successfully.')