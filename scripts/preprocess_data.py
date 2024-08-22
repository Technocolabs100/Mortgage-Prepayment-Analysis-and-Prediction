import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv("../MortgageLoanExport/LoanExport.csv")

# 1. Data Cleaning
print('Missing Values before handling:\n', data.isnull().sum())

# Handle missing values
# For numeric data, we'll use the median
numeric_data = data.select_dtypes(include=[float, int]).columns
data[numeric_data] = data[numeric_data].fillna(data[numeric_data].median())

# For categorical data, we'll replace 'X' values with 'Unknown'
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = data[col].replace('X', 'Unknown')
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

print('Missing Values after handling:\n', data.isnull().sum())

# Remove duplicates
data = data.drop_duplicates()
print(f"Number of duplicates removed: {data.duplicated().sum()}")

# Correct data types
data['FirstPaymentDate'] = pd.to_datetime(data['FirstPaymentDate'], format='%Y%m', errors='coerce')
data['MaturityDate'] = pd.to_datetime(data['MaturityDate'], format='%Y%m', errors='coerce')
data['CreditScore'] = pd.to_numeric(data['CreditScore'], errors='coerce')

print("Data Types after conversion:\n", data.dtypes)

# 2. Data Encoding
label_encoder = LabelEncoder()

# Apply Label Encoding for binary categorical variables
data['FirstTimeHomebuyer'] = label_encoder.fit_transform(data['FirstTimeHomebuyer'].astype(str))
data['PPM'] = label_encoder.fit_transform(data['PPM'].astype(str))
data['NumBorrowers'] = label_encoder.fit_transform(data['NumBorrowers'].astype(str))

# Apply One-Hot Encoding for the remaining categorical columns
data = pd.get_dummies(data, columns=[
    'MSA', 'Occupancy', 'Channel', 'ProductType', 'PropertyState', 
    'PropertyType', 'LoanPurpose', 'SellerName', 'ServicerName'
], drop_first=True)

# 3. Data Labelling
# The target variable 'EverDelinquent' is assumed to be correctly labeled.
# Save the processed data
processed_data_path = '../MortgageLoanExport/LoanExport_processed.csv'
data.to_csv(processed_data_path, index=False)

print(f"Processed data saved to {processed_data_path}")
