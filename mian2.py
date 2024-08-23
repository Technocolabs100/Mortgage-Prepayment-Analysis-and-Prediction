import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
file_path = 'LoanExport.csv'
df = pd.read_csv(file_path, low_memory=False)

# Convert columns with mixed types to strings
for col in df.columns:
    if df[col].dtype == 'object':  # 'object' type in pandas often indicates mixed types
        df[col] = df[col].astype(str)

# Define the target variables for regression and classification
target_regression = 'EverDelinquent'  # Assume this is a continuous variable
target_classification = 'EverDelinquent'  # Assume this is a categorical variable

# Prepare features and targets for regression
X_reg = df.drop(columns=[target_regression])
y_reg = df[target_regression]

# Prepare features and targets for classification
X_clf = df.drop(columns=[target_classification])
y_clf = df[target_classification]

# Identify numeric and categorical features
numeric_features = X_reg.select_dtypes(include=[np.number]).columns
categorical_features = X_reg.select_dtypes(exclude=[np.number]).columns

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split data into train and test sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Regression Model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
model_pipeline_reg = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', regressor)])

# Train the regression model
model_pipeline_reg.fit(X_train_reg, y_train_reg)

# Predict on the test set
y_pred_reg = model_pipeline_reg.predict(X_test_reg)

# Evaluate the regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f'Regression Model Mean Squared Error: {mse}')
print(f'Regression Model R² Score: {r2}')

# Classification Model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
model_pipeline_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', classifier)])

# Train the classification model
model_pipeline_clf.fit(X_train_clf, y_train_clf)

# Predict on the test set
y_pred_clf = model_pipeline_clf.predict(X_test_clf)

# Evaluate the classification model
accuracy = accuracy_score(y_test_clf, y_pred_clf)
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
class_report = classification_report(y_test_clf, y_pred_clf)

print(f'Classification Model Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
