import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the preprocessed data
data = pd.read_csv("../MortgageLoanExport/LoanExport_processed.csv")

# Drop any non-numeric columns (like dates) or convert them to numeric if applicable
# Example: If there are date columns, you might drop them or extract relevant features like the year or month.
data = data.select_dtypes(include=[float, int])

# Set 'EverDelinquent' as the target variable for classification
y = data['EverDelinquent']
X = data.drop('EverDelinquent', axis=1)

# Split the data into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model (Classification)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Make predictions (Classification)
y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Classification Accuracy: {accuracy:.4f}")
print(f"Classification Precision: {precision:.4f}")
print(f"Classification Recall: {recall:.4f}")
print(f"Classification F1-Score: {f1:.4f}")
print(f"Classification ROC-AUC Score: {roc_auc:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Plotting the ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(data['EverDelinquent'].value_counts(normalize=True))
