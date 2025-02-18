import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score
import time

# Load the datasets
# Assuming 'mathematics_data' and 'portuguese_data' are DataFrames with the last column as the target
mathematics_data = pd.read_csv('mathematics_data.csv')  # Replace with actual path
portuguese_data = pd.read_csv('portuguese_data.csv')  # Replace with actual path

# Split features and target
X_math = mathematics_data.drop(columns=['target'])
y_math = mathematics_data['target']

X_port = portuguese_data.drop(columns=['target'])
y_port = portuguese_data['target']

# Standardize features
scaler = StandardScaler()
X_math_scaled = scaler.fit_transform(X_math)
X_port_scaled = scaler.fit_transform(X_port)

# Feature Selection: RFE (Recursive Feature Elimination) for feature optimization
# You can replace this with UniSelect/MMFW-RFE if you have those implementations
selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)  # Select top 10 features
X_math_selected = selector.fit_transform(X_math_scaled, y_math)
X_port_selected = selector.fit_transform(X_port_scaled, y_port)

# Initialize base classifiers
dt = DecisionTreeClassifier(random_state=42)
nb = GaussianNB()
svm = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()
nn = MLPClassifier(random_state=42)

# Create the ensemble model using Voting Classifier (Hard voting)
ensemble_model = VotingClassifier(
    estimators=[('dt', dt), ('nb', nb), ('svm', svm), ('knn', knn), ('nn', nn)],
    voting='hard'
)

# Train and evaluate on Mathematics dataset (Binary Classification)
start_time = time.time()
ensemble_model.fit(X_math_selected, y_math)
y_pred_math = ensemble_model.predict(X_math_selected)
accuracy_math = accuracy_score(y_math, y_pred_math)
fscore_math = f1_score(y_math, y_pred_math, average='binary')  # Assuming binary classification
run_time_math = time.time() - start_time

# Train and evaluate on Portuguese dataset (Binary Classification)
start_time = time.time()
ensemble_model.fit(X_port_selected, y_port)
y_pred_port = ensemble_model.predict(X_port_selected)
accuracy_port = accuracy_score(y_port, y_pred_port)
fscore_port = f1_score(y_port, y_pred_port, average='binary')  # Assuming binary classification
run_time_port = time.time() - start_time

# Print results for Binary Classification
print(f"Mathematics Dataset - Binary Classification")
print(f"Accuracy: {accuracy_math:.2f}, F-score: {fscore_math:.2f}, Run-time: {run_time_math:.2f}s")

print(f"Portuguese Dataset - Binary Classification")
print(f"Accuracy: {accuracy_port:.2f}, F-score: {fscore_port:.2f}, Run-time: {run_time_port:.2f}s")

# Evaluate Multiclass Classification for Mathematics dataset
start_time = time.time()
ensemble_model.fit(X_math_selected, y_math)
y_pred_math_multiclass = ensemble_model.predict(X_math_selected)
accuracy_math_multiclass = accuracy_score(y_math, y_pred_math_multiclass)
fscore_math_multiclass = f1_score(y_math, y_pred_math_multiclass, average='weighted')  # Multiclass F-score
run_time_math_multiclass = time.time() - start_time

# Evaluate Multiclass Classification for Portuguese dataset
start_time = time.time()
ensemble_model.fit(X_port_selected, y_port)
y_pred_port_multiclass = ensemble_model.predict(X_port_selected)
accuracy_port_multiclass = accuracy_score(y_port, y_pred_port_multiclass)
fscore_port_multiclass = f1_score(y_port, y_pred_port_multiclass, average='weighted')  # Multiclass F-score
run_time_port_multiclass = time.time() - start_time

# Print results for Multiclass Classification
print(f"\nMathematics Dataset - Multiclass Classification")
print(f"Accuracy: {accuracy_math_multiclass:.2f}, F-score: {fscore_math_multiclass:.2f}, Run-time: {run_time_math_multiclass:.2f}s")

print(f"Portuguese Dataset - Multiclass Classification")
print(f"Accuracy: {accuracy_port_multiclass:.2f}, F-score: {fscore_port_multiclass:.2f}, Run-time: {run_time_port_multiclass:.2f}s")
