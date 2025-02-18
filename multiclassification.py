import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (adjust the file paths as needed)
math_data = pd.read_csv("math_dataset.csv")  # Replace with actual path
portuguese_data = pd.read_csv("portuguese_dataset.csv")  # Replace with actual path

# Prepare data (assuming the last column is the target)
X_math = math_data.iloc[:, :-1]
y_math = math_data.iloc[:, -1]
X_portuguese = portuguese_data.iloc[:, :-1]
y_portuguese = portuguese_data.iloc[:, -1]

# Split the dataset into training and test sets
X_train_math, X_test_math, y_train_math, y_test_math = train_test_split(X_math, y_math, test_size=0.3, random_state=42)
X_train_portuguese, X_test_portuguese, y_train_portuguese, y_test_portuguese = train_test_split(X_portuguese, y_portuguese, test_size=0.3, random_state=42)

# Standardize the data (optional)
scaler = StandardScaler()
X_train_math = scaler.fit_transform(X_train_math)
X_test_math = scaler.transform(X_test_math)
X_train_portuguese = scaler.fit_transform(X_train_portuguese)
X_test_portuguese = scaler.transform(X_test_portuguese)

# Function to train and evaluate classifiers
def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1-score for multiclass
    return accuracy, f1

# Initialize classifiers
classifiers = {
    "DT": DecisionTreeClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "NN": MLPClassifier(),
    "KNN": KNeighborsClassifier()
}

# Evaluate classifiers without feature selection
results_math_no_fs = {}
results_portuguese_no_fs = {}
for name, classifier in classifiers.items():
    accuracy, f1 = evaluate_classifier(classifier, X_train_math, X_test_math, y_train_math, y_test_math)
    results_math_no_fs[name] = (accuracy, f1)
    
    accuracy, f1 = evaluate_classifier(classifier, X_train_portuguese, X_test_portuguese, y_train_portuguese, y_test_portuguese)
    results_portuguese_no_fs[name] = (accuracy, f1)

print("Results without feature selection (Mathematics Dataset):")
for name, (acc, f1) in results_math_no_fs.items():
    print(f"{name}: Accuracy = {acc:.2f}, F-Score = {f1:.2f}")

print("\nResults without feature selection (Portuguese Dataset):")
for name, (acc, f1) in results_portuguese_no_fs.items():
    print(f"{name}: Accuracy = {acc:.2f}, F-Score = {f1:.2f}")

# Feature Selection with RFE (Recursive Feature Elimination)
def apply_rfe(X_train, y_train, X_test, n_features):
    rf = RandomForestClassifier()
    rfe = RFE(rf, n_features_to_select=n_features)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)
    return X_train_rfe, X_test_rfe

# Apply feature selection techniques: RRFE and CV-RFE
n_features_math = 10  # Choose 10 features as an example
n_features_portuguese = 10  # Choose 10 features as an example

X_train_math_rfe, X_test_math_rfe = apply_rfe(X_train_math, y_train_math, X_test_math, n_features_math)
X_train_portuguese_rfe, X_test_portuguese_rfe = apply_rfe(X_train_portuguese, y_train_portuguese, X_test_portuguese, n_features_portuguese)

# Evaluate classifiers after RFE
results_math_rfe = {}
results_portuguese_rfe = {}
for name, classifier in classifiers.items():
    accuracy, f1 = evaluate_classifier(classifier, X_train_math_rfe, X_test_math_rfe, y_train_math, y_test_math)
    results_math_rfe[name] = (accuracy, f1)
    
    accuracy, f1 = evaluate_classifier(classifier, X_train_portuguese_rfe, X_test_portuguese_rfe, y_train_portuguese, y_test_portuguese)
    results_portuguese_rfe[name] = (accuracy, f1)

print("\nResults after RFE Feature Selection (Mathematics Dataset):")
for name, (acc, f1) in results_math_rfe.items():
    print(f"{name}: Accuracy = {acc:.2f}, F-Score = {f1:.2f}")

print("\nResults after RFE Feature Selection (Portuguese Dataset):")
for name, (acc, f1) in results_portuguese_rfe.items():
    print(f"{name}: Accuracy = {acc:.2f}, F-Score = {f1:.2f}")

# Add other feature selection methods as needed (CV-RFE, RRFE + Filter, etc.)
# For example, apply PCA, CSF, FIF, and other custom techniques following similar patterns
# Example: Apply PCA
pca = PCA(n_components=10)
X_train_math_pca = pca.fit_transform(X_train_math)
X_test_math_pca = pca.transform(X_test_math)
X_train_portuguese_pca = pca.fit_transform(X_train_portuguese)
X_test_portuguese_pca = pca.transform(X_test_portuguese)

# Evaluate classifiers after PCA
results_math_pca = {}
results_portuguese_pca = {}
for name, classifier in classifiers.items():
    accuracy, f1 = evaluate_classifier(classifier, X_train_math_pca, X_test_math_pca, y_train_math, y_test_math)
    results_math_pca[name] = (accuracy, f1)
    
    accuracy, f1 = evaluate_classifier(classifier, X_train_portuguese_pca, X_test_portuguese_pca, y_train_portuguese, y_test_portuguese)
    results_portuguese_pca[name] = (accuracy, f1)

print("\nResults after PCA Feature Selection (Mathematics Dataset):")
for name, (acc, f1) in results_math_pca.items():
    print(f"{name}: Accuracy = {acc:.2f}, F-Score = {f1:.2f}")

print("\nResults after PCA Feature Selection (Portuguese Dataset):")
for name, (acc, f1) in results_portuguese_pca.items():
    print(f"{name}: Accuracy = {acc:.2f}, F-Score = {f1:.2f}")
