# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Step 1: Calculate Chi-Square score for each feature
def chi_square_selection(X, y, k1):
    chi2_selector = SelectKBest(chi2, k=k1)
    chi2_selector.fit(X, y)
    selected_features_chi2 = chi2_selector.get_support(indices=True)
    return selected_features_chi2

# Step 2: Feature Importance Selection using ExtraTreesClassifier
def feature_importance_selection(X, y, k2):
    # Using ExtraTreesClassifier to rank feature importance
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Get feature importance scores
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1]  # Sort in descending order
    top_k_features = sorted_idx[:k2]
    return top_k_features

# Step 3: Custom Scoring Metric for Feature Selection
def custom_scoring_metric(X):
    # Example: Variance as custom metric for feature selection
    variance_scores = X.var(axis=0)
    return variance_scores

# Step 4: Feature Correlation Removal (removes highly correlated features)
def remove_high_correlation_features(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop based on correlation threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    X_filtered = X.drop(X[to_drop], axis=1)
    return X_filtered

# Step 5: Recursive Feature Elimination (RFE)
def rfe_selection(X, y, model):
    rfe = RFE(model, n_features_to_select=1)  # Select the most important feature
    rfe.fit(X, y)
    return rfe.support_

# Main function for UniSelect
def UniSelect(X, y, k1, k2):
    # Step 1: Chi-Square feature selection
    selected_features_chi2 = chi_square_selection(X, y, k1)
    
    # Step 2: Feature Importance using ExtraTreesClassifier
    selected_features_importance = feature_importance_selection(X, y, k2)
    
    # Combine the two feature sets (taking union of selected features)
    selected_features = np.union1d(selected_features_chi2, selected_features_importance)
    
    # Step 3: Custom scoring metric for feature selection
    variance_scores = custom_scoring_metric(X)
    custom_scored_features = np.argsort(variance_scores)[::-1][:len(selected_features)]  # Select top features
    
    # Step 4: Remove highly correlated features
    X_filtered = remove_high_correlation_features(X)
    
    # Step 5: Train a base classifier using Recursive Feature Elimination (RFE)
    model = LogisticRegression(max_iter=1000)
    selected_rfe_features = rfe_selection(X_filtered.iloc[:, selected_features], y, model)
    
    # Final set of selected features (modified union)
    final_selected_features = np.where(selected_rfe_features)[0]
    
    # Return the final set of features
    return X_filtered.iloc[:, final_selected_features], final_selected_features

# Example usage with Iris dataset
from sklearn.datasets import load_iris

# Load the dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Preprocessing: Encode labels if necessary
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Set the number of top-k features to select
k1 = 2  # Number of features based on Chi-Square
k2 = 2  # Number of features based on Feature Importance

# Apply the UniSelect feature selection
X_selected, selected_features = UniSelect(X, y_encoded, k1, k2)

# Train a classifier on the selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a classifier (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Selected features: {selected_features}")
print(f"Model Accuracy: {accuracy}")
