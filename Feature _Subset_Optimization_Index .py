import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_importance import mutual_info_classif
from sklearn.model_selection import cross_val_score

# Load the dataset (example: using the Portuguese dataset)
# Assuming you have a CSV file with the dataset
data = pd.read_csv('portuguese_dataset.csv')

# Preprocessing: separate features and target
X = data.drop(columns=['target'])  # Replace 'target' with your target column name
y = data['target']  # Replace 'target' with your target column name

# Encode the target variable if it's categorical
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Chi-square feature selection
chi_selector = SelectKBest(chi2, k=10)  # Select top 10 features based on chi-square test
X_chi = chi_selector.fit_transform(X_train, y_train)
selected_features_chi = X.columns[chi_selector.get_support()]
print(f"Chi-square selected features: {selected_features_chi}")

# Apply Feature Importance (using Random Forest)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
feature_importance = rf_classifier.feature_importances_
sorted_idx = feature_importance.argsort()
selected_features_fi = X.columns[sorted_idx[-10:]]  # Top 10 features
print(f"Feature Importance selected features: {selected_features_fi}")

# Apply Recursive Feature Elimination (RFE)
svc = SVC(kernel="linear")
rfe_selector = RFE(svc, n_features_to_select=10)
X_rfe = rfe_selector.fit_transform(X_train, y_train)
selected_features_rfe = X.columns[rfe_selector.get_support()]
print(f"RFE selected features: {selected_features_rfe}")

# Apply Recursive Feature Elimination with Cross-validation (CV-RFE)
from sklearn.feature_selection import RFECV
rfecv_selector = RFECV(estimator=svc, step=1, cv=5)
X_rfecv = rfecv_selector.fit_transform(X_train, y_train)
selected_features_rfecv = X.columns[rfecv_selector.support_]
print(f"CV-RFE selected features: {selected_features_rfecv}")

# Calculate FSOI (Feature Subset Optimization Index)
def calculate_fsoi(selected_features, total_features):
    return len(selected_features) / total_features

# FSOI values
fsoi_chi = calculate_fsoi(selected_features_chi, len(X.columns))
fsoi_fi = calculate_fsoi(selected_features_fi, len(X.columns))
fsoi_rfe = calculate_fsoi(selected_features_rfe, len(X.columns))
fsoi_rfecv = calculate_fsoi(selected_features_rfecv, len(X.columns))

print(f"FSOI for Chi-square: {fsoi_chi}")
print(f"FSOI for Feature Importance: {fsoi_fi}")
print(f"FSOI for RFE: {fsoi_rfe}")
print(f"FSOI for CV-RFE: {fsoi_rfecv}")

# Train and evaluate classifiers (e.g., Decision Tree, SVM)
def evaluate_classifier(X_train, X_test, y_train, y_test, selected_features):
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # Classifier (Decision Tree example)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_selected, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, f_score

# Evaluate classifiers on different feature sets
accuracy_chi, f_score_chi = evaluate_classifier(X_train, X_test, y_train, y_test, selected_features_chi)
accuracy_fi, f_score_fi = evaluate_classifier(X_train, X_test, y_train, y_test, selected_features_fi)
accuracy_rfe, f_score_rfe = evaluate_classifier(X_train, X_test, y_train, y_test, selected_features_rfe)
accuracy_rfecv, f_score_rfecv = evaluate_classifier(X_train, X_test, y_train, y_test, selected_features_rfecv)

print(f"Chi-square Classifier Accuracy: {accuracy_chi}, F1 Score: {f_score_chi}")
print(f"Feature Importance Classifier Accuracy: {accuracy_fi}, F1 Score: {f_score_fi}")
print(f"RFE Classifier Accuracy: {accuracy_rfe}, F1 Score: {f_score_rfe}")
print(f"CV-RFE Classifier Accuracy: {accuracy_rfecv}, F1 Score: {f_score_rfecv}")

# You can also compare using other models or cross-validation for more robust results
