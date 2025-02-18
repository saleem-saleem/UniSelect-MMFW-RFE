import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset (replace with your dataset)
# Assuming the dataset has columns 'features' and 'target'
df = pd.read_csv('your_dataset.csv')

# Reclassify the grades into Pass/Fail
df['target'] = df['target'].apply(lambda x: 1 if x >= 10 else 0)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier()
}

# Feature selection methods
feature_selectors = {
    'CSF': SelectKBest(chi2, k=10),
    'FIF': SelectKBest(chi2, k=10),  # Can be replaced with feature importance method
    'CMF': SelectKBest(chi2, k=10),  # Can be replaced with a combination of methods
    'RRFE': RFE(DecisionTreeClassifier(), n_features_to_select=10),
    'CV-RFE': RFE(DecisionTreeClassifier(), n_features_to_select=10)  # Cross-validation RFE
}

# Function to evaluate classifiers with and without feature selection
def evaluate_classifiers(X_train, X_test, y_train, y_test, classifiers, feature_selectors):
    results = {}

    # Evaluate without feature selection
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)
        results[clf_name] = {'Accuracy': accuracy, 'F-Score': f_score}

    # Evaluate with feature selection
    for selector_name, selector in feature_selectors.items():
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_selected, y_train)
            y_pred = clf.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            f_score = f1_score(y_test, y_pred)
            results[f'{clf_name} - {selector_name}'] = {'Accuracy': accuracy, 'F-Score': f_score}

    return results

# Run evaluation
evaluation_results = evaluate_classifiers(X_train, X_test, y_train, y_test, classifiers, feature_selectors)

# Convert the results to a DataFrame for easier readability
results_df = pd.DataFrame(evaluation_results).T
print(results_df)
