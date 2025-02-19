import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# Step 1: Load Dataset
def load_data(file_path):
    """Load dataset from CSV file."""
    df = pd.read_csv(file_path)
    return df

# Step 2: Data Preprocessing
def preprocess_data(df, target_column):
    """Encode categorical variables, scale features, and split dataset."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Standardize numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Feature Selection (UniSelect)
def feature_selection_uniselect(X_train, y_train, k=10):
    """Select top-k features using Chi-Square & Feature Importance."""
    chi_selector = SelectKBest(chi2, k=k)
    chi_selector.fit(X_train, y_train)
    chi_features = X_train.columns[chi_selector.get_support()]

    etc = ExtraTreesClassifier(n_estimators=50)
    etc.fit(X_train, y_train)
    etc_features = X_train.columns[np.argsort(etc.feature_importances_)[-k:]]

    selected_features = list(set(chi_features) | set(etc_features))  # Union of features
    return X_train[selected_features], selected_features

# Step 4: Feature Optimization (MMFW-RFE)
def feature_optimization_rfe(X_train, y_train, selected_features, n_features=10):
    """Perform Recursive Feature Elimination (RFE) to refine feature selection."""
    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X_train[selected_features], y_train)
    
    optimized_features = X_train[selected_features].columns[rfe.support_]
    return X_train[optimized_features], optimized_features

# Step 5: Model Training & Evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test, features):
    """Train models and evaluate performance."""
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
    }

    results = {}

    for name, model in classifiers.items():
        model.fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = {"Accuracy": acc, "F1 Score": f1}

    # Voting Classifier (Ensemble)
    voting_clf = VotingClassifier(
        estimators=[("DT", classifiers["Decision Tree"]), 
                    ("SVM", classifiers["SVM"]), 
                    ("NB", classifiers["Naive Bayes"])], 
        voting='soft'
    )
    voting_clf.fit(X_train[features], y_train)
    y_pred_ensemble = voting_clf.predict(X_test[features])
    
    results["Voting Classifier"] = {
        "Accuracy": accuracy_score(y_test, y_pred_ensemble),
        "F1 Score": f1_score(y_test, y_pred_ensemble, average='weighted')
    }

    return results

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("student_performance.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column="Performance")

    # Feature Selection
    X_train_selected, selected_features = feature_selection_uniselect(X_train, y_train, k=10)

    # Feature Optimization
    X_train_optimized, optimized_features = feature_optimization_rfe(X_train_selected, y_train, selected_features, n_features=5)

    # Train and evaluate models
    results = train_and_evaluate(X_train_optimized, X_test, y_train, y_test, optimized_features)

    # Print Results
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
