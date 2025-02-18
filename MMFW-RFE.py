import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

class MMFW_RFE:
    def __init__(self, estimator=None, n_features_to_select=10):
        """
        Initializes the MMFW-RFE feature selection process.

        :param estimator: The base model for RFE (default: RandomForestClassifier).
        :param n_features_to_select: Number of features to retain.
        """
        self.n_features_to_select = n_features_to_select
        self.estimator = estimator if estimator else RandomForestClassifier(n_estimators=100, random_state=42)
        self.rfe_selected = None

    def fit(self, X, y):
        """
        Applies Recursive Feature Elimination (RFE) for feature selection.

        :param X: Feature matrix (pandas DataFrame).
        :param y: Target labels (pandas Series or numpy array).
        """
        # Step 1: Perform Recursive Feature Elimination (RFE)
        rfe = RFE(self.estimator, n_features_to_select=self.n_features_to_select)
        rfe.fit(X, y)
        
        # Step 2: Store selected feature names
        self.rfe_selected = set(X.columns[rfe.support_])

    def transform(self, X):
        """
        Transforms the dataset by selecting only the chosen features.

        :param X: Feature matrix (pandas DataFrame).
        :return: Transformed DataFrame with selected features.
        """
        if self.rfe_selected is None:
            raise ValueError("MMFW_RFE has not been fitted yet. Call fit() first.")
        return X[list(self.rfe_selected)]

    def fit_transform(self, X, y):
        """
        Fits the selector and transforms the dataset.

        :param X: Feature matrix.
        :param y: Target labels.
        :return: Transformed dataset with selected features.
        """
        self.fit(X, y)
        return self.transform(X)




from mmfw_rfe import MMFW_RFE
import pandas as pd

# Load dataset (replace with actual data)
df = pd.read_csv("dataset.csv")
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target variable

# Initialize and apply MMFW-RFE
selector = MMFW_RFE(n_features_to_select=10)
X_selected = selector.fit_transform(X, y)

print("Selected Features:", list(selector.rfe_selected))
