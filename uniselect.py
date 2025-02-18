import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import ExtraTreesClassifier

class UniSelect:
    def __init__(self, k1=10, k2=15):
        """
        Initializes the UniSelect feature selection process.
        
        :param k1: Number of top features selected using Chi-Square.
        :param k2: Number of top features selected using Feature Importance.
        """
        self.k1 = k1
        self.k2 = k2
        self.chi2_selected = None
        self.fif_selected = None
        self.selected_features = None

    def fit(self, X, y):
        """
        Applies Chi-Square and Feature Importance selection.
        
        :param X: Feature matrix (pandas DataFrame).
        :param y: Target labels (pandas Series or numpy array).
        """
        # Step 1: Chi-Square Feature Selection
        chi2_selector = SelectKBest(chi2, k=self.k1)
        chi2_selector.fit(X, y)
        chi2_features = X.columns[chi2_selector.get_support()]
        self.chi2_selected = set(chi2_features)

        # Step 2: Feature Importance via Extra Trees Classifier
        etc = ExtraTreesClassifier(n_estimators=100, random_state=42)
        etc.fit(X, y)
        feature_importances = pd.Series(etc.feature_importances_, index=X.columns)
        fif_features = feature_importances.nlargest(self.k2).index
        self.fif_selected = set(fif_features)

        # Step 3: Create Modified Union Set (CMF)
        self.selected_features = self.chi2_selected.union(self.fif_selected)

    def transform(self, X):
        """
        Transforms the dataset by selecting only the chosen features.
        
        :param X: Feature matrix (pandas DataFrame).
        :return: Transformed DataFrame with selected features.
        """
        if self.selected_features is None:
            raise ValueError("UniSelect has not been fitted yet. Call fit() first.")
        return X[list(self.selected_features)]

    def fit_transform(self, X, y):
        """
        Fits the selector and transforms the dataset.
        :param X: Feature matrix.
        :param y: Target labels.
        :return: Transformed dataset with selected features.
        """
        self.fit(X, y)
        return self.transform(X)





from uniselect import UniSelect
import pandas as pd

# Load dataset (replace with actual data)
df = pd.read_csv("dataset.csv")
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target variable

# Initialize and apply UniSelect
selector = UniSelect(k1=10, k2=15)
X_selected = selector.fit_transform(X, y)

print("Selected Features:", list(selector.selected_features))
