from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def chi_square_selection(X, y, k):
    """Select top-k features using Chi-Square test."""
    chi2_selector = SelectKBest(chi2, k=k)
    chi2_selector.fit(X, y)
    selected_features = chi2_selector.get_support(indices=True)
    return selected_features

def feature_importance_selection(X, y, k):
    """Select top-k features using ExtraTreesClassifier feature importance."""
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X, y)
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()[::-1]
    return sorted_idx[:k]

def rfe_selection(X, y, k):
    """Perform Recursive Feature Elimination (RFE)."""
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    return rfe.support_
