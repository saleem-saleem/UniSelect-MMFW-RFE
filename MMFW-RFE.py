# Required Libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Step 1: Select top-k features using Chi-Square
def SelectTopKFeaturesUsingChiSquare(X, y, k):
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.get_support(indices=True)

# Step 2: Select top-k features based on feature importance
def SelectTopKFeaturesUsingFeatureImportance(X, y, k, classifier=RandomForestClassifier()):
    classifier.fit(X, y)
    feature_importances = classifier.feature_importances_
    indices = np.argsort(feature_importances)[-k:]  # Top k important features
    X_selected = X[:, indices]
    return X_selected, indices

# Step 3: Perform the union of selected features
def ModifiedUnion(FS1, FS2):
    return np.unique(np.concatenate([FS1, FS2]))

# Step 4: Reduced Recursive Feature Elimination (RRFE)
def ReducedRecursiveFeatureElimination(X, y, n, classifier=RandomForestClassifier()):
    # Initializing feature set
    selected_features = np.arange(X.shape[1])
    while len(selected_features) > n:
        classifier.fit(X[:, selected_features], y)
        feature_importances = classifier.feature_importances_
        least_important = np.argsort(feature_importances)[:n]
        selected_features = np.delete(selected_features, least_important)
    return selected_features

# Step 5: Evaluate classifier performance (accuracy and F-score)
def EvaluateClassifierPerformance(classifier, X, y):
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    fscore = f1_score(y, y_pred, average='weighted')
    return accuracy, fscore

# Step 6: Cross-Validated Recursive Feature Elimination (CV-RRFE)
def CrossValidatedRecursiveFeatureElimination(X, y, n, classifier=RandomForestClassifier(), cv=5):
    selected_features = np.arange(X.shape[1])
    while len(selected_features) > n:
        classifier.fit(X[:, selected_features], y)
        feature_importances = classifier.feature_importances_
        least_important = np.argsort(feature_importances)[:n]
        selected_features = np.delete(selected_features, least_important)
        
        # Perform cross-validation and calculate average score
        scores = cross_val_score(classifier, X[:, selected_features], y, cv=cv, scoring='accuracy')
        avg_score = np.mean(scores)
    return selected_features, avg_score

# Step 7: Main MMFW-RFE function
def MMFW_RFE(X, y, k, n, classifier=RandomForestClassifier()):
    # Step 1: Feature selection using Chi-Square and Feature Importance
    FS1, _ = SelectTopKFeaturesUsingChiSquare(X, y, k)
    FS2, _ = SelectTopKFeaturesUsingFeatureImportance(X, y, k)
    
    # Step 2: Modify Union of the selected features
    FS = ModifiedUnion(FS1, FS2)
    k1 = len(FS)
    if k1 < k:
        print(f"Warning: the number of selected features is less than k. Proceeding with k1 = {k1}.")
        k1 = k
    elif k1 > k:
        print(f"Warning: the number of selected features is greater than k. Proceeding with k1 = {k}.")
        FS = FS[:k]
        k1 = k
    
    # Step 3: Reduced Recursive Feature Elimination (RRFE)
    RRFE_PA = ReducedRecursiveFeatureElimination(X[:, FS], y, n, classifier)
    classifier.fit(X[:, RRFE_PA], y)
    RRFE_Accuracy, RRFE_Fscore = EvaluateClassifierPerformance(classifier, X[:, RRFE_PA], y)
    
    # Step 4: Cross-Validated Recursive Feature Elimination (CV-RRFE)
    CV_RRFE_PA, CV_RRFE_Accuracy = CrossValidatedRecursiveFeatureElimination(X[:, FS], y, n, classifier)
    classifier.fit(X[:, CV_RRFE_PA], y)
    CV_RRFE_Fscore = f1_score(y, classifier.predict(X[:, CV_RRFE_PA]), average='weighted')
    
    # Step 5: Evaluate Feature Subset Quality
    RRFE_Filter_PA, _ = SelectTopKFeaturesUsingChiSquare(X[:, RRFE_PA], y, k)
    CV_RRFE_Filter_PA, _ = SelectTopKFeaturesUsingChiSquare(X[:, CV_RRFE_PA], y, k)
    
    RRFE_Filter_FSOI = EvaluateClassifierPerformance(classifier, X[:, RRFE_Filter_PA], y)
    CV_RRFE_Filter_FSOI = EvaluateClassifierPerformance(classifier, X[:, CV_RRFE_Filter_PA], y)
    
    # Step 6: Determine the best feature subset
    if RRFE_Fscore >= CV_RRFE_Fscore and RRFE_Filter_FSOI[1] >= CV_RRFE_Filter_FSOI[1]:
        PA = RRFE_Filter_PA
        FSOI = RRFE_Filter_FSOI[1]
        Accuracy = RRFE_Accuracy
        Fscore = RRFE_Fscore
    else:
        PA = CV_RRFE_Filter_PA
        FSOI = CV_RRFE_Filter_FSOI[1]
        Accuracy = CV_RRFE_Accuracy
        Fscore = CV_RRFE_Fscore
    
    return PA, FSOI, Accuracy, Fscore
       
