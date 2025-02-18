# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from mmfw_rfe import MMFW_RFE  # Import the MMFW-RFE algorithm

# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with actual dataset path
X = df.drop(columns=["target"])  # Features
y = df["target"]  # Target variable

# Step 1: Partition the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Apply MMFW-RFE to obtain prime attributes (PA) and feature subset optimization index (FSOI)
mmfw_rfe = MMFW_RFE(n_features_to_select=10)  # Selecting top 10 features
X_train_selected = mmfw_rfe.fit_transform(X_train, y_train)
X_test_selected = mmfw_rfe.transform(X_test)

# Get selected features
PA = mmfw_rfe.selected_features
FSOI = mmfw_rfe.optimization_index  # Assuming MMFW_RFE provides an optimization index

# Step 3: Train weak classifiers (Decision Tree, Naive Bayes, SVM)
dt = DecisionTreeClassifier(random_state=42)
nb = GaussianNB()
svm = SVC(probability=True, random_state=42)

dt.fit(X_train_selected, y_train)
nb.fit(X_train_selected, y_train)
svm.fit(X_train_selected, y_train)

# Step 4: Initialize an empty list to store predictions
predictions = []

# Step 5 & 6: Get predictions from individual classifiers and apply the Voting Classifier
ensemble_predictions = []

for i in range(X_test_selected.shape[0]):
    dt_pred = dt.predict([X_test_selected[i]])[0]
    nb_pred = nb.predict([X_test_selected[i]])[0]
    svm_pred = svm.predict([X_test_selected[i]])[0]
    
    # Store predictions
    predictions.append([dt_pred, nb_pred, svm_pred])
    
    # Compute weights (here we use uniform voting, but can be adjusted based on classifier confidence)
    w_dt = 1  # Weighting function can be modified
    w_nb = 1
    w_svm = 1

    # Voting mechanism (majority voting)
    final_pred = max(set([dt_pred, nb_pred, svm_pred]), key=[dt_pred, nb_pred, svm_pred].count)
    ensemble_predictions.append(final_pred)

# Step 7: Calculate accuracy and F-score
accuracy = accuracy_score(y_test, ensemble_predictions)
fscore = f1_score(y_test, ensemble_predictions, average="weighted")

# Step 8 & 9: Return final results
print("Prime Attributes (PA):", PA)
print("Feature Subset Optimization Index (FSOI):", FSOI)
print("Accuracy of Voting Classifier:", accuracy)
print("F-score of Voting Classifier:", fscore)





Key Features of This Implementation:
✅ Applies MMFW-RFE to select optimal features
✅ Trains three weak classifiers: Decision Tree, Naive Bayes, and SVM
✅ Uses Voting Classifier for ensemble predictions
✅ Calculates accuracy and F-score for evaluation
✅ Handles feature selection dynamically

Make sure mmfw_rfe.py is correctly implemented in the same directory.

