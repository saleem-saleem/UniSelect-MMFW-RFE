<h1>Integrated Feature Selection and Optimization Framework for Predicting Student Learning Effectiveness and Performance</h1>
<p><b>Overview</b></p>
<p>This repository presents an integrated framework for feature selection and optimization aimed at enhancing student performance prediction. The proposed approach, UniSelect, combines filter and wrapper-based methods to select the most relevant features while eliminating redundancy, improving classification accuracy, and optimizing computational efficiency</p>
<p><b>Contributions</b></p>
<p><ol>UniSelect Algorithm: A hybrid feature selection method that integrates Chi-square and Feature Importance ranking techniques with Recursive Feature Elimination (RFE).</ol></p>
<p><ol>MMFW-RFE: An advanced optimization technique that refines UniSelect’s selected features using Reduced Recursive Feature Elimination and Cross-Validation-based Recursive Feature Elimination.</ol></p>
<p><ol>EMPA-FWO-VCA: An ensemble voting classifier that leverages Decision Tree (DT), Naïve Bayes (NB), and Support Vector Machine (SVM) for improved classification accuracy.</ol></p>
<p><b>Applications</b></p>
<p><ol>Student Performance Prediction – Identifies at-risk students and improves academic interventions to enhance learning outcomes.</ol></p>
<p><ol>Adaptive Learning Systems – Personalizes e-learning experiences by recommending tailored study materials and strategies.</ol></p>
<p><ol>Career Guidance & University Admissions – Assists in evaluating student potential for university admissions and career recommendations.</ol></p>
<p><ol>Educational Policy & Institutional Planning – Supports data-driven decision-making for curriculum design and resource allocation.</ol></p>
<p><ol>Workforce Development & Skill Training – Optimizes corporate training programs by predicting employee learning effectiveness.</ol></p>
<p><b>Prerequisites for Using the Framework</b></p>
<p><ol>Python Environment

Python 3.7 or later
Recommended: Anaconda or a virtual environment (venv)</ol></p>

<p><ol>Required Libraries 

NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn</ol></p>

<p><ol>Dataset

The framework is designed to work with student performance datasets.
Ensure the dataset is in CSV format and contains relevant features like demographic, academic, and behavioral attributes.</ol></p>


<p><ol>Machine Learning Frameworks

Scikit-learn for feature selection and classification algorithms
TensorFlow/Keras (optional) for deep learning-based extensions</ol></p>




<p><ol>System Requirements

Minimum: 8GB RAM, Dual-Core CPU
Recommended: 16GB RAM, GPU for faster computations</ol></p>


<p><b>UniSelect Algorithm Workflow</b></p>
<P><ol>Input Data Preparation
  
Load dataset with n features and a target variable.
Define k1 (Chi-Square top-k features) and k2 (Feature Importance top-k features)</ol></P>
<p><ol>Filter-Based Feature Selection

Chi-Square Test: Selects k1 most relevant features (C).
Extra Trees Classifier (ETC): Computes feature importance using MDI & MDA, selecting k2 top features (I).</ol></p>
<p><ol>Feature Combination & Refinement

Compute custom scoring metrics (S).
Identify and remove highly correlated features (R).
Form the Modified Union Set = C ∪ I ∪ S ∩ R.</ol></p>
<p><ol>Wrapper-Based Optimization

Apply Recursive Feature Elimination (RFE) to refine the selected feature subset.</ol></p>
<p><ol>Final Model Training & Prediction

Train a classifier (SVM, DT, NB, etc.) on optimized features.
Predict student performance with improved accuracy.</ol></p>


<p><b>Workflow of MMFW-RFE Algorithm</b></p>
<P><ol>Input Data Preparation
  
Load dataset with n features (A) and class labels (C).
Define k (number of features to select) and n (features to eliminate per iteration).
Choose a classification algorithm (B) for evaluation. </ol></P>


<p><ol>Initial Feature Selection

Select k best features using Chi-Square Test (FS1).
Select k best features using Feature Importance (ETC) (FS2).
Combine both feature sets into a Modified Union Set (FS).</ol></p>


<p><ol>Feature Subset Optimization

If FS has more or fewer than k features, adjust accordingly.
Apply Reduced Recursive Feature Elimination (RRFE) to remove redundant features.
Evaluate performance using Accuracy and F-score.
Apply Cross-Validated RFE (CV-RFE) with k-fold validation.</ol></p>


<p><ol>Final Feature Selection

Refine RRFE and CV-RFE subsets using Chi-Square filtering.
Compute Feature Subset Optimization Index (FSOI).</ol></p>


<p><ol>Best Feature Subset Selection

Compare RRFE and CV-RFE results.
Select the optimal feature subset with the highest FSOI, Accuracy, and F-score.
</ol></p>


<p><ol>Output

Return Prime Attributes (PA), FSOI, Accuracy, and F-score for final classification.
</ol></p>




<p><b>Workflow of EMPA-FWO-VCA Algorithm </b></p>
<P><ol>Dataset Partitioning
  
Split dataset into training set (A_train) and testing set (A_test). </ol></P>


<p><ol>Feature Selection & Optimization

Apply MMFW-RFE (Algorithm 2) on A_train to obtain Prime Attributes (PA) and Feature Subset Optimization Index (FSOI).</ol></p>


<p><ol>Training Weak Classifiers

Train Decision Tree (DT), Naïve Bayes (NB), and Support Vector Machine (SVM) on A_train using PA.</ol></p>


<p><ol>Voting Classifier for Final Prediction

Use a weighted voting mechanism to combine predictions from all classifiers..</ol></p>


<p><ol>Performance Evaluation

Calculate Accuracy and F-score for the ensemble model.
</ol></p>


Scripts & Their Purpose
1. EMPA-FWO-VCA.py
📌 Purpose: Implements the Ensemble Model with Prime Attributes and Filter-Wrapper Optimization using Voting Classifier Algorithm (EMPA-FWO-VCA).
🚀 Run the script:


python EMPA-FWO-VCA.py
2. EMPA_FWO_VCA_Analysis.ipynb
📌 Purpose: Jupyter Notebook for analyzing the EMPA-FWO-VCA ensemble model’s performance and feature impact.
🚀 Run the notebook:


jupyter notebook EMPA_FWO_VCA_Analysis.ipynb
3. Feature_Subset_Optimization_Index.py
📌 Purpose: Computes the Feature Subset Optimization Index (FSOI) to evaluate selected features' quality.
🚀 Run the script:


python Feature_Subset_Optimization_Index.py
4. MMFW-RFE.py
📌 Purpose: Implements the Multi-Method Filter-Wrapper with Recursive Feature Elimination (MMFW-RFE) for feature selection and optimization.
🚀 Run the script:


python MMFW-RFE.py
5. MMFW_RFE_Optimization.ipynb
📌 Purpose: Jupyter Notebook for visualizing the MMFW-RFE feature selection process and its impact on classification performance.
🚀 Run the notebook:


jupyter notebook MMFW_RFE_Optimization.ipynb
6. UniSelect_Analysis.ipynb
📌 Purpose: Notebook for analyzing UniSelect, a hybrid feature selection algorithm combining Chi-Square and Feature Importance methods.
🚀 Run the notebook:


jupyter notebook UniSelect_Analysis.ipynb
7. binary_classification.py
📌 Purpose: Performs binary classification (pass/fail prediction) using selected features.
🚀 Run the script:

python binary_classification.py
8. multiclassification.py
📌 Purpose: Implements multi-class classification for student performance prediction using optimized features.
🚀 Run the script:


python multiclassification.py
9. data_preprocessing.py
📌 Purpose: Cleans and preprocesses dataset, including handling missing values, encoding categorical data, and normalizing features.
🚀 Run the script:

python data_preprocessing.py
10. feature_selection.py
📌 Purpose: Selects the most relevant features using UniSelect, MMFW-RFE, and Feature Importance techniques.
🚀 Run the script:


python feature_selection.py
11. model_training.py
📌 Purpose: Trains machine learning models (Decision Tree, SVM, Naïve Bayes, etc.) using optimized feature subsets.
🚀 Run the script:


python model_training.py
12. ensemble_voting_classifier.py
📌 Purpose: Implements an ensemble learning approach using Voting Classifier for improved prediction accuracy.
🚀 Run the script:

python ensemble_voting_classifier.py
13. uniselect.py
📌 Purpose: Implements UniSelect, a hybrid feature selection algorithm combining filter and wrapper-based methods.
🚀 Run the script:

python uniselect.py
How to Run the Full Pipeline
To execute the full feature selection, model training, and prediction pipeline, run:

python pipeline.py





<p><ol>Output

Return final predictions, Accuracy, F-score, PA, and FSOI.
</ol></p>



