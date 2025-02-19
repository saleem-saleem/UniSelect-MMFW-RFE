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









<p><ol>Output

Return final predictions, Accuracy, F-score, PA, and FSOI.
</ol></p>

<p><b> How to Run </b></p>
<p>Clone the Repository</p>
<p>Install Dependencies</p>
<p>Ensure Dataset is Available</p>
<p>Run Scripts Individually</p>
<p>Run the Full Pipeline (Automates All Steps)</p>
<p>Run Jupyter Notebooks for Analysis</p>


