Cybersecurity Incident Triage Classification

Overview:

   This project aims to improve the efficiency of Microsoft’s Security Operation Centers (SOCs) by developing a machine learning model that predicts the triage grade of cybersecurity incidents. The model categorizes incidents as True Positive (TP), Benign Positive (BP), or False Positive (FP) based on historical data, assisting    SOC analysts in prioritizing responses effectively.

*Problem Statement:

  The task is to build a robust classification model using the GUIDE dataset to categorize incidents into TP, BP, or FP.   This model will support SOC analysts with precise recommendations to strengthen enterprise security. Evaluation metrics include macro-F1 score, precision, and recall, ensuring reliability for real-world use.

*Packages Used:

#The following packages are required:

   * pandas, numpy: Data manipulation
   * scikit-learn: Model training and evaluation
   * imblearn: SMOTE for class balancing
   * xgboost: XGBoost model implementation
   * matplotlib, seaborn: Data visualization

*Steps:

 Step 1- Data Preprocessing: Handle missing values, encode categorical features, remove outliers, and normalize data.
 Step 2- Data Splitting: Divide the train dataset into train set and validation set.
 Step 3- Class Balancing: Use SMOTE for balancing classes.
 Step 4- Model Training: Train multiple models (Random Forest, XGBoost, Decision Tree), with cross-validation and hyperparameter tuning.
 Step 5- Evaluation: Select the Best Model based on metrics (accuracy, precision, recall, F1 score, AUC) on the validation set.
 Step 6- Test Set Evaluation: Evaluate the selected model on the test set and compare its performance with validation metrics to ensure consistency.

*Results:

  The model is chosen based on performance across macro-F1, precision, and recall, ensuring it supports accurate and efficient triage for SOC operations.

