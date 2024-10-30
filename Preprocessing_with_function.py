import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint, uniform


def preprocess_data(df, is_train=True):
    # Fill missing values in 'IncidentGrade' column by mode if in training set
    if is_train:
        df['IncidentGrade'] = df['IncidentGrade'].fillna(df['IncidentGrade'].mode()[0])

    # Drop columns with more than 50% missing values
    threshold = len(df) * 0.5
    df.dropna(thresh=threshold, axis=1, inplace=True)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Check and handle missing values
    missing_values = df.isnull().sum()
    print("Missing Values per Column:\n", missing_values[missing_values > 0])

    # Convert 'Timestamp' to datetime and extract time features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['Hour'] = df['Timestamp'].dt.hour
    df['Time'] = df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second

    # Drop original timestamp after extracting time features
    df.drop('Timestamp', axis=1, inplace=True)

    # Encode categorical columns
    categorical_columns = ['IncidentId', 'AlertId', 'DetectorId', 'AlertTitle', 'Category', 'IncidentGrade',
                           'EntityType', 'Usage',
                           'EvidenceRole', 'OrgId', 'DeviceId', 'Sha256', 'IpAddress', 'Url', 'AccountSid',
                           'AccountUpn', 'AccountObjectId', 'AccountName', 'DeviceName', 'NetworkMessageId',
                           'RegistryKey', 'RegistryValueName', 'RegistryValueData', 'ApplicationId', 'ApplicationName',
                           'OAuthApplicationId', 'FileName', 'FolderPath', 'ResourceIdName', 'OSFamily', 'OSVersion',
                           'CountryCode', 'State', 'City']

    label_encoder = LabelEncoder()
    for column in categorical_columns:
        if column in df.columns:
            df[column] = label_encoder.fit_transform(df[column].astype(str))

    # Identify and remove outliers in numeric columns based on IQR
    numeric_columns = ['Day', 'Month', 'Year', 'Time', 'Hour']
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect and remove outliers
    outliers = ((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)
    df = df[~outliers]
    print(f"Number of rows after removing outliers: {df.shape[0]}")

    # Apply MinMaxScaler normalization to numeric columns
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Generate correlation matrix and select features with high correlation to target variable
    corr_matrix = df.corr()
    plt.figure(figsize=(25, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap between Numeric Columns and IncidentGrade')
    plt.show()

    # Feature selection based on correlation threshold
    #if 'IncidentGrade' in df.columns:
    corr_target = abs(corr_matrix['IncidentGrade'])
    relevant_features = corr_target[corr_target > 0.15].index
    df = df[relevant_features]
    print(f"Selected Features based on correlation threshold: {relevant_features}")

    return df


# Usage on Train and Test Datasets
train_data = pd.read_csv("Sample_Train.csv")
test_data = pd.read_csv("Sample_Test.csv")

train_data = preprocess_data(train_data, is_train=True)
test_data = preprocess_data(test_data, is_train=False)


# Split dataset into train and validation sets
X = train_data.drop(columns=['IncidentGrade'])  # Features
y = train_data['IncidentGrade']  # Target

# Perform stratified split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled = X_train_resampled.astype('float32')
y_train_resampled = y_train_resampled.astype('int32')


# Step 1: Random Forest Classifier

# Initialize Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the Random Forest model
rf_model.fit(X_train_resampled, y_train_resampled)

# Make predictions with Random Forest
y_pred_rf = rf_model.predict(X_val)
y_pred_proba_rf = rf_model.predict_proba(X_val)

# Calculate metrics for Random Forest
accuracy_rf = accuracy_score(y_val, y_pred_rf)
precision_rf = precision_score(y_val, y_pred_rf, average='weighted')
recall_rf = recall_score(y_val, y_pred_rf, average='weighted')
f1_rf = f1_score(y_val, y_pred_rf, average='weighted')
roc_auc_rf = roc_auc_score(y_val, y_pred_proba_rf, multi_class='ovr')

# Print Random Forest metrics
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1 Score: {f1_rf:.2f}")
print(f"AUC: {roc_auc_rf:.2f}\n")

# RandomizedSearchCV for Random Forest
param_dist_rf = {
    'n_estimators': randint(100, 500),       # Randomly sample n_estimators from 100 to 500
    'max_depth': [10, 20, None],             # Randomly choose between 10, 20, or None for max depth
    'min_samples_split': randint(2, 10),     # Randomly sample min_samples_split between 2 and 10
    'min_samples_leaf': randint(1, 4),       # Randomly sample min_samples_leaf between 1 and 4
    'bootstrap': [True, False]               # Bootstrap sampling or not
}

# Use RandomizedSearchCV for faster tuning
# Setting up RandomizedSearchCV with error_score set to 'raise'
rf_random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist_rf,
    n_iter=10, cv=3, scoring='accuracy',
    error_score='raise',  # Halt and show error on failure
    random_state=42, n_jobs=-1
)

# Fit RandomizedSearchCV with exception handling
try:
    rf_random_search.fit(X_train_resampled, y_train_resampled)

    # Print best parameters and cross-validation score only if fitting was successful
    print("Best Parameters for Random Forest:", rf_random_search.best_params_)
    print("Best Cross-Validation Accuracy for Random Forest:", rf_random_search.best_score_)
except Exception as e:
    print("An error occurred during hyperparameter tuning:", e)

# Step 2: XGBoost Classifier

# Initialize XGBoost classifier
xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')

# Train the XGBoost model
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions with XGBoost
y_pred_xgb = xgb_model.predict(X_val)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)

# Calculate metrics for XGBoost
accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
precision_xgb = precision_score(y_val, y_pred_xgb, average='weighted')
recall_xgb = recall_score(y_val, y_pred_xgb, average='weighted')
f1_xgb = f1_score(y_val, y_pred_xgb, average='weighted')
roc_auc_xgb = roc_auc_score(y_val, y_pred_proba_xgb, multi_class='ovr')

# Print XGBoost metrics
print("XGBoost Classifier:")
print(f"Accuracy: {accuracy_xgb:.2f}")
print(f"Precision: {precision_xgb:.2f}")
print(f"Recall: {recall_xgb:.2f}")
print(f"F1 Score: {f1_xgb:.2f}")
print(f"AUC: {roc_auc_xgb:.2f}\n")

# RandomizedSearchCV for XGBoost
param_dist_xgb = {
    'n_estimators': randint(100, 300),          # Randomly sample n_estimators between 100 and 300
    'max_depth': randint(3, 10),                # Randomly sample max_depth between 3 and 10
    'learning_rate': uniform(0.01, 0.2),        # Randomly sample learning_rate between 0.01 and 0.2
    'subsample': uniform(0.8, 0.2),             # Ensure subsample ratio between 0.8 and 1.0
    'colsample_bytree': uniform(0.8, 0.2)       # Ensure colsample_bytree ratio between 0.8 and 1.0
}


# Use RandomizedSearchCV for faster tuning
xgb_random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb,
                                       n_iter=10, cv=3, scoring='accuracy',
                                       random_state=42, n_jobs=-1)

# Fit RandomizedSearchCV
xgb_random_search.fit(X_train_resampled, y_train_resampled)

# Print best parameters and cross-validation score
print("Best Parameters for XGBoost:", xgb_random_search.best_params_)
print("Best Cross-Validation Accuracy for XGBoost:", xgb_random_search.best_score_)


# Step 3: Decision Tree Classifier

# Initialize a Decision Tree classifier
Dt_model = DecisionTreeClassifier(random_state=42)
# Train the Decision Tree model
Dt_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred_dt = Dt_model.predict(X_val)
y_pred_proba_dt = xgb_model.predict_proba(X_val)

# Calculate metrics for Decision Tree
accuracy_dt = accuracy_score(y_val, y_pred_dt)
precision_dt = precision_score(y_val, y_pred_dt, average='weighted')
recall_dt = recall_score(y_val, y_pred_dt, average='weighted')
f1_dt = f1_score(y_val, y_pred_dt, average='weighted')
roc_auc_dt= roc_auc_score(y_val, y_pred_proba_dt, multi_class='ovr')

# Print Decision Tree metrics
print("DecisionTree Classifier:")
print(f"Accuracy: {accuracy_dt:.2f}")
print(f"Precision: {precision_dt:.2f}")
print(f"Recall: {recall_dt:.2f}")
print(f"F1 Score: {f1_dt:.2f}")
print(f"AUC: {roc_auc_dt:.2f}\n")

# Define hyperparameters for tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Set up GridSearchCV for hyperparameter tuning with cross-validation
dt_cv = GridSearchCV(Dt_model, param_grid, cv=5, scoring='accuracy')

# Fit the model to your training data
dt_cv.fit(X_train_resampled, y_train_resampled)

# Best parameters and score
print("Best Parameters for Decision Tree:", dt_cv.best_params_)
print("Best Cross-Validation Accuracy:", dt_cv.best_score_)

# Step 5: Compare Results

print("Comparison of Random Forest, XGBoost, and Decision Tree:")
#print(f"Logistic Regression - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {roc_auc:.2f}")
print(f"Random Forest      - Accuracy: {accuracy_rf:.2f}, Precision: {precision_rf:.2f}, Recall: {recall_rf:.2f}, F1 Score: {f1_rf:.2f}, AUC: {roc_auc_rf:.2f}")
print(f"XGBoost            - Accuracy: {accuracy_xgb:.2f}, Precision: {precision_xgb:.2f}, Recall: {recall_xgb:.2f}, F1 Score: {f1_xgb:.2f}, AUC: {roc_auc_xgb:.2f}")
print(f"Decision Tree      - Accuracy: {accuracy_dt:.2f}, Precision: {precision_dt:.2f}, Recall: {recall_dt:.2f}, F1 Score: {f1_dt:.2f}, AUC: {roc_auc_dt:.2f}")
