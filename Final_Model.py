import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

# Preprocessing function
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
                           'EntityType', 'Usage', 'EvidenceRole', 'OrgId', 'DeviceId', 'Sha256', 'IpAddress', 'Url',
                           'AccountSid', 'AccountUpn', 'AccountObjectId', 'AccountName', 'DeviceName', 'NetworkMessageId',
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
    corr_target = abs(corr_matrix['IncidentGrade'])
    relevant_features = corr_target[corr_target > 0.15].index
    df = df[relevant_features]
    print(f"Selected Features based on correlation threshold: {relevant_features}")

    return df


# Usage on Train and Test Datasets
train_data = pd.read_csv("Train_Data.csv")
test_data = pd.read_csv("Test_Data.csv")

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


#  Random Forest Classifier

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
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

# Use RandomizedSearchCV for faster tuning
rf_random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist_rf,
    n_iter=2, cv=3, scoring='accuracy',
    error_score='raise',
    random_state=42, n_jobs=-1
)

# Fit RandomizedSearchCV
try:
    rf_random_search.fit(X_train_resampled, y_train_resampled)
    print("Best Parameters for Random Forest:", rf_random_search.best_params_)
    print("Best Cross-Validation Accuracy for Random Forest:", rf_random_search.best_score_)
except Exception as e:
    print("An error occurred during hyperparameter tuning:", e)

# Evaluating the best selected model on test set
best_rf_model = rf_random_search.best_estimator_

# Prepare Test Data (features and target)
X_test = test_data.drop(columns=['IncidentGrade'])  # Features
y_test = test_data['IncidentGrade']  # Target

# Align test data with training data columns to avoid feature mismatch error
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Make Predictions on Test Data
y_test_pred = best_rf_model.predict(X_test)
y_test_pred_proba = best_rf_model.predict_proba(X_test)

# Calculate Metrics for Test Data
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')

# Print Test Metrics
print("Random Forest Classifier - Test Set Performance:")
print(f"Test Accuracy: {accuracy_test:.2f}")
print(f"Test Precision: {precision_test:.2f}")
print(f"Test Recall: {recall_test:.2f}")
print(f"Test F1 Score: {f1_test:.2f}")
print(f"Test AUC: {roc_auc_test:.2f}\n")

# Comparison of Validation and Test Performance
print("Comparison of Validation and Test Performance:")
print(f"Validation Accuracy: {accuracy_rf:.2f} vs Test Accuracy: {accuracy_test:.2f}")
print(f"Validation Precision: {precision_rf:.2f} vs Test Precision: {precision_test:.2f}")
print(f"Validation Recall: {recall_rf:.2f} vs Test Recall: {recall_test:.2f}")
print(f"Validation F1 Score: {f1_rf:.2f} vs Test F1 Score: {f1_test:.2f}")
print(f"Validation AUC: {roc_auc_rf:.2f} vs Test AUC: {roc_auc_test:.2f}")

importance = best_rf_model.feature_importances_  # Extract feature importances
feature_names = X_train.columns  # Get feature names

# Create a DataFrame to display features with their importance values
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top 10 features
print("Top 10 Most Important Features:")
print(feature_importance_df.head(5))

# Plotting Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance_df.head(5), x='Importance', y='Feature')
plt.title('Top 10 Feature Importance in Random Forest Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()