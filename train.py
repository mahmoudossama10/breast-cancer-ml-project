import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Data Loading and Initial Inspection
df = pd.read_csv('data.csv')

# Step 2: Data Cleaning and Preprocessing
# Drop unnecessary column
df = df.drop(columns=['id'])

# Encode target variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check for duplicates
print("\nNumber of duplicates:", df.duplicated().sum())

# Split features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Step 3: Data Splitting
# Split into train+validation and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 5: Feature Engineering (PCA)
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nOriginal features:", X_train_scaled.shape[1])
print("PCA reduced features:", X_train_pca.shape[1])

# Step 6: Model Selection and Training
models = {
    'Logistic Regression': LogisticRegression(penalty='l2', max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10]
    },
    'Random Forest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 5, 10]
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__gamma': ['scale', 'auto']
    }
}

# Store results
results = {}

# Train and tune models
for model_name in models:
    print(f"\n=== Training {model_name} ===")
    
    # Create pipeline with PCA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', models[model_name])
    ])
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grids[model_name], 
                              cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_val, y_train_val)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Validation performance
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Store results
    results[model_name] = {
        'model': best_model,
        'val_accuracy': accuracy_score(y_val, val_pred),
        'val_precision': precision_score(y_val, val_pred),
        'val_recall': recall_score(y_val, val_pred),
        'val_f1': f1_score(y_val, val_pred),
        'val_roc_auc': roc_auc_score(y_val, val_proba),
        'best_params': grid_search.best_params_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Validation ROC-AUC: {results[model_name]['val_roc_auc']:.4f}")

# Step 7: Final Evaluation on Test Set
final_results = {}

for model_name in results:
    print(f"\n=== Evaluating {model_name} on Test Set ===")
    model = results[model_name]['model']
    
    # Test predictions
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    final_results[model_name] = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred),
        'recall': recall_score(y_test, test_pred),
        'f1': f1_score(y_test, test_pred),
        'roc_auc': roc_auc_score(y_test, test_proba)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Step 8: Results Comparison
results_df = pd.DataFrame(final_results).T
print("\n=== Final Test Results ===")
print(results_df)

# Step 9: Feature Importance Analysis (for tree-based models)
if 'Random Forest' in models:
    rf_model = results['Random Forest']['model'].named_steps['model']
    feature_importances = rf_model.feature_importances_
    features = X.columns
    
    # Get PCA component names
    pca_components = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': pca_components,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Important PCA Components (Random Forest)')
    plt.show()