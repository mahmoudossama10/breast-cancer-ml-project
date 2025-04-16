import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Switch to an interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Data Loading and Initial Inspection
df = pd.read_csv('data.csv')

# Step 2: Data Cleaning and Preprocessing
# Drop unnecessary column
df = df.drop(columns=['id'])

# Encode target variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Check for missing values and duplicates
print("Missing values:\n", df.isnull().sum())
print("\nNumber of duplicates:", df.duplicated().sum())

# Split features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Step 3: Data Splitting
# Split into train+validation and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Split train_val into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# Step 4: Feature Scaling (for initial exploration)
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
# Replace 'Random Forest' with 'Perceptron'
models = {
    'Logistic Regression': LogisticRegression(penalty='l2', max_iter=10000),
    'Perceptron': Perceptron(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True)
}

# Hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10]
    },
    'Perceptron': {
         'model__alpha': [0.0001, 0.001, 0.01],
         'model__penalty': ['l2', 'l1', 'elasticnet', None]
    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__gamma': ['scale', 'auto']
    }
}

# Store training and validation results here
results = {}

# Train and tune models
for model_name in models:
    print(f"\n=== Training {model_name} ===")
    
    # Create pipeline with scaling and PCA included
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', models[model_name])
    ])
    
    # Grid search with cross-validation (using ROC-AUC as scoring)
    grid_search = GridSearchCV(pipeline, param_grids[model_name], 
                               cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_val, y_train_val)
    
    # Best model from grid search
    best_model = grid_search.best_estimator_
    
    # Get validation predictions
    val_pred = best_model.predict(X_val)
    # Use predict_proba if available; otherwise, use decision_function for ROC-AUC
    if hasattr(best_model, "predict_proba"):
        val_proba = best_model.predict_proba(X_val)[:, 1]
    else:
        val_proba = best_model.decision_function(X_val)
    
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
    
    # Use predict_proba if available; otherwise, use decision_function for ROC-AUC
    if hasattr(model, "predict_proba"):
        test_score = model.predict_proba(X_test)[:, 1]
    else:
        test_score = model.decision_function(X_test)
    
    # Calculate test metrics
    final_results[model_name] = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred),
        'recall': recall_score(y_test, test_pred),
        'f1': f1_score(y_test, test_pred),
        'roc_auc': roc_auc_score(y_test, test_score)
    }
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Step 8: Results Comparison
results_df = pd.DataFrame(final_results).T
print("\n=== Final Test Results ===")
print(results_df)

# Step 9: Feature Importance Analysis for models that support it
# Note: Perceptron does not have a built-in feature importance method.
# For tree-based models, you might use feature_importances_ but here we show for example purposes.
if 'Perceptron' in models:
    # Extract the underlying Perceptron model from the pipeline
    # (In this simple linear model, feature coefficients could act as a proxy for feature importance)
    model = results['Perceptron']['model'].named_steps['model']
    # Note: The coefficients correspond to the PCA components, not the original features.
    try:
        importances = model.coef_[0]
        pca_components = [f'PC{i+1}' for i in range(len(importances))]
        importance_df = pd.DataFrame({
            'PCA Component': pca_components,
            'Coefficient': importances
        }).sort_values('Coefficient', key=lambda col: abs(col), ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='PCA Component', data=importance_df.head(10))
        plt.title('Top 10 PCA Component Coefficients (Perceptron)')
        plt.show()
    except Exception as e:
        print("Feature importance could not be computed:", e)
