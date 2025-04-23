import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import VotingClassifier



# Set random seed for reproducibility
np.random.seed(42)

# ======================
# STEP 1: Data Loading
# ======================
print("=== STEP 1: Data Loading ===")
df = pd.read_csv('data.csv')
print(f"Initial dataset shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Class distribution plot
plt.figure(figsize=(8,6))
sns.countplot(x='diagnosis', data=df)
plt.title('Initial Class Distribution (M=1, B=0)')
plt.show()

# ======================
# STEP 2: Data Cleaning
# ======================
print("\n=== STEP 2: Data Cleaning ===")

# Drop ID column
df = df.drop(columns=['id'])
print(f"Dropped 'id' column. New shape: {df.shape}")

# Convert diagnosis to binary
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("\nTarget variable distribution:")
print(df['diagnosis'].value_counts())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicates: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicates. New shape: {df.shape}")

# ===================================
# STEP 3: Correlation-based Filtering
# ===================================
print("\n=== STEP 3: Correlation-based Feature Filtering ===")
threshold = 0.95

# Calculate initial correlations and variances
corr_matrix = df.drop('diagnosis', axis=1).corr().abs()
variances = df.drop('diagnosis', axis=1).var()

# Pre-removal correlation visualization
correlated_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > threshold:
            pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            correlated_pairs.append(pair)

correlated_pairs.sort(key=lambda x: x[2], reverse=True)

plt.figure(figsize=(10, 6))
if correlated_pairs:
    labels = [f"{pair[0]} - {pair[1]}" for pair in correlated_pairs]
    values = [pair[2] for pair in correlated_pairs]
    
    plt.barh(range(len(correlated_pairs)), values, align='center', color='darkred')
    plt.yticks(range(len(correlated_pairs)), labels)
    plt.xlabel('Correlation Coefficient')
    plt.title(f'Pre-Removal Correlated Pairs (>{threshold})')
    plt.gca().invert_yaxis()
    plt.xlim(0.9, 1.0)
    plt.grid(axis='x', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print(f"No feature pairs with correlation > {threshold} found")

# Remove features with lower variance in correlated pairs
high_corr = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > threshold:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            
            # Compare variances
            if variances[feature1] < variances[feature2]:
                remove_feature = feature1
            else:
                remove_feature = feature2
                
            high_corr.add(remove_feature)

print(f"\nRemoving {len(high_corr)} features with lower variance:")
print(high_corr)

df_filtered = df.drop(columns=high_corr)
print(f"\nRemaining features: {len(df_filtered.columns)}")

# Post-removal correlation check
post_corr = df_filtered.drop('diagnosis', axis=1).corr().abs()
remaining_pairs = []
for i in range(len(post_corr.columns)):
    for j in range(i+1, len(post_corr.columns)):
        if post_corr.iloc[i, j] > threshold:
            pair = (post_corr.columns[i], post_corr.columns[j], post_corr.iloc[i, j])
            remaining_pairs.append(pair)

plt.figure(figsize=(10, 6))
if remaining_pairs:
    labels = [f"{pair[0]} - {pair[1]}" for pair in remaining_pairs]
    values = [pair[2] for pair in remaining_pairs]
    
    plt.barh(range(len(remaining_pairs)), values, align='center', color='darkblue')
    plt.yticks(range(len(remaining_pairs)), labels)
    plt.xlabel('Correlation Coefficient')
    plt.title(f'Post-Removal Correlated Pairs (>{threshold})')
    plt.gca().invert_yaxis()
    plt.xlim(0.9, 1.0)
    plt.grid(axis='x', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print(f"No remaining correlations > {threshold} after removal")

# ============================
# STEP 4: Data Visualization
# ============================
print("\n=== STEP 4: Data Visualization ===")

plt.figure(figsize=(12,6))
df_filtered.drop('diagnosis', axis=1).boxplot()
plt.xticks(rotation=90)
plt.title('Feature Distributions After Filtering')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df_filtered.drop('diagnosis', axis=1).corr(), 
            annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Filtered Feature Correlation Matrix')
plt.show()

# ======================
# STEP 5: Data Splitting
# ======================
print("\n=== STEP 5: Data Splitting ===")

X = df_filtered.drop(columns=['diagnosis'])
y = df_filtered['diagnosis']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

print(f"\nSplits:\nTrain: {X_train.shape}\nVal: {X_val.shape}\nTest: {X_test.shape}")

# ========================
# STEP 6: Preprocessing
# ========================
print("\n=== STEP 6: Preprocessing ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.80)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nPCA reduced to {X_train_pca.shape[1]} components")

# ======================
# Initialize Results Structure
# ======================
results = {
    'models': {},
    'ensemble': None,
    'feature_importances': None
}

# ======================
# STEP 7: Enhanced Model Training
# ======================
print("\n=== STEP 7: Enhanced Model Training ===")

# Define base models (ONLY adding Perceptron here)
base_models = {
    'Logistic Regression': {
        'obj': LogisticRegression(max_iter=10000),
        'params': {
            'model__C': [0.01, 0.1, 1, 10],
            'model__class_weight': ['balanced']
        }
    },
    'SVM': {
        'obj': SVC(probability=True),
        'params': {
            'model__C': [1, 10],
            'model__gamma': ['scale', 'auto']
        }
    },
    'Gradient Boosting': {
        'obj': GradientBoostingClassifier(),
        'params': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1]
        }
    },
    # New Perceptron model added here
    'Perceptron': {
        'obj': CalibratedClassifierCV(
            Perceptron(random_state=42, 
                      penalty='l2', 
                      alpha=0.1,  # High regularization
                      max_iter=50,  # Few iterations
                      eta0=0.001),  # Small learning rate
            method='sigmoid'
        ),
        'params': {
            # No hyperparameter tuning - fixed configuration
            'model__cv': [2]  # Minimal calibration folds
        }
    }
}

# Initialize results storage
results['models'] = {name: {} for name in base_models}

# Train individual models
for name, config in base_models.items():
    print(f"\n--- Training {name} ---")
    
    try:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.85)),
            ('model', config['obj'])
        ])
        
        grid = GridSearchCV(pipeline, config['params'],
                          cv=StratifiedKFold(5),
                          scoring='accuracy',
                          n_jobs=-1)
        
        grid.fit(X_train_val, y_train_val)
        
        # Store results
        results['models'][name] = {
            'pipeline': grid.best_estimator_,
            'best_params': grid.best_params_,
            'best_score': grid.best_score_
        }
        
    except Exception as e:
        print(f"Error training {name}: {str(e)}")
        results['models'][name] = None

# ======================
# STEP 7a: Ensemble Training
# ======================
print("\n--- Creating Ensemble ---")

# Create list of successful models EXCLUDING Perceptron
successful_models = [
    (name, results['models'][name]['pipeline']) 
    for name in base_models 
    if (results['models'][name] is not None and name != 'Perceptron')  # Exclude Perceptron
]

if len(successful_models) >= 2:
    ensemble = VotingClassifier(
        estimators=successful_models,
        voting='soft',
        n_jobs=-1
    )
    
    try:
        ensemble.fit(X_train_val, y_train_val)
        results['ensemble'] = ensemble
    except Exception as e:
        print(f"Error creating ensemble: {str(e)}")
else:
    print("Not enough successful models for ensemble")

# ========================
# STEP 8: Enhanced Evaluation
# ========================
print("\n=== STEP 8: Model Evaluation ===")

# Initialize metrics storage
metrics = {}

# Evaluate individual models
for name in base_models:
    if results['models'][name] is not None:
        model = results['models'][name]['pipeline']
        test_pred = model.predict(X_test)
        
        # Get probabilities for all models
        test_proba = model.predict_proba(X_test)[:,1]
        
        metrics[name] = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'roc_auc': roc_auc_score(y_test, test_proba)
        }

# Evaluate ensemble
if results['ensemble'] is not None:
    test_pred = results['ensemble'].predict(X_test)
    test_proba = results['ensemble'].predict_proba(X_test)[:,1]
    
    metrics['Ensemble'] = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred),
        'recall': recall_score(y_test, test_pred),
        'f1': f1_score(y_test, test_pred),
        'roc_auc': roc_auc_score(y_test, test_proba)
    }

# Display results
print("\n=== Final Metrics ===")
print(pd.DataFrame(metrics).T.sort_values('accuracy', ascending=False))

# Modified Feature importance analysis section
if 'Gradient Boosting' in base_models and results['models']['Gradient Boosting'] is not None:
    try:
        # Get the PCA-transformed feature names
        pca = results['models']['Gradient Boosting']['pipeline'].named_steps['pca']
        n_components = pca.n_components_
        pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        # Get feature importances
        gb = results['models']['Gradient Boosting']['pipeline'].named_steps['model']
        importances = gb.feature_importances_
        
        # Create series with PCA component names
        results['feature_importances'] = pd.Series(
            importances,
            index=pca_feature_names
        ).sort_values(ascending=False)
        
        # Plot top 10 PCA component importances
        plt.figure(figsize=(10,6))
        results['feature_importances'].head(10).plot(kind='barh')
        plt.title('Top 10 Important PCA Components (Gradient Boosting)')
        plt.show()
        
        # Show original features contributing most to top PCA components
        print("\nTop original features contributing to important PCA components:")
        components = pca.components_
        for i, pc in enumerate(results['feature_importances'].index[:3]):
            component_idx = int(pc[2:])-1  # Extract PC number from 'PC1' etc.
            most_important_features = X_train_val.columns[
                np.argsort(-np.abs(components[component_idx]))[:5]
            ]
            print(f"{pc}: {', '.join(most_important_features)}")
            
    except Exception as e:
        print(f"Error analyzing feature importances: {str(e)}")

# ============================
# STEP 9: Feature Reduction Report
# ============================
print("\n=== STEP 9: Feature Reduction Report ===")

initial = len(pd.read_csv('data.csv').columns) - 1  # Exclude ID
final = X_train_pca.shape[1]

print(f"""
Feature Reduction Summary:
- Initial features: {initial}
- After correlation filtering: {X_train.shape[1]}
- After PCA: {final}
- Total reduction: {initial - final} features
- Retention rate: {final/initial:.1%}""")

# ============================
# STEP 10: Model Interpretation
# ============================
print("\n=== STEP 10: Model Interpretation ===")

# Logistic Regression coefficients
if 'Logistic Regression' in results['models'] and results['models']['Logistic Regression'] is not None:
    lr_pipeline = results['models']['Logistic Regression']['pipeline']
    lr_model = lr_pipeline.named_steps['model']
    coefs = pd.Series(lr_model.coef_[0], 
                    index=[f'PC{i+1}' for i in range(len(lr_model.coef_[0]))])
    plt.figure(figsize=(10,6))
    coefs.sort_values().plot(kind='barh')
    plt.title('Logistic Regression Coefficients')
    plt.show()

# Model performance comparison
metrics_df = pd.DataFrame(metrics).T.sort_values('accuracy', ascending=False)
plt.figure(figsize=(10,6))
metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()