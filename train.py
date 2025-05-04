import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge, Lasso, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef, 
                             confusion_matrix, mean_squared_error)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier, 
                             RandomForestClassifier, AdaBoostClassifier, BaggingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import math
from sklearn.base import clone
from sklearn.utils import resample
import requests
from io import StringIO

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path):
    # Load the dataset from a CSV file or URL.
    print("=== STEP 1: Data Loading ===")
    
    # Check if file_path is a URL
    if file_path.startswith('http'):
        # Fetch data from URL
        response = requests.get(file_path)
        if response.status_code == 200:
            data_content = StringIO(response.text)
            df = pd.read_csv(data_content)
            print("Successfully loaded data from URL")
        else:
            raise Exception(f"Failed to fetch data from URL: {response.status_code}")
    else:
        # Load from local file
        df = pd.read_csv(file_path)
    
    print(f"Initial dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())

   # Calculate the count of each class
    class_counts = df['diagnosis'].value_counts()

    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Initial Class Distribution (M=Malignant, B=Benign)')
    plt.show()
    return df


def clean_data(df):
    # Perform initial cleaning such as handling duplicates and converting target
    print("\n=== STEP 2: Data Cleaning ===")
    
    # Drop ID column
    df = df.drop(columns=['id'])
    print(f"Dropped 'id' column. New shape: {df.shape}")

    # Convert diagnosis to binary
    # Concept: Preparing data for binary classification
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    print("\nTarget variable distribution:")
    print(df['diagnosis'].value_counts())

    # Convert all columns to numeric (some might be loaded as strings)
    # Concept: Handling deterministic vs. stochastic noise
    for col in df.columns:
        if col != 'diagnosis':  # Skip the target variable
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception as e:
                print(f"Error converting column {col} to numeric: {e}")
                # If conversion fails, keep as is

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')  # Mean imputation for stochastic noise
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    print("Imputed missing values")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicates: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicates. New shape: {df.shape}")
    
    return df


def feature_selection(df):
    # Remove highly correlated features and perform SelectKBest feature selection.
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
    # Concept: Model Complexity Reduction
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

    # Additional Feature Selection with SelectKBest
    # Concept: Bias-Variance tradeoff - reducing features to reduce variance
    X = df_filtered.drop('diagnosis', axis=1)
    y = df_filtered['diagnosis']
    
    # Using SelectKBest with f_classif (ANOVA F-value)
    selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))  # Ensure k is not larger than number of features
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    
    df_filtered = pd.DataFrame(X_selected, columns=selected_features)
    df_filtered['diagnosis'] = y

    # Post-removal correlation check
    post_corr = df_filtered.drop('diagnosis', axis=1).corr().abs()
    remaining_pairs = []
    for i in range(len(post_corr.columns)):
        for j in range(i+1, len(post_corr.columns)):
            if post_corr.iloc[i, j] > threshold:
                pair = (post_corr.columns[i], post_corr.columns[j], post_corr.iloc[i, j])
                remaining_pairs.append(pair)

    if remaining_pairs:
        plt.figure(figsize=(10, 6))
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
    
    return df_filtered


def visualize_data(df_filtered):
    # Visualize filtered feature distributions.
    print("\n=== STEP 4: Data Visualization ===")

    plt.figure(figsize=(12, 6))
    df_filtered.drop('diagnosis', axis=1).boxplot()
    plt.xticks(rotation=90)
    plt.title('Feature Distributions After Filtering')
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_filtered.drop('diagnosis', axis=1).corr(), 
                annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Filtered Feature Correlation Matrix')
    plt.show()


def split_data(df_filtered):
    # Split data into training, validation, and test sets.
    print("\n=== STEP 5: Data Splitting ===")

    X = df_filtered.drop(columns=['diagnosis'])
    y = df_filtered['diagnosis']

    # Concept: Sampling Bias - Using stratified sampling to maintain class distribution
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

    print(f"\nSplits:\nTrain: {X_train.shape}\nVal: {X_val.shape}\nTest: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(X_train, X_val, X_test):
    # Apply Standardization and PCA.
    print("\n=== STEP 6: Preprocessing ===")

    # Concept: Feature scaling for gradient-based methods
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Concept: Dimensionality reduction to reduce model complexity
    # Use a smaller number of components if dataset is small
    n_components = max(0.80, 0.95)  # Use 80% variance or 95% if dataset is small
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"\nPCA reduced to {X_train_pca.shape[1]} components")
    
    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()
    
    return X_train_pca, X_val_pca, X_test_pca


def hoeffding_bound(epsilmon, n, delta=0.05):
    # Calculate Hoeffding's bound for generalization error.
    
    # Concept: Hoeffding's Inequality - Provides a bound on the probability that 
    # the empirical mean deviates from the true mean by more than epsilon.
    
    # Parameters:
    # epsilon: Deviation from true mean
    # n: Sample size
    # delta: Confidence parameter (default 0.05 for 95% confidence)
    
    # Returns:
    # The bound value
    return math.sqrt(math.log(2/delta) / (2 * n))


def vc_bound(n, h, delta=0.05):
    # Calculate VC generalization bound.
    
    # Concept: VC Generalization Bound - Relates the generalization error to the 
    # VC dimension (h) of the hypothesis class and the sample size.
    
    # Parameters:
    # n: Sample size
    # h: VC dimension
    # delta: Confidence parameter
    
    # Returns:
    # The bound value
    return math.sqrt((h * (math.log(2*n/h) + 1) + math.log(1/delta)) / n)


def analyze_generalization_bounds(X_train, y_train):

    # Analyze generalization bounds for different models.
    
    # Concept: Generalization Bounds - Theoretical guarantees on model performance

    print("\n=== STEP 7: Generalization Bounds Analysis ===")
    
    n = len(X_train)
    
    # Hoeffding's bound for different confidence levels
    epsilons = [0.01, 0.05, 0.1, 0.2]
    deltas = [0.01, 0.05, 0.1]
    
    print("Hoeffding's Inequality Bounds:")
    for delta in deltas:
        bounds = [hoeffding_bound(eps, n, delta) for eps in epsilons]
        print(f"  Confidence {(1-delta)*100}%: {bounds}")
    
    # VC bounds for different hypothesis classes
    # Approximate VC dimensions for different models
    vc_dims = {
        "Linear Classifier": X_train.shape[1] + 1,  # d+1 for d-dimensional data
        "Decision Tree (depth 4)": 2**4,  # 2^depth for binary tree
        "SVM with RBF kernel": np.inf  # Infinite VC dimension
    }
    
    print("\nVC Generalization Bounds:")
    for model, h in vc_dims.items():
        if h == np.inf:
            print(f"  {model}: Infinite VC dimension - bound not applicable")
        else:
            bound = vc_bound(n, h)
            print(f"  {model}: VC dim = {h}, bound = {bound:.4f}")
    
    # Practical implication
    print("\nPractical Implications:")
    print("  - Models with lower VC dimension have tighter generalization bounds")
    print("  - As sample size increases, bounds get tighter")
    print("  - Regularization helps reduce effective VC dimension")


def train_individual_models(X_train_val, y_train_val):
    # Train individual models with hyperparameter tuning.
    print("\n=== STEP 8: Enhanced Model Training ===")

    base_models = {
        'Logistic Regression': {
            # Concept: Logistic Regression with L2 regularization
            'obj': LogisticRegression(max_iter=10000, solver='liblinear'),
            'params': {
                'model__C': [0.01, 0.1, 1, 10],  # C is inverse of regularization strength
                'model__penalty': ['l1', 'l2'],  # L1 and L2 regularization
                'model__class_weight': ['balanced']
            }
        },
        'SVM': {
            # Concept: SVM with different kernels
            'obj': SVC(probability=True),
            'params': {
                'model__C': [1, 10],  # C parameter for margin-error tradeoff
                'model__kernel': ['linear', 'rbf', 'poly'],  # Different kernels
                'model__gamma': ['scale', 'auto']  # Kernel coefficient
            }
        },
        'Gradient Boosting': {
            # Concept: Boosting ensemble method
            'obj': GradientBoostingClassifier(),
            'params': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 4]
            }
        },
        'Perceptron': {
            # Concept: Perceptron with regularization
            'obj': CalibratedClassifierCV(
                Perceptron(random_state=42, 
                          penalty='l2',  # L2 regularization
                          alpha=0.1,  # Regularization strength
                          max_iter=50,
                          eta0=0.001),  # Learning rate
                method='sigmoid'
            ),
            'params': {
                'model__cv': [2]
            }
        },
        'SGD Classifier': {
            # Concept: Stochastic Gradient Descent
            'obj': SGDClassifier(loss='log_loss', random_state=42, eta0=0.01),  # Fixed eta0 > 0
            'params': {
                'model__alpha': [0.0001, 0.001, 0.01],  # Regularization strength
                'model__penalty': ['l1', 'l2', 'elasticnet'],  # Different regularization types
                'model__learning_rate': ['constant', 'optimal', 'adaptive'],  # Learning rate schedules
                'model__eta0': [0.01, 0.1, 1.0]  # Explicitly set eta0 values > 0
            }
        },
        'AdaBoost': {
            # Concept: AdaBoost algorithm
            'obj': AdaBoostClassifier(random_state=42),
            'params': {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.1, 1.0]
            }
        },
        'Bagging': {
            # Concept: Bootstrap Aggregation
            'obj': BaggingClassifier(random_state=42),
            'params': {
                'model__n_estimators': [10, 20],
                'model__max_samples': [0.5, 0.7, 1.0],
                'model__bootstrap': [True]  # Use bootstrap samples
            }
        }
    }

    results = {'models': {}}
    
    for name, config in base_models.items():
        print(f"\n--- Training {name} ---")
        
        try:
            # Concept: SMOTE for handling class imbalance
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=0.85)),
                ('model', config['obj'])
            ])
            
            # Concept: K-fold Cross Validation for hyperparameter tuning
            grid = GridSearchCV(pipeline, config['params'],
                              cv=StratifiedKFold(5),  # Stratified K-fold
                              scoring='accuracy',
                              n_jobs=-1)
            
            grid.fit(X_train_val, y_train_val)
            
            # Store results
            results['models'][name] = {
                'pipeline': grid.best_estimator_,
                'best_params': grid.best_params_,
                'best_score': grid.best_score_
            }
            
            # Concept: Learning curves to analyze bias-variance tradeoff
            if name in ['Logistic Regression', 'SVM']:
                train_sizes, train_scores, test_scores = learning_curve(
                    grid.best_estimator_, X_train_val, y_train_val, 
                    cv=5, train_sizes=np.linspace(0.1, 1.0, 5),
                    scoring='accuracy', n_jobs=-1)
                
                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                # Plot learning curve
                plt.figure(figsize=(10, 6))
                plt.title(f'Learning Curve: {name}')
                plt.xlabel('Training examples')
                plt.ylabel('Score')
                plt.grid()
                
                plt.fill_between(train_sizes, train_mean - train_std,
                                train_mean + train_std, alpha=0.1, color='r')
                plt.fill_between(train_sizes, test_mean - test_std,
                                test_mean + test_std, alpha=0.1, color='g')
                plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
                plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
                plt.legend(loc='best')
                plt.show()
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results['models'][name] = None
    
    # Analyze out-of-bag error for Bagging
    if 'Bagging' in results['models'] and results['models']['Bagging'] is not None:
        try:
            # Concept: Out-of-Bag Error estimation
            bagging = results['models']['Bagging']['pipeline'].named_steps['model']
            if hasattr(bagging, 'oob_score_'):
                print(f"\nBagging Out-of-Bag Score: {bagging.oob_score_:.4f}")
            else:
                # Create a new bagging classifier with oob_score=True
                X_train_processed = results['models']['Bagging']['pipeline'][:-1].transform(X_train_val)
                bagging_oob = BaggingClassifier(
                    n_estimators=bagging.n_estimators,
                    max_samples=bagging.max_samples,
                    bootstrap=True,
                    oob_score=True,
                    random_state=42
                )
                bagging_oob.fit(X_train_processed, y_train_val)
                print(f"\nBagging Out-of-Bag Score: {bagging_oob.oob_score_:.4f}")
        except Exception as e:
            print(f"Error calculating OOB score: {str(e)}")
            
    return results


def train_ensemble(results, X_train_val, y_train_val):
    # Create ensemble model from successful models.
    print("\n--- Creating Ensemble ---")

    # Create list of successful models EXCLUDING Perceptron
    successful_models = [
        (name, results['models'][name]['pipeline']) 
        for name in results['models'] 
        if (results['models'][name] is not None and name != 'Perceptron')  # Exclude Perceptron
    ]

    if len(successful_models) >= 2:
        # Concept: Voting Classifier ensemble
        ensemble = VotingClassifier(
            estimators=successful_models,
            voting='soft',  # Use probability estimates
            n_jobs=-1
        )
        
        try:
            ensemble.fit(X_train_val, y_train_val)
            results['ensemble'] = ensemble
            
            # Analyze individual model contributions
            print("\nEnsemble Model Weights Analysis:")
            for name, _ in successful_models:
                # Make predictions with individual model
                model = results['models'][name]['pipeline']
                y_pred = model.predict(X_train_val)
                acc = accuracy_score(y_train_val, y_pred)
                print(f"  {name} individual accuracy: {acc:.4f}")
                
        except Exception as e:
            print(f"Error creating ensemble: {str(e)}")
    else:
        print("Not enough successful models for ensemble")
    
    return results


def evaluate_models(results, X_test, y_test):
    # Evaluate models and display metrics.
    print("\n=== STEP 9: Model Evaluation ===")

    metrics = {}

    for name in results['models']:
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
                'roc_auc': roc_auc_score(y_test, test_proba),
                'mcc': matthews_corrcoef(y_test, test_pred)
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, test_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {name}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

    # Evaluate ensemble
    if 'ensemble' in results and results['ensemble'] is not None:
        test_pred = results['ensemble'].predict(X_test)
        test_proba = results['ensemble'].predict_proba(X_test)[:,1]
        
        metrics['Ensemble'] = {
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred),
            'roc_auc': roc_auc_score(y_test, test_proba),
            'mcc': matthews_corrcoef(y_test, test_pred)
        }
        
        # Ensemble confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix: Ensemble')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    # Display results
    print("\n=== Final Metrics ===")
    metrics_df = pd.DataFrame(metrics).T.sort_values('accuracy', ascending=False)
    print(metrics_df)
    return metrics_df


def analyze_feature_importance(results, X_train_val):
    # Analyze feature importance in PCA components.
    print("\n=== STEP 10: Feature Importance Analysis ===")
    
    # Modified Feature importance analysis section
    if 'Gradient Boosting' in results['models'] and results['models']['Gradient Boosting'] is not None:
        try:
            # Get the PCA-transformed feature names
            pca = results['models']['Gradient Boosting']['pipeline'].named_steps['pca']
            n_components = pca.n_components_
            pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
            
            # Get feature importances
            gb = results['models']['Gradient Boosting']['pipeline'].named_steps['model']
            importances = gb.feature_importances_
            
            # Create series with PCA component names
            feature_importances = pd.Series(importances, index=pca_feature_names).sort_values(ascending=False)
            
            # Plot top 10 PCA component importances
            plt.figure(figsize=(10, 6))
            feature_importances.head(10).plot(kind='barh')
            plt.title('Top 10 Important PCA Components (Gradient Boosting)')
            plt.show()
            
            # Show original features contributing most to top PCA components
            print("\nTop original features contributing to important PCA components:")
            components = pca.components_
            for i, pc in enumerate(feature_importances.index[:3]):
                component_idx = int(pc[2:])-1  # Extract PC number from 'PC1' etc.
                most_important_features = X_train_val.columns[
                    np.argsort(-np.abs(components[component_idx]))[:5]
                ]
                print(f"{pc}: {', '.join(most_important_features)}")
                
        except Exception as e:
            print(f"Error analyzing feature importances: {str(e)}")
    
    # L1 Regularization for feature selection
    if 'Logistic Regression' in results['models'] and results['models']['Logistic Regression'] is not None:
        try:
            lr_pipeline = results['models']['Logistic Regression']['pipeline']
            lr_model = lr_pipeline.named_steps['model']
            
            # Check if L1 regularization was used
            if hasattr(lr_model, 'penalty') and lr_model.penalty == 'l1':
                print("\nL1 Regularization Feature Selection:")
                
                # Get coefficients
                coefs = lr_model.coef_[0]
                
                # Get PCA feature names
                pca = lr_pipeline.named_steps['pca']
                n_components = pca.n_components_
                pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
                
                # Create series with PCA component names
                coef_series = pd.Series(coefs, index=pca_feature_names)
                
                # Count non-zero coefficients
                non_zero = (coef_series != 0).sum()
                print(f"  Non-zero coefficients: {non_zero} out of {len(coef_series)}")
                
                # Plot non-zero coefficients
                plt.figure(figsize=(10, 6))
                coef_series[coef_series != 0].sort_values().plot(kind='barh')
                plt.title('Non-zero Coefficients (L1 Regularization)')
                plt.show()
        except Exception as e:
            print(f"Error analyzing L1 regularization: {str(e)}")


def feature_reduction_report(df, X_train_pca):
    # Generate feature reduction report.
    print("\n=== STEP 11: Feature Reduction Report ===")

    initial = len(df.columns) - 1  # Exclude ID
    final = X_train_pca.shape[1]

    print(f"""
    Feature Reduction Summary:
    - Initial features: {initial}
    - After correlation filtering and selection by SelectKBest: {X_train_pca.shape[1]}
    - After PCA: {final}
    - Total reduction: {initial - final} features
    - Retention rate: {final/initial:.1%}""")


def interpret_logistic_regression(results):
    # Interpret Logistic Regression coefficients.
    print("\n=== STEP 12: Model Interpretation ===")

    if 'Logistic Regression' in results['models'] and results['models']['Logistic Regression'] is not None:
        lr_pipeline = results['models']['Logistic Regression']['pipeline']
        lr_model = lr_pipeline.named_steps['model']
        
        # Concept: Interpreting logistic regression coefficients
        coefs = pd.Series(lr_model.coef_[0], index=[f'PC{i+1}' for i in range(len(lr_model.coef_[0]))])
        plt.figure(figsize=(10, 6))
        coefs.sort_values().plot(kind='barh')
        plt.title('Logistic Regression Coefficients')
        plt.show()
        
        # Analyze regularization effect
        print("\nRegularization Analysis:")
        print(f"  Penalty type: {lr_model.penalty}")
        print(f"  C parameter (inverse of regularization strength): {lr_model.C}")
        
        # Calculate L1 and L2 norms
        l1_norm = np.sum(np.abs(lr_model.coef_[0]))
        l2_norm = np.sqrt(np.sum(lr_model.coef_[0]**2))
        print(f"  L1 norm of coefficients: {l1_norm:.4f}")
        print(f"  L2 norm of coefficients: {l2_norm:.4f}")


def performance_comparison(metrics_df):
    # Compare model performances using a bar plot.
    metrics_df_sorted = metrics_df[['accuracy', 'precision', 'recall', 'f1']].sort_values('accuracy', ascending=False)

    plt.figure(figsize=(10, 6))
    metrics_df_sorted.plot(kind='bar', rot=0)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def cross_validation_analysis(results, X_train_val, y_train_val):

    # Perform cross-validation analysis on the best models.
    
    # Concept: K-fold Cross Validation - Evaluating model performance

    print("\n=== STEP 13: Cross-Validation Analysis ===")
    
    # Select top 3 models based on previous results
    top_models = {}
    for name, model_info in results['models'].items():
        if model_info is not None:
            top_models[name] = model_info['pipeline']
    
    if len(top_models) > 3:
        # Keep only top 3
        top_model_names = list(top_models.keys())[:3]
        top_models = {name: top_models[name] for name in top_model_names}
    
    # Perform k-fold cross-validation
    k = 5
    cv_results = {}
    
    for name, model in top_models.items():
        print(f"\nCross-validating {name}...")
        
        try:
            # Concept: K-fold Cross Validation
            cv_scores = cross_val_score(model, X_train_val, y_train_val, 
                                       cv=StratifiedKFold(k), 
                                       scoring='accuracy',
                                       n_jobs=-1)
            
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"  Mean CV Score: {cv_scores.mean():.4f}")
            print(f"  Std Dev: {cv_scores.std():.4f}")
            print(f"  Individual Fold Scores: {cv_scores}")
            
        except Exception as e:
            print(f"Error during cross-validation for {name}: {str(e)}")
    
    # Visualize cross-validation results
    if cv_results:
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        model_names = list(cv_results.keys())
        means = [cv_results[name]['mean'] for name in model_names]
        stds = [cv_results[name]['std'] for name in model_names]
        
        # Create bar plot with error bars
        bars = plt.bar(model_names, means, yerr=stds, alpha=0.8, capsize=10)
        
        # Add individual fold scores as scatter points
        for i, name in enumerate(model_names):
            scores = cv_results[name]['scores']
            plt.scatter([i] * len(scores), scores, color='black', alpha=0.6)
        
        plt.title(f'{k}-Fold Cross-Validation Results')
        plt.ylabel('Accuracy')
        plt.ylim(0.5, 1.0)  # Adjust as needed
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


def bias_variance_analysis(results, X_train, y_train, X_test, y_test):

    # Analyze bias-variance tradeoff for different models.
    
    # Concept: Bias-Variance Tradeoff - Understanding model complexity

    print("\n=== STEP 14: Bias-Variance Analysis ===")
    
    # Select a few representative models
    models_to_analyze = ['Logistic Regression', 'SVM', 'Gradient Boosting']
    available_models = [m for m in models_to_analyze if m in results['models'] and results['models'][m] is not None]
    
    if not available_models:
        print("No suitable models available for bias-variance analysis")
        return
    
    # Function to estimate bias and variance using bootstrap
    def estimate_bias_variance(model, X_train, y_train, X_test, y_test, n_bootstraps=100):
        y_preds = np.zeros((n_bootstraps, len(y_test)))
        
        # Create bootstrap samples and get predictions
        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = resample(range(len(X_train)), replace=True, n_samples=len(X_train))
            X_boot, y_boot = X_train[indices], y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
            
            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_boot, y_boot)
            
            # Predict
            y_preds[i] = model_clone.predict(X_test)
        
        # Calculate average prediction
        y_pred_mean = np.mean(y_preds, axis=0)
        
        # Calculate squared error of average prediction (bias²)
        bias_squared = np.mean((y_pred_mean - y_test) ** 2)
        
        # Calculate variance of predictions
        variance = np.mean(np.var(y_preds, axis=0))
        
        # Calculate average error
        error = np.mean([(y_preds[i] - y_test) ** 2 for i in range(n_bootstraps)])
        
        # Calculate noise (irreducible error)
        noise = error - bias_squared - variance
        
        return {
            'bias_squared': bias_squared,
            'variance': variance,
            'total_error': error,
            'noise': max(0, noise)  # Ensure non-negative
        }
    
    # Analyze each model
    bias_variance_results = {}
    
    for name in available_models:
        print(f"\nAnalyzing bias-variance for {name}...")
        
        try:
            model = results['models'][name]['pipeline']
            
            # Convert to numpy arrays if needed
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
            
            # Estimate bias and variance
            bv_result = estimate_bias_variance(model, X_train_np, y_train_np, X_test_np, y_test_np, n_bootstraps=50)
            bias_variance_results[name] = bv_result
            
            print(f"  Bias²: {bv_result['bias_squared']:.4f}")
            print(f"  Variance: {bv_result['variance']:.4f}")
            print(f"  Total Error: {bv_result['total_error']:.4f}")
            
        except Exception as e:
            print(f"Error during bias-variance analysis for {name}: {str(e)}")
    
    # Visualize bias-variance decomposition
    if bias_variance_results:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for stacked bar chart
        model_names = list(bias_variance_results.keys())
        bias_squared = [bias_variance_results[name]['bias_squared'] for name in model_names]
        variance = [bias_variance_results[name]['variance'] for name in model_names]
        noise = [bias_variance_results[name]['noise'] for name in model_names]
        
        # Create stacked bar chart
        width = 0.6
        plt.bar(model_names, bias_squared, width, label='Bias²', color='#3498db')
        plt.bar(model_names, variance, width, bottom=bias_squared, label='Variance', color='#e74c3c')
        
        if any(n > 0.001 for n in noise):  # Only show noise if significant
            bottom = [b + v for b, v in zip(bias_squared, variance)]
            plt.bar(model_names, noise, width, bottom=bottom, label='Irreducible Error', color='#95a5a6')
        
        plt.title('Bias-Variance Decomposition')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


def main():
    # Load data from the provided URL
    df = load_data('https://hebbkx1anhila5yf.public.blob.vercel-storage.com/data-rZr5Sn6P3FxXvOU6ZNLh49pmihOL4O.csv')
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature selection
    df_filtered = feature_selection(df_clean)
    
    # Visualize data
    visualize_data(df_filtered)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_filtered)
    
    # Preprocess data
    X_train_pca, X_val_pca, X_test_pca = preprocess_data(X_train, X_val, X_test)
    
    # Analyze generalization bounds
    analyze_generalization_bounds(X_train, y_train)
    
    # Train individual models
    X_train_val_combined = np.vstack((X_train_pca, X_val_pca))
    y_train_val_combined = np.concatenate((y_train, y_val))
    results = train_individual_models(X_train_val_combined, y_train_val_combined)
    
    # Train ensemble
    results = train_ensemble(results, X_train_val_combined, y_train_val_combined)
    
    # Evaluate models
    metrics_df = evaluate_models(results, X_test_pca, y_test)
    
    # Analyze feature importance
    analyze_feature_importance(results, pd.DataFrame(X_train, columns=X_train.columns))
    
    # Feature reduction report
    feature_reduction_report(df, X_train_pca)
    
    # Interpret logistic regression
    interpret_logistic_regression(results)
    
    # Performance comparison
    performance_comparison(metrics_df)
    
    # Cross-validation analysis
    cross_validation_analysis(results, X_train_val_combined, y_train_val_combined)
    
    # Bias-variance analysis
    bias_variance_analysis(results, X_train_pca, y_train, X_test_pca, y_test)
    
    print("\n=== Machine Learning Pipeline Complete ===")


if __name__ == '__main__':
    main()