import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import pickle
import json
import warnings
import mlflow
import mlflow.sklearn
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up directory for models
if not os.path.exists('models'):
    os.makedirs('models')

# Configure MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = "insurance_enrollment_prediction"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
mlflow.set_experiment(experiment_name)

# Set random seed for reproducibility
RANDOM_STATE = 42

# Function to load and prepare the data
def load_data(file_path="data/processed_employee_data.csv"):
    """
    Load the dataset and prepare it for ML modeling
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Display basic info about the dataset
    print("\nTarget distribution:")
    print(df['enrolled'].value_counts(normalize=True) * 100)
    
    return df

# Function to split the data
def split_data(df, target_col='enrolled', test_size=0.2, random_state=RANDOM_STATE):
    """
    Split the dataset into training and testing sets
    """
    print(f"\nSplitting data into train and test sets (test size: {test_size})...")
    X = df.drop(columns=[target_col, 'employee_id'] if 'employee_id' in df.columns else [target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

# Function to identify feature types
def identify_feature_types(X):
    """
    Identify numerical, categorical, and engineered features
    """
    # Original numerical features
    numerical_features = ['age', 'salary', 'tenure_years']
    
    # Original categorical features
    categorical_features = ['gender', 'marital_status', 'employment_type', 'region', 'has_dependents']
    
    # Engineered features
    engineered_features = [col for col in X.columns if col not in numerical_features + categorical_features]
    
    # Filter to keep only columns present in the dataframe
    numerical_features = [col for col in numerical_features if col in X.columns]
    categorical_features = [col for col in categorical_features if col in X.columns]
    
    # Print feature information
    print(f"\nNumerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Engineered features: {len(engineered_features)}")
    
    return numerical_features, categorical_features, engineered_features

# Function to create preprocessing pipeline
def create_preprocessing_pipeline(numerical_features, categorical_features, engineered_categorical=None):
    """
    Create a preprocessing pipeline for numerical and categorical features
    """
    # If no engineered categorical features specified, default to empty list
    if engineered_categorical is None:
        engineered_categorical = []
    
    # All categorical features (original + engineered categorical)
    all_categorical = categorical_features + engineered_categorical
    
    # Preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, all_categorical)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

# Function to build and evaluate a model
def build_model(preprocessor, X_train, X_test, y_train, y_test, model_type='rf', params=None):
    """
    Build and evaluate a machine learning model
    """
    print(f"\nBuilding {model_type} model...")
    
    # Set default parameters if not provided
    if params is None:
        params = {}
    
    # Initialize the model
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, **params)
    elif model_type == 'svm':
        model = SVC(random_state=RANDOM_STATE, probability=True, **params)
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(random_state=RANDOM_STATE, **params)
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print evaluation metrics
    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # Create evaluation results dictionary
    eval_results = {
        'model_type': model_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'params': params
    }
    
    return pipeline, eval_results

# Function to perform cross-validation
def cross_validate_model(pipeline, X, y, cv=5):
    """
    Perform cross-validation on the model
    """
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    # Calculate cross-validation scores
    cv_accuracy = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(pipeline, X, y, cv=skf, scoring='precision')
    cv_recall = cross_val_score(pipeline, X, y, cv=skf, scoring='recall')
    cv_f1 = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')
    cv_auc = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
    
    # Print cross-validation results
    print(f"CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"CV Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"CV Recall: {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"CV AUC-ROC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    
    # Create cross-validation results dictionary
    cv_results = {
        'accuracy': {
            'mean': float(cv_accuracy.mean()),
            'std': float(cv_accuracy.std())
        },
        'precision': {
            'mean': float(cv_precision.mean()),
            'std': float(cv_precision.std())
        },
        'recall': {
            'mean': float(cv_recall.mean()),
            'std': float(cv_recall.std())
        },
        'f1_score': {
            'mean': float(cv_f1.mean()),
            'std': float(cv_f1.std())
        },
        'auc_roc': {
            'mean': float(cv_auc.mean()),
            'std': float(cv_auc.std())
        }
    }
    
    return cv_results

# Function to perform hyperparameter tuning
def tune_hyperparameters(preprocessor, X_train, y_train, model_type='rf', param_grid=None):
    """
    Perform hyperparameter tuning using Grid Search
    """
    print(f"\nPerforming hyperparameter tuning for {model_type}...")
    
    # Set default parameter grid if not provided
    if param_grid is None:
        if model_type == 'rf':
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gb':
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5, 10]
            }
        elif model_type == 'xgb':
            param_grid = {
                'model__n_estimators': [100, 200, 300],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            print(f"No default param grid for {model_type}, using basic model.")
            return build_model(preprocessor, X_train, X_train, y_train, y_train, model_type=model_type)
    
    # Initialize the model
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_type == 'gb':
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    elif model_type == 'svm':
        model = SVC(random_state=RANDOM_STATE, probability=True)
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(random_state=RANDOM_STATE)
    elif model_type == 'lgb':
        model = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Print best results
    print(f"Best AUC-ROC: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    return best_pipeline, best_params, best_score

# Function to plot feature importance
def plot_feature_importance(model, preprocessor, X_train, output_path="models"):
    """
    Plot feature importance for tree-based models
    """
    # Extract model from pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_step = model.named_steps['model']
    else:
        model_step = model
    
    # Check if the model has feature_importances_ attribute
    if not hasattr(model_step, 'feature_importances_'):
        print("This model doesn't have feature importances available.")
        return None
    
    # Get feature names from preprocessor
    if hasattr(preprocessor, 'transformers_'):
        # Get feature names from each transformer
        feature_names = []
        
        # Get numerical feature names
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat' and hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                # Get categorical feature names
                cat_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
                feature_names.extend(cat_features)
            elif name == 'remainder' and transformer == 'passthrough':
                # Handle passthrough features
                if hasattr(preprocessor, 'feature_names_in_'):
                    remainder_cols = [col for col in preprocessor.feature_names_in_ if col not in feature_names]
                    feature_names.extend(remainder_cols)
    else:
        feature_names = [f"Feature_{i}" for i in range(len(model_step.feature_importances_))]
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names[:len(model_step.feature_importances_)],
        'Importance': model_step.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), palette='viridis')
    plt.title(f'Top 20 Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'feature_importance.png'))
    plt.close()
    
    return feature_importance

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_proba, output_path="models"):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'roc_curve.png'))
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, output_path="models"):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks([0.5, 1.5], ['Not Enrolled (0)', 'Enrolled (1)'])
    plt.yticks([0.5, 1.5], ['Not Enrolled (0)', 'Enrolled (1)'])
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    plt.close()

# Function to save model
def save_model(model, model_name="insurance_enrollment_model", output_path="models"):
    """
    Save the trained model and preprocessing information
    """
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, f"{model_name}.pkl")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model_path

# Function to save model details
def save_model_details(model_details, model_name="insurance_enrollment_model", output_path="models"):
    """
    Save model details (evaluation metrics, parameters, etc.)
    """
    os.makedirs(output_path, exist_ok=True)
    details_path = os.path.join(output_path, f"{model_name}_details.json")
    
    # Save details as JSON
    with open(details_path, 'w') as f:
        json.dump(model_details, f, indent=4)
    
    print(f"Model details saved to {details_path}")
    
    return details_path

# Function to run the entire ML pipeline
def run_ml_pipeline(data_path="data/processed_employee_data.csv", 
                    output_path="models", 
                    model_type="rf", 
                    tune=True,
                    test_size=0.2,
                    random_state=RANDOM_STATE):
    """
    Run the full machine learning pipeline
    """
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Load data
        df = load_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)
        
        # Identify feature types
        numerical_features, categorical_features, engineered_features = identify_feature_types(X_train)
        
        # Determine which engineered features are categorical
        engineered_categorical = [col for col in engineered_features if X_train[col].dtype == 'object']
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(
            numerical_features, categorical_features, engineered_categorical
        )
        
        # Tune hyperparameters or build model directly
        if tune:
            print("\nPerforming hyperparameter tuning...")
            best_model, best_params, best_score = tune_hyperparameters(
                preprocessor, X_train, y_train, model_type=model_type
            )
            
            # Make predictions with the best model
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Evaluate with test set
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            print("\nTest Set Evaluation:")
            print(f"Accuracy: {test_accuracy:.4f}")
            print(f"Precision: {test_precision:.4f}")
            print(f"Recall: {test_recall:.4f}")
            print(f"F1 Score: {test_f1:.4f}")
            print(f"AUC-ROC: {test_auc:.4f}")
            
            # Cross-validate the best model
            cv_results = cross_validate_model(best_model, X_train, y_train)
            
            # Create model details dictionary
            model_details = {
                'model_type': model_type,
                'best_params': best_params,
                'cv_results': cv_results,
                'test_results': {
                    'accuracy': test_accuracy,
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1_score': test_f1,
                    'auc_roc': test_auc
                },
                'data_info': {
                    'total_samples': len(df),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': X_train.shape[1],
                    'numerical_features': numerical_features,
                    'categorical_features': categorical_features,
                    'engineered_features': engineered_features
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Plot feature importance if applicable
            feature_importance = plot_feature_importance(best_model, preprocessor, X_train, output_path)
            
            # Plot ROC curve
            plot_roc_curve(y_test, y_pred_proba, output_path)
            
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, output_path)
            
            # Save model and details
            model_path = save_model(best_model, model_name=f"{model_type}_model", output_path=output_path)
            details_path = save_model_details(model_details, model_name=f"{model_type}_model", output_path=output_path)
            
            # Log parameters and metrics to MLflow
            for param_name, param_value in best_params.items():
                param_short_name = param_name.replace('model__', '')
                mlflow.log_param(param_short_name, param_value)
            
            mlflow.log_metric("accuracy", test_accuracy)
            mlflow.log_metric("precision", test_precision)
            mlflow.log_metric("recall", test_recall)
            mlflow.log_metric("f1_score", test_f1)
            mlflow.log_metric("auc_roc", test_auc)
            mlflow.log_metric("cv_accuracy_mean", cv_results['accuracy']['mean'])
            
            # Log artifacts
            mlflow.log_artifact(os.path.join(output_path, 'feature_importance.png'))
            mlflow.log_artifact(os.path.join(output_path, 'roc_curve.png'))
            mlflow.log_artifact(os.path.join(output_path, 'confusion_matrix.png'))
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            result = {
                'model': best_model,
                'details': model_details,
                'model_path': model_path,
                'details_path': details_path
            }
            
        else:
            # Build model directly
            model, eval_results = build_model(
                preprocessor, X_train, X_test, y_train, y_test, model_type=model_type
            )
            
            # Cross-validate model
            cv_results = cross_validate_model(model, X_train, y_train)
            
            # Create model details dictionary
            model_details = {
                'model_type': model_type,
                'params': eval_results['params'],
                'cv_results': cv_results,
                'test_results': {
                    'accuracy': eval_results['accuracy'],
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'f1_score': eval_results['f1_score'],
                    'auc_roc': eval_results['auc_roc']
                },
                'data_info': {
                    'total_samples': len(df),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': X_train.shape[1],
                    'numerical_features': numerical_features,
                    'categorical_features': categorical_features,
                    'engineered_features': engineered_features
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Get predictions for plots
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Plot feature importance if applicable
            feature_importance = plot_feature_importance(model, preprocessor, X_train, output_path)
            
            # Plot ROC curve
            plot_roc_curve(y_test, y_pred_proba, output_path)
            
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, output_path)
            
            # Save model and details
            model_path = save_model(model, model_name=f"{model_type}_model", output_path=output_path)
            details_path = save_model_details(model_details, model_name=f"{model_type}_model", output_path=output_path)
            
            # Log parameters and metrics to MLflow
            if 'params' in eval_results and eval_results['params']:
                for param_name, param_value in eval_results['params'].items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.log_metric("accuracy", eval_results['accuracy'])
            mlflow.log_metric("precision", eval_results['precision'])
            mlflow.log_metric("recall", eval_results['recall'])
            mlflow.log_metric("f1_score", eval_results['f1_score'])
            mlflow.log_metric("auc_roc", eval_results['auc_roc'])
            mlflow.log_metric("cv_accuracy_mean", cv_results['accuracy']['mean'])
            
            # Log artifacts
            if os.path.exists(os.path.join(output_path, 'feature_importance.png')):
                mlflow.log_artifact(os.path.join(output_path, 'feature_importance.png'))
            mlflow.log_artifact(os.path.join(output_path, 'roc_curve.png'))
            mlflow.log_artifact(os.path.join(output_path, 'confusion_matrix.png'))
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            result = {
                'model': model,
                'details': model_details,
                'model_path': model_path,
                'details_path': details_path
            }
    
    print("\nML pipeline completed successfully!")
    return result

# Function to make predictions on new data
def predict_enrollment(model, data):
    """
    Make predictions on new data
    """
    # Check if data is a dataframe or a dictionary
    if isinstance(data, dict):
        # Convert single record dictionary to dataframe
        data = pd.DataFrame([data])
    
    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    
    # Format results
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        # Determine confidence level
        if abs(prob - 0.5) > 0.4:
            confidence = "Very High"
        elif abs(prob - 0.5) > 0.3:
            confidence = "High"
        elif abs(prob - 0.5) > 0.2:
            confidence = "Medium"
        elif abs(prob - 0.5) > 0.1:
            confidence = "Low"
        else:
            confidence = "Very Low"
        
        # Determine risk level
        if pred == 1:
            if prob > 0.9:
                risk_level = "Very Low Risk"
            elif prob > 0.8:
                risk_level = "Low Risk"
            elif prob > 0.7:
                risk_level = "Moderate Risk"
            elif prob > 0.6:
                risk_level = "Elevated Risk"
            else:
                risk_level = "High Risk"
        else:
            if prob < 0.1:
                risk_level = "Very Low Interest"
            elif prob < 0.2:
                risk_level = "Low Interest"
            elif prob < 0.3:
                risk_level = "Moderate Interest"
            elif prob < 0.4:
                risk_level = "Potential Interest"
            else:
                risk_level = "Strong Potential"
        
        # Create result dictionary
        result = {
            'employee_index': i,
            'prediction': int(pred),
            'probability': float(prob),
            'enrolled': "Yes" if pred == 1 else "No",
            'confidence': confidence,
            'risk_level': risk_level
        }
        
        results.append(result)
    
    return results

# Function to compare multiple models
def compare_models(X_train, X_test, y_train, y_test, preprocessor, models_to_compare=['rf', 'gb', 'xgb', 'lr']):
    """
    Compare multiple ML models
    """
    print("\nComparing multiple models...")
    
    results = {}
    best_auc = 0
    best_model_name = None
    
    for model_type in models_to_compare:
        print(f"\n{'-'*50}")
        pipeline, eval_results = build_model(
            preprocessor, X_train, X_test, y_train, y_test, model_type=model_type
        )
        
        results[model_type] = {
            'pipeline': pipeline,
            'eval_results': eval_results
        }
        
        # Track best model
        if eval_results['auc_roc'] > best_auc:
            best_auc = eval_results['auc_roc']
            best_model_name = model_type
    
    print(f"\nModel comparison completed. Best model: {best_model_name} (AUC-ROC: {best_auc:.4f})")
    
    return results, best_model_name

# Main function
def main():
    """
    Main function to run the ML pipeline
    """
    print("=" * 80)
    print("INSURANCE ENROLLMENT PREDICTION - ML PIPELINE")
    print("=" * 80)
    
    # Set parameters
    data_path = "data/processed_employee_data.csv"
    output_path = "models"
    model_type = "xgb"  # Choose from: 'rf', 'gb', 'xgb', 'lgb', 'lr', 'svm'
    tune = True  # Whether to perform hyperparameter tuning
    
    # Run the pipeline
    result = run_ml_pipeline(
        data_path=data_path,
        output_path=output_path,
        model_type=model_type,
        tune=tune
    )
    
    # Print completion message
    print("\nML pipeline execution completed!")
    print(f"Model saved to: {result['model_path']}")
    print(f"Details saved to: {result['details_path']}")

# Run the main function
if __name__ == "__main__":
    main()