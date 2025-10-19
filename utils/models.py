
import pandas as pd
import numpy as np
# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score
import joblib
import time
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Universal model training and evaluation class for classification and regression
    """
    
    def __init__(self, problem_type='classification', random_state=42):
        """
        Initialize the model trainer
        
        Args:
            problem_type (str): 'classification' or 'regression'
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.problem_type = problem_type
        self.is_regression = problem_type == 'regression'
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.is_trained = False
        
        # Initialize models based on problem type
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models based on problem type"""
        if self.is_regression:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(
                    random_state=self.random_state,
                    alpha=1.0
                ),
                'Lasso Regression': Lasso(
                    random_state=self.random_state,
                    alpha=1.0,
                    max_iter=1000
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'SVR': SVR(
                    kernel='rbf',
                    C=1.0
                ),
                'K-Nearest Neighbors': KNeighborsRegressor(
                    n_neighbors=5,
                    n_jobs=-1
                ),
                'Decision Tree': DecisionTreeRegressor(
                    random_state=self.random_state,
                    max_depth=10
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    random_state=self.random_state,
                    n_estimators=100
                )
            }
        else:
            # Classification models
            self.models = {
                'Logistic Regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    solver='lbfgs'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'SVM': SVC(
                    random_state=self.random_state,
                    probability=True,
                    kernel='rbf'
                ),
                'K-Nearest Neighbors': KNeighborsClassifier(
                    n_neighbors=5,
                    n_jobs=-1
                ),
                'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    random_state=self.random_state,
                    n_estimators=100
                )
            }
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        Train all models on the training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            dict: Training results and metrics
        """
        print("🚀 Starting model training...")
        training_results = {}
        
        for model_name, model in self.models.items():
            print(f"📊 Training {model_name}...")
            start_time = time.time()
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
                
                # Calculate training time
                training_time = time.time() - start_time
                
                # Evaluate on all available sets
                train_pred = model.predict(X_train)
                results = {
                    'model': model,
                    'training_time': training_time
                }
                
                # Add training metrics based on problem type
                if self.is_regression:
                    results['train_r2'] = r2_score(y_train, train_pred)
                    results['train_mse'] = mean_squared_error(y_train, train_pred)
                    results['train_mae'] = mean_absolute_error(y_train, train_pred)
                    results['train_rmse'] = np.sqrt(mean_squared_error(y_train, train_pred))
                else:
                    results['train_accuracy'] = accuracy_score(y_train, train_pred)
                
                # Add validation and test metrics based on problem type
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    if self.is_regression:
                        results['val_r2'] = r2_score(y_val, val_pred)
                        results['val_mse'] = mean_squared_error(y_val, val_pred)
                        results['val_mae'] = mean_absolute_error(y_val, val_pred)
                        results['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                        try:
                            results['val_mape'] = mean_absolute_percentage_error(y_val, val_pred)
                        except:
                            results['val_mape'] = 'N/A'
                    else:
                        results['val_accuracy'] = accuracy_score(y_val, val_pred)
                        results['val_precision'] = precision_score(y_val, val_pred, average='weighted', zero_division=0)
                        results['val_recall'] = recall_score(y_val, val_pred, average='weighted', zero_division=0)
                        results['val_f1'] = f1_score(y_val, val_pred, average='weighted', zero_division=0)
                
                if X_test is not None and y_test is not None:
                    test_pred = model.predict(X_test)
                    if self.is_regression:
                        results['test_r2'] = r2_score(y_test, test_pred)
                        results['test_mse'] = mean_squared_error(y_test, test_pred)
                        results['test_mae'] = mean_absolute_error(y_test, test_pred)
                        results['test_rmse'] = np.sqrt(mean_squared_error(y_test, test_pred))
                        try:
                            results['test_mape'] = mean_absolute_percentage_error(y_test, test_pred)
                        except:
                            results['test_mape'] = 'N/A'
                    else:
                        results['test_accuracy'] = accuracy_score(y_test, test_pred)
                        results['test_precision'] = precision_score(y_test, test_pred, average='weighted', zero_division=0)
                        results['test_recall'] = recall_score(y_test, test_pred, average='weighted', zero_division=0)
                        results['test_f1'] = f1_score(y_test, test_pred, average='weighted', zero_division=0)
                
                # Cross-validation score
                cv_scoring = 'r2' if self.is_regression else 'accuracy'
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=cv_scoring)
                results['cv_mean'] = cv_scores.mean()
                results['cv_std'] = cv_scores.std()
                
                training_results[model_name] = results
                print(f"✅ {model_name} completed in {training_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                training_results[model_name] = {'error': str(e)}
        
        self.results = training_results
        self.is_trained = True
        print("🎉 All models trained successfully!")
        
        return training_results
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Detailed evaluation of a specific model
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Detailed evaluation metrics
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        
        # Calculate all metrics
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        # Add ROC AUC for binary/multiclass classification
        try:
            if len(np.unique(y_test)) == 2:  # Binary classification
                evaluation['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1] if y_pred_proba is not None else y_pred)
            else:  # Multiclass classification
                evaluation['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            evaluation['roc_auc'] = None
            evaluation['roc_auc_error'] = str(e)
        
        return evaluation
    
    def get_model_comparison(self):
        """
        Get comparison of all trained models
        
        Returns:
            pd.DataFrame: Comparison of model performance
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models first.")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                # Include failed models with error indication
                if self.is_regression:
                    row = {
                        'Model': model_name,
                        'Train R²': 'Error',
                        'Val R²': 'Error',
                        'Test R²': 'Error',
                        'Val RMSE': 'Error',
                        'Test RMSE': 'Error',
                        'Val MAE': 'Error',
                        'Test MAE': 'Error',
                        'CV Mean': 'Error',
                        'CV Std': 'Error',
                        'Training Time (s)': 'Error',
                        'Error': results.get('error', 'Unknown error')
                    }
                else:
                    row = {
                        'Model': model_name,
                        'Train Accuracy': 'Error',
                        'Val Accuracy': 'Error',
                        'Test Accuracy': 'Error',
                        'Val F1 Score': 'Error',
                        'Test F1 Score': 'Error',
                        'CV Mean': 'Error',
                        'CV Std': 'Error',
                        'Training Time (s)': 'Error',
                        'Error': results.get('error', 'Unknown error')
                    }
                comparison_data.append(row)
                continue
                
            # Format numeric values properly
            def format_metric(value, decimal_places=4):
                if value is None or value == 'N/A':
                    return 'N/A'
                try:
                    return round(float(value), decimal_places)
                except (ValueError, TypeError):
                    return 'N/A'
            
            if self.is_regression:
                row = {
                    'Model': model_name,
                    'Train R²': format_metric(results.get('train_r2')),
                    'Val R²': format_metric(results.get('val_r2')),
                    'Test R²': format_metric(results.get('test_r2')),
                    'Val RMSE': format_metric(results.get('val_rmse')),
                    'Test RMSE': format_metric(results.get('test_rmse')),
                    'Val MAE': format_metric(results.get('val_mae')),
                    'Test MAE': format_metric(results.get('test_mae')),
                    'CV Mean': format_metric(results.get('cv_mean')),
                    'CV Std': format_metric(results.get('cv_std')),
                    'Training Time (s)': format_metric(results.get('training_time'), 3)
                }
            else:
                row = {
                    'Model': model_name,
                    'Train Accuracy': format_metric(results.get('train_accuracy')),
                    'Val Accuracy': format_metric(results.get('val_accuracy')),
                    'Test Accuracy': format_metric(results.get('test_accuracy')),
                    'Val F1 Score': format_metric(results.get('val_f1')),
                    'Test F1 Score': format_metric(results.get('test_f1')),
                    'CV Mean': format_metric(results.get('cv_mean')),
                    'CV Std': format_metric(results.get('cv_std')),
                    'Training Time (s)': format_metric(results.get('training_time'), 3)
                }
            comparison_data.append(row)
        
        if not comparison_data:
            # Return empty dataframe with expected columns based on problem type
            if self.is_regression:
                columns = ['Model', 'Train R²', 'Val R²', 'Test R²', 
                          'Val RMSE', 'Test RMSE', 'Val MAE', 'Test MAE', 
                          'CV Mean', 'CV Std', 'Training Time (s)']
            else:
                columns = ['Model', 'Train Accuracy', 'Val Accuracy', 'Test Accuracy', 
                          'Val F1 Score', 'Test F1 Score', 'CV Mean', 'CV Std', 'Training Time (s)']
            return pd.DataFrame(columns=columns)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric=None):
        """
        Get the best performing model based on specified metric
        
        Args:
            metric (str): Metric to use for comparison. If None, uses default based on problem type.
            
        Returns:
            tuple: (best_model_name, best_model, best_score)
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models first.")
        
        # Set default metric based on problem type
        if metric is None:
            metric = 'val_r2' if self.is_regression else 'val_accuracy'
        
        best_score = -float('inf')  # Use negative infinity for both high-is-better and low-is-better metrics
        best_model_name = None
        
        # For metrics where lower is better (RMSE, MAE)
        minimize_metrics = ['val_rmse', 'test_rmse', 'val_mae', 'test_mae']
        is_minimize = metric.lower() in minimize_metrics
        
        if is_minimize:
            best_score = float('inf')  # Start with positive infinity for minimize metrics
        
        for model_name, results in self.results.items():
            if 'error' in results or metric not in results:
                continue
                
            score = results[metric]
            if isinstance(score, (int, float)):
                if is_minimize:
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name
                else:
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"No valid models found with metric {metric}")
        
        return best_model_name, self.trained_models[best_model_name], best_score
    
    def get_feature_importance(self, model_name, feature_names=None, top_n=20):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature importance
        if not hasattr(model, 'feature_importances_'):
            return pd.DataFrame({'message': [f'{model_name} does not support feature importance']})
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_models(self, filepath_prefix='model'):
        """
        Save all trained models
        
        Args:
            filepath_prefix (str): Prefix for model files
        """
        if not self.is_trained:
            raise ValueError("No models to save. Train models first.")
        
        saved_files = []
        for model_name, model in self.trained_models.items():
            filename = f"{filepath_prefix}_{model_name.replace(' ', '_').lower()}.joblib"
            joblib.dump(model, filename)
            saved_files.append(filename)
        
        # Save results
        results_filename = f"{filepath_prefix}_results.joblib"
        joblib.dump(self.results, results_filename)
        saved_files.append(results_filename)
        
        return saved_files
    
    def load_models(self, filepath_prefix='model'):
        """
        Load previously saved models
        
        Args:
            filepath_prefix (str): Prefix for model files
        """
        import os
        
        # Load results
        results_filename = f"{filepath_prefix}_results.joblib"
        if os.path.exists(results_filename):
            self.results = joblib.load(results_filename)
        
        # Load models
        for model_name in self.models.keys():
            filename = f"{filepath_prefix}_{model_name.replace(' ', '_').lower()}.joblib"
            if os.path.exists(filename):
                self.trained_models[model_name] = joblib.load(filename)
        
        if self.trained_models:
            self.is_trained = True
    
    def predict_sample(self, model_name, sample_features, feature_names=None):
        """
        Make prediction for a single sample
        
        Args:
            model_name (str): Name of the model to use
            sample_features: Feature values for the sample
            feature_names (list): Names of features for display
            
        Returns:
            dict: Prediction results
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.trained_models[model_name]
        
        # Ensure sample is in correct format
        if len(sample_features.shape) == 1:
            sample_features = sample_features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(sample_features)[0]
        
        # Get prediction probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(sample_features)[0]
        
        return {
            'prediction': prediction,
            'prediction_probability': prediction_proba,
            'model_used': model_name,
            'feature_names': feature_names
        }


def quick_train_models(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, random_state=42):
    """
    Quick function to train all models
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)  
        X_test: Test features (optional)
        y_test: Test labels (optional)
        random_state (int): Random state
        
    Returns:
        tuple: (trainer, results, comparison_df)
    """
    trainer = ModelTrainer(random_state=random_state)
    results = trainer.train_models(X_train, y_train, X_val, y_val, X_test, y_test)
    comparison_df = trainer.get_model_comparison()
    
    return trainer, results, comparison_df