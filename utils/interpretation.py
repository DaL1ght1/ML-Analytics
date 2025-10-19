import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import interpretation libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP library not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    st.warning("LIME library not available. Install with: pip install lime")


class ModelInterpreter:
    """
    Comprehensive model interpretation class using SHAP and LIME
    """
    
    def __init__(self, model, X_train, feature_names=None, class_names=None):
        """
        Initialize the interpreter
        
        Args:
            model: Trained ML model
            X_train: Training data for baseline
            feature_names (list): Names of features
            class_names (list): Names of target classes
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize explainers
        self._initialize_shap()
        self._initialize_lime()
    
    def _initialize_shap(self):
        """Initialize SHAP explainer"""
        if not SHAP_AVAILABLE:
            return
        
        try:
            # Choose explainer based on model type
            model_name = type(self.model).__name__.lower()
            
            if any(name in model_name for name in ['randomforest', 'tree', 'forest', 'xgb', 'lgb', 'catboost', 'gradientboosting']):
                # Tree-based models
                try:
                    self.shap_explainer = shap.TreeExplainer(self.model)
                except Exception as tree_error:
                    # Fallback to Kernel explainer for tree models that don't support TreeExplainer
                    st.warning(f"TreeExplainer failed for {model_name}, using KernelExplainer: {tree_error}")
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        shap.sample(self.X_train, min(50, len(self.X_train)))  # Smaller sample for performance
                    )
            elif any(name in model_name for name in ['linear', 'logistic']):
                # Linear models
                try:
                    self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
                except Exception as linear_error:
                    st.warning(f"LinearExplainer failed for {model_name}, using KernelExplainer: {linear_error}")
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        shap.sample(self.X_train, min(50, len(self.X_train)))
                    )
            else:
                # General explainer for other models (SVM, KNN, Naive Bayes, etc.)
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    shap.sample(self.X_train, min(50, len(self.X_train)))  # Use smaller sample for better performance
                )
            
            # Test the explainer with a small sample to ensure it works
            if self.shap_explainer is not None:
                test_sample = self.X_train[:1]
                try:
                    test_values = self.shap_explainer.shap_values(test_sample)
                    # If we get here, the explainer is working
                except Exception as test_error:
                    st.warning(f"SHAP explainer test failed for {model_name}: {test_error}")
                    self.shap_explainer = None
                    
        except Exception as e:
            st.warning(f"Could not initialize SHAP explainer for {type(self.model).__name__}: {str(e)}")
            self.shap_explainer = None
    
    def _initialize_lime(self):
        """Initialize LIME explainer"""
        if not LIME_AVAILABLE:
            return
        
        try:
            # Determine if classification or regression
            is_classification = hasattr(self.model, 'predict_proba')
            
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification' if is_classification else 'regression',
                discretize_continuous=True
            )
        except Exception as e:
            st.warning(f"Could not initialize LIME explainer: {str(e)}")
            self.lime_explainer = None
    
    def get_shap_values(self, X_explain, max_samples=100):
        """
        Get SHAP values for explanation
        
        Args:
            X_explain: Data to explain
            max_samples (int): Maximum number of samples to explain
            
        Returns:
            np.array: SHAP values
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None
        
        try:
            # Limit samples for performance
            if len(X_explain) > max_samples:
                indices = np.random.choice(len(X_explain), max_samples, replace=False)
                X_sample = X_explain[indices]
            else:
                X_sample = X_explain
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            return shap_values, X_sample
        except Exception as e:
            st.error(f"Error computing SHAP values: {str(e)}")
            return None
    
    def plot_shap_summary(self, X_explain, max_samples=100, plot_type='bar', class_idx=None):
        """
        Create SHAP summary plot
        
        Args:
            X_explain: Data to explain
            max_samples (int): Maximum samples for performance
            plot_type (str): Type of plot ('dot', 'bar', 'violin')
            class_idx (int): Class index for multiclass (None for auto-aggregate)
            
        Returns:
            matplotlib.Figure: SHAP summary plot
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            st.error("SHAP not available")
            return None
        
        result = self.get_shap_values(X_explain, max_samples)
        if result is None:
            return None
        
        shap_values, X_sample = result
        
        try:
            # Handle multiclass SHAP values
            if isinstance(shap_values, list) and len(shap_values) > 1:
                if class_idx is not None and class_idx < len(shap_values):
                    # Use specific class
                    plot_values = shap_values[class_idx]
                    class_name = self.class_names[class_idx] if self.class_names and class_idx < len(self.class_names) else f"Class {class_idx}"
                    title_suffix = f" - {class_name}"
                else:
                    # Aggregate across classes (use mean absolute values)
                    plot_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    title_suffix = " - All Classes (Mean Absolute)"
            else:
                # Single class or binary classification
                plot_values = shap_values[0] if isinstance(shap_values, list) else shap_values
                title_suffix = ""
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            try:
                if plot_type == 'bar':
                    # For bar plots, aggregate to feature importance
                    if len(plot_values.shape) == 2:
                        feature_importance = np.mean(np.abs(plot_values), axis=0)
                    else:
                        feature_importance = np.abs(plot_values)
                    
                    # Get feature names
                    feature_names = self.feature_names[:len(feature_importance)] if self.feature_names else [f"Feature_{i}" for i in range(len(feature_importance))]
                    
                    # Sort by importance
                    sorted_idx = np.argsort(feature_importance)[::-1][:20]  # Top 20
                    sorted_importance = feature_importance[sorted_idx]
                    sorted_names = [feature_names[i] for i in sorted_idx]
                    
                    # Create bar plot
                    bars = ax.barh(range(len(sorted_importance)), sorted_importance, color='steelblue', alpha=0.7)
                    ax.set_yticks(range(len(sorted_importance)))
                    ax.set_yticklabels(sorted_names)
                    ax.set_xlabel('Mean |SHAP Value|')
                    ax.set_title(f'SHAP Feature Importance{title_suffix}')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
                        ax.text(val + 0.001, i, f'{val:.3f}', va='center', ha='left', fontsize=8)
                
                elif plot_type in ['dot', 'violin']:
                    # Try original SHAP plots for dot and violin
                    shap.summary_plot(plot_values if not isinstance(shap_values, list) else shap_values, 
                                    X_sample, 
                                    feature_names=self.feature_names, 
                                    show=False, plot_type=plot_type,
                                    max_display=20)
                    plt.title(f'SHAP Summary Plot ({plot_type}){title_suffix}')
                
            except Exception as plot_error:
                # Fallback to simple bar plot
                st.warning(f"Standard SHAP plot failed, using fallback: {plot_error}")
                if len(plot_values.shape) == 2:
                    feature_importance = np.mean(np.abs(plot_values), axis=0)
                else:
                    feature_importance = np.abs(plot_values) if plot_values.ndim == 1 else np.mean(np.abs(plot_values), axis=0)
                
                feature_names = self.feature_names[:len(feature_importance)] if self.feature_names else [f"Feature_{i}" for i in range(len(feature_importance))]
                sorted_idx = np.argsort(feature_importance)[::-1][:15]
                
                ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='steelblue', alpha=0.7)
                ax.set_yticks(range(len(sorted_idx)))
                ax.set_yticklabels([feature_names[i] for i in sorted_idx])
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'SHAP Feature Importance{title_suffix}')
                ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating SHAP summary plot: {str(e)}")
            return self._create_fallback_importance_plot(X_explain, max_samples)
    
    def plot_shap_waterfall(self, X_sample, sample_idx=0, class_idx=None):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X_sample: Sample data
            sample_idx (int): Index of sample to explain
            class_idx (int): Class index for multiclass (None for auto-select)
            
        Returns:
            matplotlib.Figure: SHAP waterfall plot
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            st.error("SHAP not available")
            return None
        
        try:
            if len(X_sample.shape) == 1:
                X_sample = X_sample.reshape(1, -1)
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle multiclass case
            if isinstance(shap_values, list):
                # Multiple classes - choose class with highest prediction probability
                if class_idx is None:
                    # Get prediction probabilities to choose the predicted class
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba(X_sample)
                        class_idx = np.argmax(probs[sample_idx])
                    else:
                        class_idx = 0  # Default to first class
                
                class_idx = min(class_idx, len(shap_values) - 1)  # Ensure valid index
                values = shap_values[class_idx][sample_idx]
                base_value = self.shap_explainer.expected_value[class_idx]
                class_name = self.class_names[class_idx] if self.class_names and class_idx < len(self.class_names) else f"Class {class_idx}"
            else:
                # Single output (binary classification or regression)
                if len(shap_values.shape) == 2:
                    values = shap_values[sample_idx]
                else:
                    values = shap_values
                base_value = self.shap_explainer.expected_value
                class_name = "Prediction"
            
            # Create the explanation object
            explanation = shap.Explanation(
                values=values,
                base_values=base_value,
                data=X_sample[sample_idx],
                feature_names=self.feature_names[:len(values)] if self.feature_names else [f"Feature_{i}" for i in range(len(values))]
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create waterfall plot
            shap.waterfall_plot(explanation, show=False)
            
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx+1} ({class_name})')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"Error creating SHAP waterfall plot: {str(e)}")
            # Try alternative approach for problematic cases
            try:
                return self._create_alternative_waterfall(X_sample, sample_idx)
            except:
                return None
    
    def get_shap_feature_importance(self, X_explain, max_samples=100):
        """
        Get global feature importance from SHAP values
        
        Args:
            X_explain: Data to explain
            max_samples (int): Maximum samples
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return pd.DataFrame()
            
        try:
            # Limit samples for performance and stability
            if len(X_explain) > max_samples:
                indices = np.random.choice(len(X_explain), max_samples, replace=False)
                X_sample = X_explain[indices]
            else:
                X_sample = X_explain
            
            # Try to get SHAP values with progressive fallback
            try:
                shap_values = self.shap_explainer.shap_values(X_sample)
            except Exception as shap_error:
                # If SHAP computation fails, try with smaller sample
                try:
                    smaller_sample = X_sample[:min(10, len(X_sample))]
                    shap_values = self.shap_explainer.shap_values(smaller_sample)
                    X_sample = smaller_sample
                except Exception as smaller_error:
                    # If still failing, return empty DataFrame silently
                    return pd.DataFrame()
            
            if shap_values is None:
                return pd.DataFrame()
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                if len(shap_values) == 0:
                    return pd.DataFrame()
                
                # Multiclass case - aggregate across classes
                try:
                    # Check if all arrays in list have the same shape
                    shapes = [sv.shape for sv in shap_values if sv is not None]
                    if not shapes or len(set(shapes)) > 1:
                        # Inconsistent shapes, use first valid one
                        valid_shap = next((sv for sv in shap_values if sv is not None and sv.size > 0), None)
                        if valid_shap is None:
                            return pd.DataFrame()
                        shap_importance = np.abs(valid_shap).mean(0)
                    else:
                        # All shapes consistent, take mean across classes
                        shap_importance = np.mean([np.abs(sv).mean(0) for sv in shap_values if sv is not None], axis=0)
                except Exception as list_error:
                    # Fallback: use first valid array
                    valid_shap = next((sv for sv in shap_values if sv is not None and sv.size > 0), None)
                    if valid_shap is None:
                        return pd.DataFrame()
                    shap_importance = np.abs(valid_shap).mean(0)
            else:
                # Single output case
                if shap_values.size == 0:
                    return pd.DataFrame()
                
                if len(shap_values.shape) == 2:
                    shap_importance = np.abs(shap_values).mean(0)
                elif len(shap_values.shape) == 1:
                    shap_importance = np.abs(shap_values)
                else:
                    return pd.DataFrame()
            
            # Ensure we have valid importance scores
            if shap_importance is None or len(shap_importance) == 0:
                return pd.DataFrame()
            
            # Create feature names
            if self.feature_names and len(self.feature_names) >= len(shap_importance):
                feature_names = self.feature_names[:len(shap_importance)]
            else:
                feature_names = [f'Feature_{i}' for i in range(len(shap_importance))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': shap_importance
            })
            
            # Filter out any invalid values and sort
            importance_df = importance_df[importance_df['importance'].notna()]
            importance_df = importance_df[importance_df['importance'] >= 0]
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            # Return empty DataFrame instead of showing error to user
            return pd.DataFrame()
    
    def explain_instance_lime(self, instance, num_features=10):
        """
        Explain single instance using LIME
        
        Args:
            instance: Single instance to explain
            num_features (int): Number of features to show
            
        Returns:
            dict: LIME explanation
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            st.error("LIME not available")
            return None
        
        try:
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)
            
            explanation = self.lime_explainer.explain_instance(
                instance[0],
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=num_features
            )
            
            return explanation
        except Exception as e:
            st.error(f"Error creating LIME explanation: {str(e)}")
            return None
    
    def plot_lime_explanation(self, explanation, title="LIME Explanation"):
        """
        Create LIME explanation plot
        
        Args:
            explanation: LIME explanation object
            title (str): Plot title
            
        Returns:
            plotly.Figure: LIME explanation plot
        """
        if explanation is None:
            return None
        
        try:
            # Get explanation data
            exp_data = explanation.as_list()
            features = [item[0] for item in exp_data]
            values = [item[1] for item in exp_data]
            
            # Create color based on positive/negative impact
            colors = ['red' if v < 0 else 'green' for v in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=features,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f'{v:.3f}' for v in values],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title='Feature Impact',
                yaxis_title='Features',
                height=max(400, len(features) * 30),
                showlegend=False
            )
            
            return fig
        except Exception as e:
            st.error(f"Error plotting LIME explanation: {str(e)}")
            return None
    
    def create_interpretation_dashboard(self, X_explain, sample_idx=0, max_samples=50):
        """
        Create comprehensive interpretation dashboard
        
        Args:
            X_explain: Data to explain
            sample_idx (int): Sample index for individual explanation
            max_samples (int): Maximum samples for global explanations
            
        Returns:
            dict: Dashboard components
        """
        dashboard = {}
        
        # SHAP Global Feature Importance
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            dashboard['shap_importance'] = self.get_shap_feature_importance(X_explain, max_samples)
            dashboard['shap_summary_plot'] = self.plot_shap_summary(X_explain, max_samples, 'bar')
            
            # Individual explanation
            if sample_idx < len(X_explain):
                sample = X_explain[sample_idx:sample_idx+1]
                dashboard['shap_waterfall'] = self.plot_shap_waterfall(sample, 0)
        
        # LIME Individual Explanation
        if LIME_AVAILABLE and self.lime_explainer is not None and sample_idx < len(X_explain):
            instance = X_explain[sample_idx:sample_idx+1]
            lime_explanation = self.explain_instance_lime(instance)
            if lime_explanation:
                dashboard['lime_plot'] = self.plot_lime_explanation(lime_explanation)
                dashboard['lime_explanation'] = lime_explanation
        
        return dashboard
    
    def _create_alternative_waterfall(self, X_sample, sample_idx=0):
        """
        Create alternative SHAP explanation when waterfall fails
        
        Args:
            X_sample: Sample data
            sample_idx (int): Index of sample
            
        Returns:
            matplotlib.Figure: Alternative SHAP plot
        """
        try:
            if len(X_sample.shape) == 1:
                X_sample = X_sample.reshape(1, -1)
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multiclass - use first class or predicted class
                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(X_sample)
                    class_idx = np.argmax(probs[sample_idx])
                else:
                    class_idx = 0
                
                values = shap_values[class_idx][sample_idx]
            else:
                if len(shap_values.shape) == 2:
                    values = shap_values[sample_idx]
                else:
                    values = shap_values
            
            # Create bar plot as alternative
            fig, ax = plt.subplots(figsize=(10, 8))
            
            feature_names = self.feature_names[:len(values)] if self.feature_names else [f"Feature_{i}" for i in range(len(values))]
            
            # Sort by absolute value
            sorted_idx = np.argsort(np.abs(values))[::-1][:15]  # Top 15 features
            sorted_values = values[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]
            
            # Create horizontal bar plot
            colors = ['red' if v < 0 else 'green' for v in sorted_values]
            bars = ax.barh(range(len(sorted_values)), sorted_values, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(sorted_values)))
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title(f'SHAP Feature Impact - Sample {sample_idx+1}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, sorted_values)):
                ax.text(val + (0.001 if val >= 0 else -0.001), i, f'{val:.3f}', 
                       va='center', ha='left' if val >= 0 else 'right', fontsize=8)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.warning(f"Alternative SHAP plot also failed: {e}")
            return None
    
    def _create_fallback_importance_plot(self, X_explain, max_samples=50):
        """
        Create fallback feature importance plot when SHAP fails
        
        Args:
            X_explain: Data to explain
            max_samples (int): Maximum samples
            
        Returns:
            matplotlib.Figure: Fallback importance plot
        """
        try:
            # Use model's built-in feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.feature_names[:len(importance)] if self.feature_names else [f"Feature_{i}" for i in range(len(importance))]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1][:15]
                sorted_importance = importance[sorted_idx]
                sorted_names = [feature_names[i] for i in sorted_idx]
                
                bars = ax.barh(range(len(sorted_importance)), sorted_importance, color='orange', alpha=0.7)
                ax.set_yticks(range(len(sorted_importance)))
                ax.set_yticklabels(sorted_names)
                ax.set_xlabel('Feature Importance')
                ax.set_title('Model Feature Importance (Fallback)')
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                return fig
            else:
                # Create simple message plot
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                return fig
                
        except Exception as e:
            st.warning(f"Fallback plot also failed: {e}")
            return None
    
    def compare_explanations(self, X_sample, sample_idx=0):
        """
        Compare SHAP and LIME explanations for the same sample
        
        Args:
            X_sample: Sample data
            sample_idx (int): Index of sample
            
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        # Get SHAP explanation
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            shap_result = self.get_shap_values(X_sample)
            if shap_result:
                shap_values, _ = shap_result
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # First class
                
                shap_importance = np.abs(shap_values[sample_idx])
                shap_features = list(zip(self.feature_names, shap_importance))
                shap_features.sort(key=lambda x: x[1], reverse=True)
                comparison['shap'] = shap_features[:10]
        
        # Get LIME explanation
        if LIME_AVAILABLE and self.lime_explainer is not None:
            lime_exp = self.explain_instance_lime(X_sample[sample_idx:sample_idx+1])
            if lime_exp:
                comparison['lime'] = lime_exp.as_list()
        
        return comparison
    
    def get_prediction_confidence(self, X_sample):
        """
        Get prediction confidence/probability
        
        Args:
            X_sample: Sample data
            
        Returns:
            dict: Prediction confidence information
        """
        try:
            prediction = self.model.predict(X_sample)
            
            confidence_info = {
                'predictions': prediction,
                'probabilities': None,
                'confidence_scores': None
            }
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_sample)
                confidence_info['probabilities'] = probabilities
                confidence_info['confidence_scores'] = np.max(probabilities, axis=1)
            elif hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(X_sample)
                confidence_info['decision_scores'] = decision_scores
                
                # Convert to confidence (higher absolute value = higher confidence)
                if len(decision_scores.shape) == 1:
                    confidence_info['confidence_scores'] = np.abs(decision_scores)
                else:
                    confidence_info['confidence_scores'] = np.max(np.abs(decision_scores), axis=1)
            
            return confidence_info
        except Exception as e:
            st.error(f"Error getting prediction confidence: {str(e)}")
            return {}


def create_interpretation_summary(interpreter, X_explain, sample_indices=[0], max_samples=50):
    """
    Create comprehensive interpretation summary
    
    Args:
        interpreter (ModelInterpreter): Initialized interpreter
        X_explain: Data to explain
        sample_indices (list): Indices of samples to explain individually
        max_samples (int): Maximum samples for global analysis
        
    Returns:
        dict: Interpretation summary
    """
    summary = {
        'global_importance': None,
        'individual_explanations': {},
        'model_insights': {}
    }
    
    # Global importance
    if SHAP_AVAILABLE and interpreter.shap_explainer is not None:
        summary['global_importance'] = interpreter.get_shap_feature_importance(X_explain, max_samples)
    
    # Individual explanations
    for idx in sample_indices[:5]:  # Limit to 5 samples
        if idx < len(X_explain):
            dashboard = interpreter.create_interpretation_dashboard(X_explain, idx, max_samples)
            summary['individual_explanations'][idx] = dashboard
    
    # Model insights
    summary['model_insights'] = {
        'shap_available': SHAP_AVAILABLE and interpreter.shap_explainer is not None,
        'lime_available': LIME_AVAILABLE and interpreter.lime_explainer is not None,
        'feature_count': len(interpreter.feature_names),
        'sample_count': len(X_explain)
    }
    
    return summary