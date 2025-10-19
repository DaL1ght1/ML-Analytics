"""
Visualization Module for Student Dropout Prediction

This module contains functions for creating interactive plots and visualizations
including confusion matrices, ROC curves, feature importance, and model comparison charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization class for ML model analysis
    """
    
    def __init__(self, color_scheme='plotly'):
        """
        Initialize visualizer with color scheme
        
        Args:
            color_scheme (str): Color scheme to use ('plotly', 'seaborn', 'custom')
        """
        self.color_scheme = color_scheme
        self.colors = self._get_color_palette()
    
    def _get_color_palette(self):
        """Get color palette based on scheme"""
        if self.color_scheme == 'plotly':
            return px.colors.qualitative.Plotly
        elif self.color_scheme == 'seaborn':
            return sns.color_palette("husl", 10).as_hex()
        else:
            return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix", 
                            normalize=False, use_plotly=True):
        """
        Create confusion matrix plot
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title (str): Plot title
            normalize (bool): Whether to normalize values
            use_plotly (bool): Use Plotly for interactive plot
            
        Returns:
            plotly.Figure or matplotlib.Figure: Confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        if labels is None:
            labels = [f'Class {i}' for i in range(len(cm))]
        
        if use_plotly:
            # Create Plotly heatmap
            fig = px.imshow(
                cm,
                x=labels,
                y=labels,
                color_continuous_scale='Blues',
                title=title,
                labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'}
            )
            
            # Add text annotations
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    text = f'{cm[i][j]:.2%}' if normalize else f'{cm[i][j]}'
                    fig.add_annotation(
                        x=j, y=i, text=text,
                        showarrow=False,
                        font=dict(color='white' if cm[i][j] > cm.max()/2 else 'black')
                    )
            
            fig.update_layout(width=500, height=500)
            return fig
        
        else:
            # Create matplotlib plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            return fig
    
    def plot_roc_curves(self, models_data, title="ROC Curves Comparison", use_plotly=True):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_data (dict): Dictionary with model names as keys and 
                              (y_true, y_pred_proba) as values
            title (str): Plot title
            use_plotly (bool): Use Plotly for interactive plot
            
        Returns:
            plotly.Figure or matplotlib.Figure: ROC curves plot
        """
        if use_plotly:
            fig = go.Figure()
            
            for i, (model_name, (y_true, y_pred_proba)) in enumerate(models_data.items()):
                try:
                    # Handle binary and multiclass cases
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'{model_name} (AUC = {roc_auc:.3f})',
                            line=dict(color=self.colors[i % len(self.colors)], width=2)
                        ))
                    else:
                        # Multiclass - plot average ROC
                        from sklearn.metrics import roc_auc_score
                        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                        
                        # For multiclass, we'll show the macro-average ROC curve
                        classes = np.unique(y_true)
                        y_bin = label_binarize(y_true, classes=classes)
                        
                        fpr_grid = np.linspace(0, 1, 100)
                        mean_tpr = np.zeros_like(fpr_grid)
                        
                        for class_idx in range(len(classes)):
                            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_pred_proba[:, class_idx])
                            mean_tpr += np.interp(fpr_grid, fpr, tpr)
                        
                        mean_tpr /= len(classes)
                        
                        fig.add_trace(go.Scatter(
                            x=fpr_grid, y=mean_tpr,
                            mode='lines',
                            name=f'{model_name} (AUC = {roc_auc:.3f})',
                            line=dict(color=self.colors[i % len(self.colors)], width=2)
                        ))
                        
                except Exception as e:
                    st.warning(f"Could not plot ROC curve for {model_name}: {str(e)}")
            
            # Add diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=600, height=500,
                showlegend=True
            )
            
            return fig
        
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i, (model_name, (y_true, y_pred_proba)) in enumerate(models_data.items()):
                try:
                    if len(np.unique(y_true)) == 2:
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', 
                               color=self.colors[i % len(self.colors)], linewidth=2)
                except Exception:
                    continue
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
    
    def plot_feature_importance(self, importance_df, title="Feature Importance", 
                              top_n=20, use_plotly=True):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
            title (str): Plot title
            top_n (int): Number of top features to show
            use_plotly (bool): Use Plotly for interactive plot
            
        Returns:
            plotly.Figure or matplotlib.Figure: Feature importance plot
        """
        # Get top N features
        plot_df = importance_df.head(top_n).copy()
        
        if use_plotly:
            fig = px.bar(
                plot_df,
                x='importance',
                y='feature',
                orientation='h',
                title=title,
                labels={'importance': 'Importance Score', 'feature': 'Features'},
                color='importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=max(400, top_n * 20),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
            
            bars = ax.barh(plot_df['feature'], plot_df['importance'], 
                          color=self.colors[0])
            
            ax.set_xlabel('Importance Score')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(plot_df.iterrows()):
                ax.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}',
                       va='center', fontsize=8)
            
            plt.tight_layout()
            return fig
    
    def plot_model_comparison(self, comparison_df, metric='Val Accuracy', 
                            title="Model Performance Comparison", use_plotly=True):
        """
        Plot model comparison chart
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            metric (str): Metric to compare
            title (str): Plot title
            use_plotly (bool): Use Plotly for interactive plot
            
        Returns:
            plotly.Figure or matplotlib.Figure: Model comparison plot
        """
        if comparison_df is None or comparison_df.empty:
            return None
            
        # Clean data - remove N/A values and handle different data types
        plot_df = comparison_df.copy()
        
        # Check if metric column exists
        if metric not in plot_df.columns:
            available_metrics = [col for col in plot_df.columns if 'Accuracy' in col or 'F1' in col or 'Score' in col]
            if available_metrics:
                metric = available_metrics[0]  # Use first available metric
                title = f"Model Performance Comparison ({metric})"
            else:
                st.warning(f"Metric '{metric}' not found in comparison data")
                return None
        
        # Filter out N/A values (both string 'N/A' and actual NaN)
        plot_df = plot_df[plot_df[metric] != 'N/A'].copy()
        
        if len(plot_df) == 0:
            st.warning(f"No valid data available for metric '{metric}'")
            return None
        
        # Convert metric to numeric, handling any remaining non-numeric values
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors='coerce')
        plot_df = plot_df.dropna(subset=[metric])
        
        if len(plot_df) == 0:
            st.warning(f"No numeric values found for metric '{metric}'")
            return None
        
        # Sort by metric value
        plot_df = plot_df.sort_values(metric, ascending=True)
        
        if use_plotly:
            try:
                fig = px.bar(
                    plot_df,
                    x=metric,
                    y='Model',
                    orientation='h',
                    title=title,
                    labels={metric: f'{metric} Score', 'Model': 'Models'},
                    color=metric,
                    color_continuous_scale='viridis',
                    text=metric
                )
                
                # Format text on bars
                fig.update_traces(
                    texttemplate='%{text:.3f}',
                    textposition='outside'
                )
                
                fig.update_layout(
                    height=max(400, len(plot_df) * 50),
                    yaxis={'categoryorder': 'total ascending'},
                    showlegend=False
                )
                
                return fig
            except Exception as e:
                st.warning(f"Plotly visualization failed: {e}. Using fallback.")
                return self._create_fallback_comparison_plot(plot_df, metric, title)
        
        else:
            fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))
            
            bars = ax.barh(plot_df['Model'], plot_df[metric], 
                          color=self.colors[:len(plot_df)])
            
            ax.set_xlabel(f'{metric} Score')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(plot_df.iterrows()):
                ax.text(row[metric] + 0.001, i, f'{row[metric]:.3f}',
                       va='center', fontsize=10)
            
            plt.tight_layout()
            return fig
    
    def plot_training_history(self, training_times, accuracies, model_names, 
                            title="Training Time vs Accuracy", use_plotly=True):
        """
        Plot training time vs accuracy scatter plot
        
        Args:
            training_times (list): Training times for each model
            accuracies (list): Accuracies for each model
            model_names (list): Model names
            title (str): Plot title
            use_plotly (bool): Use Plotly for interactive plot
            
        Returns:
            plotly.Figure or matplotlib.Figure: Training analysis plot
        """
        if use_plotly:
            fig = px.scatter(
                x=training_times,
                y=accuracies,
                text=model_names,
                title=title,
                labels={'x': 'Training Time (seconds)', 'y': 'Accuracy Score'},
                size=[50] * len(model_names),  # Uniform size
                color=accuracies,
                color_continuous_scale='viridis'
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(showlegend=False)
            
            return fig
        
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(training_times, accuracies, 
                               c=accuracies, cmap='viridis', 
                               s=100, alpha=0.7)
            
            # Add model name labels
            for i, name in enumerate(model_names):
                ax.annotate(name, (training_times[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
            
            ax.set_xlabel('Training Time (seconds)')
            ax.set_ylabel('Accuracy Score')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, label='Accuracy')
            
            return fig
    
    def plot_data_distribution(self, df, column, title=None, use_plotly=True):
        """
        Plot data distribution for a column
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to plot
            title (str): Plot title
            use_plotly (bool): Use Plotly for interactive plot
            
        Returns:
            plotly.Figure or matplotlib.Figure: Distribution plot
        """
        if title is None:
            title = f"Distribution of {column}"
        
        if df[column].dtype in ['object', 'category']:
            # Categorical data - bar chart
            value_counts = df[column].value_counts()
            
            if use_plotly:
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=title,
                    labels={'x': column, 'y': 'Count'}
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot(kind='bar', ax=ax, color=self.colors[0])
                ax.set_title(title)
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig
        
        else:
            # Numeric data - histogram
            if use_plotly:
                fig = px.histogram(
                    df, x=column,
                    title=title,
                    nbins=30,
                    labels={'x': column, 'y': 'Frequency'}
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[column], bins=30, color=self.colors[0], alpha=0.7)
                ax.set_title(title)
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                return fig
    
    def create_dashboard_metrics(self, metrics_dict):
        """
        Create metrics display for Streamlit dashboard
        
        Args:
            metrics_dict (dict): Dictionary of metrics to display
            
        Returns:
            None: Displays metrics in Streamlit
        """
        cols = st.columns(len(metrics_dict))
        
        for i, (metric_name, value) in enumerate(metrics_dict.items()):
            with cols[i]:
                if isinstance(value, float):
                    st.metric(metric_name, f"{value:.3f}")
                else:
                    st.metric(metric_name, value)


def quick_confusion_matrix(y_true, y_pred, labels=None, normalize=False):
    """
    Quick function to create confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize (bool): Whether to normalize
        
    Returns:
        plotly.Figure: Confusion matrix plot
    """
    viz = Visualizer()
    return viz.plot_confusion_matrix(y_true, y_pred, labels, normalize=normalize)


    def _create_fallback_comparison_plot(self, plot_df, metric, title):
        """
        Create fallback matplotlib comparison plot when Plotly fails
        
        Args:
            plot_df (pd.DataFrame): Cleaned comparison data
            metric (str): Metric to plot
            title (str): Plot title
            
        Returns:
            matplotlib.Figure: Comparison plot
        """
        try:
            fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.4)))
            
            # Create horizontal bar chart
            bars = ax.barh(plot_df['Model'], plot_df[metric], 
                          color=self.colors[:len(plot_df)])
            
            ax.set_xlabel(f'{metric} Score')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (idx, row) in enumerate(plot_df.iterrows()):
                ax.text(row[metric] + 0.001, i, f'{row[metric]:.3f}',
                       va='center', fontsize=10)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Fallback plot creation failed: {e}")
            return None


def quick_model_comparison(comparison_df, metric='Val Accuracy'):
    """
    Quick function to create model comparison plot
    
    Args:
        comparison_df (pd.DataFrame): Model comparison dataframe
        metric (str): Metric to compare
        
    Returns:
        plotly.Figure: Model comparison plot
    """
    viz = Visualizer()
    return viz.plot_model_comparison(comparison_df, metric)
