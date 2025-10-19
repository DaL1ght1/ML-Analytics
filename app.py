import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Page config
st.set_page_config(
    page_title="Universal ML Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utility modules
try:
    from utils.preprocessing import DataPreprocessor
    from utils.models import ModelTrainer
    from utils.visualization import Visualizer
    from utils.interpretation import ModelInterpreter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.exception(e)
    st.stop()

def main():
    """Main application function"""
    
    # Title and description
    st.title("🤖 Universal ML Analytics Platform")
    st.markdown("""
    **Comprehensive Machine Learning Analysis for Any Dataset**
    
    Upload your dataset and get automated:
    • **Smart Data Analysis** - Automatic detection of data types and problem type (classification/regression)
    • **Multi-Model Training** - 7 different ML algorithms with automatic optimization
    • **Advanced Evaluation** - Comprehensive metrics, visualizations, and model comparison
    • **AI Interpretability** - SHAP and LIME explanations for any model
    • **Experiment Tracking** - Complete history of your ML experiments
    
    **Works with any tabular dataset** - Whether it's business, finance, healthcare, research, or any domain!
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a page:",
        ["Data Upload", "Data Overview", "Model Training", "Model Evaluation", "Model Interpretation", "History"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []
    if 'excluded_values' not in st.session_state:
        st.session_state.excluded_values = []
    
    # Page routing
    if page == "Data Upload":
        data_upload_page()
    elif page == "Data Overview":
        data_overview_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Model Evaluation":
        model_evaluation_page()
    elif page == "Model Interpretation":
        model_interpretation_page()
    elif page == "History":
        history_page()

def data_upload_page():
    """Data upload and basic preprocessing page"""
    st.header("📊 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset with features and a target variable for machine learning analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Target selection
            st.subheader("Target Variable Selection")
            target_column = st.selectbox(
                "Select the target variable:",
                df.columns.tolist(),
                help="Choose the column you want to predict (the target/dependent variable)"
            )
            
            if target_column:
                st.session_state.target_column = target_column
                
                # Show target distribution
                st.write(f"**Original Target Distribution:**")
                target_counts = df[target_column].value_counts()
                st.bar_chart(target_counts)
                
                # Target value filtering
                st.subheader("🎯 Target Value Filtering")
                unique_values = df[target_column].unique().tolist()
                
                # Multi-select for excluding values
                # Only use previous excluded values if they exist in current dataset
                valid_defaults = [val for val in st.session_state.excluded_values if val in unique_values]
                
                excluded_values = st.multiselect(
                    "Select target values to EXCLUDE from training:",
                    unique_values,
                    default=valid_defaults,
                    help="Selected values will be filtered out before training. This helps focus on specific classifications (e.g., binary instead of multiclass)."
                )
                
                st.session_state.excluded_values = excluded_values
                
                # Show filtered distribution if exclusions are made
                if excluded_values:
                    filtered_df = df[~df[target_column].isin(excluded_values)]
                    st.write(f"**Filtered Target Distribution** (excluding {excluded_values}):")
                    if len(filtered_df) > 0:
                        filtered_counts = filtered_df[target_column].value_counts()
                        st.bar_chart(filtered_counts)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Samples", len(df))
                        with col2:
                            st.metric("Filtered Samples", len(filtered_df))
                            
                        st.info(f"💡 {len(df) - len(filtered_df)} samples will be excluded from training ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)")
                    else:
                        st.error("❌ No samples remain after filtering. Please adjust your exclusions.")
                else:
                    st.info("ℹ️ No values excluded - all data will be used for training.")
                
                # Save to upload history
                if st.button("💾 Save Dataset Configuration"):
                    history_entry = {
                        'timestamp': pd.Timestamp.now(),
                        'filename': uploaded_file.name,
                        'original_shape': df.shape,
                        'target_column': target_column,
                        'original_distribution': target_counts.to_dict(),
                        'excluded_values': excluded_values.copy(),
                        'filtered_shape': (len(df[~df[target_column].isin(excluded_values)]) if excluded_values else len(df), df.shape[1]),
                        'filtered_distribution': (df[~df[target_column].isin(excluded_values)][target_column].value_counts().to_dict() if excluded_values else target_counts.to_dict())
                    }
                    st.session_state.upload_history.append(history_entry)
                    st.success("✅ Dataset configuration saved to history!")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a dataset to get started.")
    
    # Display upload history
    if st.session_state.upload_history:
        st.subheader("📊 Upload History")
        
        # Create expandable history sections
        for i, entry in enumerate(reversed(st.session_state.upload_history)):
            with st.expander(f"📝 {entry['filename']} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}" + 
                           (f" (Excluded: {', '.join(entry['excluded_values'])})" if entry['excluded_values'] else "")):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Info:**")
                    st.write(f"- Original Shape: {entry['original_shape'][0]} rows × {entry['original_shape'][1]} columns")
                    st.write(f"- Target Column: {entry['target_column']}")
                    st.write(f"- Filtered Shape: {entry['filtered_shape'][0]} rows × {entry['filtered_shape'][1]} columns")
                    if entry['excluded_values']:
                        st.write(f"- Excluded Values: {', '.join(entry['excluded_values'])}")
                
                with col2:
                    st.write("**Target Distribution:**")
                    if entry['excluded_values']:
                        st.write("*Filtered:*")
                        for value, count in entry['filtered_distribution'].items():
                            st.write(f"- {value}: {count}")
                    else:
                        st.write("*Original:*")
                        for value, count in entry['original_distribution'].items():
                            st.write(f"- {value}: {count}")
                
                # Display training results if available
                if 'training_results' in entry:
                    st.write("**Training Results:**")
                    training = entry['training_results']
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write(f"- Training Date: {training['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"- Models Trained: {', '.join(training['models_trained'])}")
                        if training['best_model']:
                            st.write(f"- Best Model: {training['best_model']['name']} ({training['best_model']['score']:.3f})")
                    
                    with col4:
                        st.write(f"- Test Size: {training['training_config']['test_size']}")
                        st.write(f"- Random State: {training['training_config']['random_state']}")
                        st.write(f"- Cross Validation: {training['training_config']['cross_validation']}")
                    
                    # Show model comparison table
                    if training['comparison_results']:
                        st.write("**Model Performance:**")
                        results_df = pd.DataFrame(training['comparison_results'])
                        # Display key metrics only
                        display_cols = ['Model', 'Val Accuracy', 'Test Accuracy', 'Val F1 Score', 'Test F1 Score']
                        available_cols = [col for col in display_cols if col in results_df.columns]
                        if available_cols:
                            st.dataframe(results_df[available_cols], use_container_width=True)
                
                # Option to reload this configuration
                if st.button(f"🔄 Use This Configuration", key=f"reload_{i}"):
                    st.session_state.excluded_values = entry['excluded_values'].copy()
                    st.success(f"✅ Configuration from {entry['filename']} loaded!")
                    st.rerun()
        
        # Clear history option
        if st.button("🗑️ Clear History"):
            st.session_state.upload_history = []
            st.success("✅ Upload history cleared!")
            st.rerun()

def data_overview_page():
    """Data exploration and overview page"""
    st.header("🔍 Data Overview")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first in the Data Upload page.")
        return
    
    df = st.session_state.data
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    
    # Feature types
    st.subheader("Feature Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numeric Features:**")
        st.write(numeric_cols)
    with col2:
        st.write("**Categorical Features:**")
        st.write(categorical_cols)

def model_training_page():
    """Model training page"""
    st.header("🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first.")
        return
    
    if 'target_column' not in st.session_state:
        st.warning("⚠️ Please select a target column in the Data Upload page first.")
        return
    
    df = st.session_state.data
    target_column = st.session_state.target_column
    
    # Apply data filtering if exclusions are set
    excluded_values = st.session_state.excluded_values
    if excluded_values:
        original_len = len(df)
        df = df[~df[target_column].isin(excluded_values)].copy()
        filtered_len = len(df)
        
        st.info(f"📊 **Data Filtering Applied**: Excluded {', '.join(excluded_values)} - {original_len - filtered_len} samples removed ({(original_len - filtered_len)/original_len*100:.1f}%)")
        
        if filtered_len == 0:
            st.error("❌ No samples remain after filtering. Please adjust your exclusions in the Data Upload page.")
            return
        elif filtered_len < 50:
            st.warning("⚠️ Very few samples remaining after filtering. Consider adjusting your exclusions for better model performance.")
    
    st.subheader("📋 Training Configuration")
    
    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
        random_state = st.number_input("Random State", 1, 100, 42)
    with col2:
        max_samples_shap = st.number_input("Max Samples for SHAP", 50, 500, 100)
        cross_validation = st.checkbox("Use Cross Validation", True)
    
    # Model selection
    st.subheader("🎯 Model Selection")
    available_models = ['Logistic Regression', 'Random Forest', 'SVM', 
                       'K-Nearest Neighbors', 'Naive Bayes', 'Decision Tree', 'Gradient Boosting']
    selected_models = st.multiselect(
        "Select models to train:",
        available_models,
        default=available_models[:4]  # Default to first 4 models
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train.")
        return
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        try:
            with st.spinner("Training models... This may take a few minutes."):
                # Initialize preprocessor
                preprocessor = DataPreprocessor(
                    target_column=target_column,
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Clean and prepare data
                df_clean = preprocessor.clean_data(df)
                df_features = preprocessor.create_features(df_clean)
                
                # Prepare features and target
                X, y, pipeline = preprocessor.prepare_features(df_features)
                y_encoded = preprocessor.encode_target(y)
                
                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y_encoded)
                
                # Fit preprocessor and transform data
                X_train_processed, X_val_processed, X_test_processed = preprocessor.fit_transform(
                    X_train, X_val, X_test
                )
                
                # Detect problem type from preprocessed data
                problem_type_info = preprocessor.detect_problem_type(df)
                problem_type = problem_type_info['type']
                
                # Initialize trainer with problem type
                trainer = ModelTrainer(problem_type=problem_type, random_state=random_state)
                # Filter models based on selection
                trainer.models = {k: v for k, v in trainer.models.items() if k in selected_models}
                
                # Train models
                results = trainer.train_models(
                    X_train_processed, y_train,
                    X_val_processed, y_val,
                    X_test_processed, y_test
                )
                
                # Store results in session state
                st.session_state.preprocessor = preprocessor
                st.session_state.trainer = trainer
                st.session_state.model_results = results
                st.session_state.X_train = X_train_processed
                st.session_state.X_val = X_val_processed
                st.session_state.X_test = X_test_processed
                st.session_state.y_train = y_train
                st.session_state.y_val = y_val
                st.session_state.y_test = y_test
                
                st.success("✅ Model training completed successfully!")
                
                # Display training summary
                st.subheader("📊 Training Summary")
                comparison_df = trainer.get_model_comparison()
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best model
                best_model_info = None
                try:
                    best_model_name, best_model, best_score = trainer.get_best_model()
                    metric_name = "Validation R²" if trainer.is_regression else "Validation Accuracy"
                    st.info(f"🏆 Best Model: **{best_model_name}** ({metric_name}: {best_score:.3f})")
                    best_model_info = {'name': best_model_name, 'score': best_score}
                except:
                    st.info("Could not determine best model automatically.")
                
                # Save training results to history
                if st.session_state.upload_history:
                    # Update the latest history entry with training results
                    latest_entry = st.session_state.upload_history[-1]
                    latest_entry['training_results'] = {
                        'timestamp': pd.Timestamp.now(),
                        'models_trained': list(selected_models),
                        'comparison_results': comparison_df.to_dict('records'),
                        'best_model': best_model_info,
                        'training_config': {
                            'test_size': test_size,
                            'random_state': random_state,
                            'cross_validation': cross_validation
                        }
                    }
                    st.success("💾 Training results saved to upload history!")
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.exception(e)
    
    # Display existing results if available
    if 'trainer' in st.session_state and st.session_state.trainer.is_trained:
        st.subheader("📈 Training Results")
        comparison_df = st.session_state.trainer.get_model_comparison()
        st.dataframe(comparison_df, use_container_width=True)
        
        # Training time vs accuracy plot
        if len(comparison_df) > 0:
            try:
                viz = Visualizer()
                training_times = [r.get('training_time', 0) for r in st.session_state.model_results.values() if 'training_time' in r]
                accuracies = [r.get('val_accuracy', 0) for r in st.session_state.model_results.values() if 'val_accuracy' in r]
                model_names = [name for name, r in st.session_state.model_results.items() if 'training_time' in r]
                
                if training_times and accuracies:
                    fig = viz.plot_training_history(training_times, accuracies, model_names)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create training history plot: {e}")

def model_evaluation_page():
    """Model evaluation page"""
    st.header("📈 Model Evaluation")
    
    if 'trainer' not in st.session_state or not st.session_state.trainer.is_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        return
    
    trainer = st.session_state.trainer
    
    # Model selection for detailed evaluation
    st.subheader("🎯 Select Model for Detailed Evaluation")
    available_models = list(trainer.trained_models.keys())
    selected_model = st.selectbox("Choose a model:", available_models)
    
    if selected_model:
        col1, col2 = st.columns(2)
        
        with col1:
            # Model comparison chart
            st.subheader("📊 Model Comparison")
            comparison_df = trainer.get_model_comparison()
            
            # Metric selection based on problem type
            if trainer.is_regression:
                metrics = ['Val R²', 'Test R²', 'Val RMSE', 'Test RMSE', 'Val MAE', 'Test MAE', 'CV Mean']
            else:
                metrics = ['Val Accuracy', 'Test Accuracy', 'Val F1 Score', 'Test F1 Score', 'CV Mean']
            selected_metric = st.selectbox("Select metric for comparison:", metrics)
            
            if selected_metric in comparison_df.columns:
                viz = Visualizer()
                fig = viz.plot_model_comparison(comparison_df, selected_metric)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance for tree-based models
            st.subheader("🌳 Feature Importance")
            importance_df = trainer.get_feature_importance(
                selected_model, 
                st.session_state.preprocessor.feature_names if 'preprocessor' in st.session_state else None
            )
            
            if not importance_df.empty and 'message' not in importance_df.columns:
                top_n = st.slider("Number of top features to show:", 5, 20, 10)
                viz = Visualizer()
                fig = viz.plot_feature_importance(importance_df, f"{selected_model} Feature Importance", top_n)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show feature importance table
                st.dataframe(importance_df.head(top_n), use_container_width=True)
            else:
                st.info(f"{selected_model} does not support feature importance visualization.")
        
        # Detailed evaluation metrics
        if 'X_test' in st.session_state and 'y_test' in st.session_state:
            st.subheader("🔍 Detailed Evaluation Metrics")
            
            try:
                detailed_eval = trainer.evaluate_model(
                    selected_model,
                    st.session_state.X_test,
                    st.session_state.y_test
                )
                
                # Display metrics based on problem type
                if trainer.is_regression:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² Score", f"{detailed_eval.get('r2_score', 'N/A')}")
                    with col2:
                        st.metric("RMSE", f"{detailed_eval.get('rmse', 'N/A')}")
                    with col3:
                        st.metric("MAE", f"{detailed_eval.get('mae', 'N/A')}")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{detailed_eval['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{detailed_eval['precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{detailed_eval['recall']:.3f}")
                    with col4:
                        st.metric("F1 Score", f"{detailed_eval['f1_score']:.3f}")
                
                # Confusion matrix (only for classification)
                if not trainer.is_regression:
                    if 'preprocessor' in st.session_state:
                        class_names = st.session_state.preprocessor.target_classes
                    else:
                        class_names = None
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Confusion Matrix")
                        viz = Visualizer()
                        cm_fig = viz.plot_confusion_matrix(
                            st.session_state.y_test,
                            detailed_eval['predictions'],
                            labels=class_names,
                            title=f"{selected_model} Confusion Matrix"
                        )
                        st.plotly_chart(cm_fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🎯 Normalized Confusion Matrix")
                        cm_fig_norm = viz.plot_confusion_matrix(
                            st.session_state.y_test,
                            detailed_eval['predictions'],
                            labels=class_names,
                            title=f"{selected_model} Normalized Confusion Matrix",
                            normalize=True
                        )
                        st.plotly_chart(cm_fig_norm, use_container_width=True)
                
                # Classification report
                st.subheader("📋 Classification Report")
                if 'classification_report' in detailed_eval:
                    report_df = pd.DataFrame(detailed_eval['classification_report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                
                # ROC Curves (if available)
                if 'prediction_probabilities' in detailed_eval and detailed_eval['prediction_probabilities'] is not None:
                    st.subheader("📈 ROC Curve")
                    try:
                        models_data = {
                            selected_model: (st.session_state.y_test, detailed_eval['prediction_probabilities'])
                        }
                        viz = Visualizer()
                        roc_fig = viz.plot_roc_curves(models_data, f"{selected_model} ROC Curve")
                        st.plotly_chart(roc_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not plot ROC curve: {e}")
                
            except Exception as e:
                st.error(f"Error in detailed evaluation: {e}")
        
        # Model comparison across all models
        st.subheader("🔄 All Models ROC Comparison")
        if 'X_test' in st.session_state and 'y_test' in st.session_state:
            try:
                models_data = {}
                for model_name in trainer.trained_models.keys():
                    try:
                        eval_result = trainer.evaluate_model(model_name, st.session_state.X_test, st.session_state.y_test)
                        if 'prediction_probabilities' in eval_result and eval_result['prediction_probabilities'] is not None:
                            models_data[model_name] = (st.session_state.y_test, eval_result['prediction_probabilities'])
                    except:
                        continue
                
                if models_data:
                    viz = Visualizer()
                    roc_comparison_fig = viz.plot_roc_curves(models_data, "All Models ROC Comparison")
                    st.plotly_chart(roc_comparison_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create ROC comparison: {e}")

def model_interpretation_page():
    """Model interpretation page"""
    st.header("🔬 Model Interpretation")
    
    if 'trainer' not in st.session_state or not st.session_state.trainer.is_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        return
    
    trainer = st.session_state.trainer
    
    # Model selection for interpretation
    st.subheader("🎯 Select Model for Interpretation")
    available_models = list(trainer.trained_models.keys())
    selected_model = st.selectbox("Choose a model:", available_models, key="interp_model")
    
    if selected_model and 'X_train' in st.session_state:
        model = trainer.trained_models[selected_model]
        
        # Initialize interpreter
        try:
            with st.spinner("Initializing model interpreter..."):
                feature_names = st.session_state.preprocessor.feature_names if 'preprocessor' in st.session_state else None
                class_names = st.session_state.preprocessor.target_classes if 'preprocessor' in st.session_state else None
                
                interpreter = ModelInterpreter(
                    model=model,
                    X_train=st.session_state.X_train,
                    feature_names=feature_names,
                    class_names=class_names
                )
                
                st.success("✅ Interpreter initialized successfully!")
                
                # Interpretation options
                interpretation_type = st.radio(
                    "Select interpretation type:",
                    ["Global Feature Importance", "Individual Prediction Explanation", "Model Comparison"]
                )
                
                if interpretation_type == "Global Feature Importance":
                    st.subheader("🌍 Global Feature Importance")
                    
                    # SHAP global importance
                    if hasattr(interpreter, 'shap_explainer') and interpreter.shap_explainer is not None:
                        with st.spinner("Computing SHAP feature importance..."):
                            shap_importance = interpreter.get_shap_feature_importance(
                                st.session_state.X_test[:100],  # Use first 100 test samples
                                max_samples=50
                            )
                        
                        if not shap_importance.empty:
                            st.subheader("📊 SHAP Feature Importance")
                            
                            # Plot
                            viz = Visualizer()
                            top_n = st.slider("Number of top features:", 5, 25, 15, key="shap_top_n")
                            shap_fig = viz.plot_feature_importance(
                                shap_importance, 
                                f"{selected_model} SHAP Feature Importance",
                                top_n
                            )
                            st.plotly_chart(shap_fig, use_container_width=True)
                            
                            # Table
                            st.dataframe(shap_importance.head(top_n), use_container_width=True)
                        else:
                            st.warning("Could not compute SHAP feature importance.")
                    else:
                        st.warning("SHAP explainer not available for this model.")
                    
                    # Tree-based feature importance (if available)
                    importance_df = trainer.get_feature_importance(selected_model, feature_names)
                    if not importance_df.empty and 'message' not in importance_df.columns:
                        st.subheader("🌳 Tree-based Feature Importance")
                        viz = Visualizer()
                        tree_fig = viz.plot_feature_importance(
                            importance_df,
                            f"{selected_model} Built-in Feature Importance",
                            15
                        )
                        st.plotly_chart(tree_fig, use_container_width=True)
                
                elif interpretation_type == "Individual Prediction Explanation":
                    st.subheader("🔍 Individual Prediction Explanation")
                    
                    # Sample selection
                    max_samples = min(10, len(st.session_state.X_test))
                    sample_idx = st.selectbox(
                        "Select sample to explain:",
                        range(max_samples),
                        format_func=lambda x: f"Sample {x+1}"
                    )
                    
                    if sample_idx < len(st.session_state.X_test):
                        sample = st.session_state.X_test[sample_idx:sample_idx+1]
                        actual_label = st.session_state.y_test[sample_idx]
                        
                        # Show prediction
                        prediction = model.predict(sample)[0]
                        confidence = interpreter.get_prediction_confidence(sample)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Actual Label", actual_label)
                        with col2:
                            st.metric("Predicted Label", prediction)
                        with col3:
                            if 'confidence_scores' in confidence and confidence['confidence_scores'] is not None:
                                st.metric("Confidence", f"{confidence['confidence_scores'][0]:.3f}")
                        
                        # SHAP explanation
                        if hasattr(interpreter, 'shap_explainer') and interpreter.shap_explainer is not None:
                            st.subheader("📈 SHAP Explanation")
                            try:
                                with st.spinner("Computing SHAP values..."):
                                    shap_fig = interpreter.plot_shap_waterfall(sample, 0)
                                if shap_fig:
                                    st.pyplot(shap_fig)
                                else:
                                    st.warning("Could not generate SHAP waterfall plot.")
                            except Exception as e:
                                st.warning(f"SHAP explanation error: {e}")
                        
                        # LIME explanation
                        if hasattr(interpreter, 'lime_explainer') and interpreter.lime_explainer is not None:
                            st.subheader("🍋 LIME Explanation")
                            try:
                                with st.spinner("Computing LIME explanation..."):
                                    lime_exp = interpreter.explain_instance_lime(sample)
                                if lime_exp:
                                    lime_fig = interpreter.plot_lime_explanation(lime_exp)
                                    if lime_fig:
                                        st.plotly_chart(lime_fig, use_container_width=True)
                                else:
                                    st.warning("Could not generate LIME explanation.")
                            except Exception as e:
                                st.warning(f"LIME explanation error: {e}")
                        
                        # Show feature values for this sample
                        if feature_names:
                            st.subheader("📋 Feature Values for This Sample")
                            feature_values = pd.DataFrame({
                                'Feature': feature_names[:len(sample[0])],
                                'Value': sample[0]
                            })
                            st.dataframe(feature_values, use_container_width=True)
                
                elif interpretation_type == "Model Comparison":
                    st.subheader("⚖️ Model Interpretation Comparison")
                    
                    if len(available_models) >= 2:
                        # Select models to compare
                        models_to_compare = st.multiselect(
                            "Select models to compare:",
                            available_models,
                            default=available_models[:2]
                        )
                        
                        if len(models_to_compare) >= 2:
                            comparison_results = {}
                            
                            for model_name in models_to_compare:
                                try:
                                    model_obj = trainer.trained_models[model_name]
                                    model_interpreter = ModelInterpreter(
                                        model=model_obj,
                                        X_train=st.session_state.X_train,
                                        feature_names=feature_names,
                                        class_names=class_names
                                    )
                                    
                                    # Get feature importance
                                    importance = model_interpreter.get_shap_feature_importance(
                                        st.session_state.X_test[:50], max_samples=25
                                    )
                                    comparison_results[model_name] = importance
                                    
                                except Exception as e:
                                    st.warning(f"Could not interpret {model_name}: {e}")
                            
                            # Display comparison
                            if comparison_results:
                                st.subheader("📊 Feature Importance Comparison")
                                
                                # Create side-by-side comparison
                                cols = st.columns(len(comparison_results))
                                
                                for i, (model_name, importance_df) in enumerate(comparison_results.items()):
                                    with cols[i]:
                                        st.write(f"**{model_name}**")
                                        if not importance_df.empty:
                                            viz = Visualizer()
                                            fig = viz.plot_feature_importance(
                                                importance_df.head(10),
                                                f"{model_name}",
                                                10
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Please select at least 2 models to compare.")
                    else:
                        st.info("Need at least 2 trained models for comparison.")
                        
        except Exception as e:
            st.error(f"Error initializing interpreter: {e}")
            st.exception(e)

def history_page():
    """Comprehensive upload and training history page"""
    st.header("📊 Upload & Training History")
    
    if not st.session_state.upload_history:
        st.info("📍 No upload history available. Upload and configure datasets to start tracking history.")
        return
    
    # Summary statistics
    st.subheader("📈 Summary")
    total_uploads = len(st.session_state.upload_history)
    trained_datasets = len([entry for entry in st.session_state.upload_history if 'training_results' in entry])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Uploads", total_uploads)
    with col2:
        st.metric("Trained Datasets", trained_datasets)
    with col3:
        st.metric("Success Rate", f"{(trained_datasets/total_uploads*100) if total_uploads > 0 else 0:.1f}%")
    
    # Detailed history
    st.subheader("📋 Detailed History")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_trained_only = st.checkbox("Show only trained datasets", False)
    with col2:
        sort_by = st.selectbox("Sort by", ["Upload Date (Newest)", "Upload Date (Oldest)", "Filename"])
    
    # Filter and sort history
    filtered_history = st.session_state.upload_history.copy()
    if show_trained_only:
        filtered_history = [entry for entry in filtered_history if 'training_results' in entry]
    
    if sort_by == "Upload Date (Newest)":
        filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)
    elif sort_by == "Upload Date (Oldest)":
        filtered_history = sorted(filtered_history, key=lambda x: x['timestamp'])
    else:  # Filename
        filtered_history = sorted(filtered_history, key=lambda x: x['filename'])
    
    if not filtered_history:
        st.info("🔍 No entries match the current filters.")
        return
    
    # Display detailed entries
    for i, entry in enumerate(filtered_history):
        with st.expander(
            f"📝 {entry['filename']} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}" +
            (f" 🏆 TRAINED" if 'training_results' in entry else "") +
            (f" (Excluded: {', '.join(entry['excluded_values'])})" if entry['excluded_values'] else ""),
            expanded=i == 0  # Expand first entry by default
        ):
            # Dataset information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("📄 **Dataset Information**")
                st.write(f"- **File**: {entry['filename']}")
                st.write(f"- **Upload Date**: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- **Original Shape**: {entry['original_shape'][0]:,} rows × {entry['original_shape'][1]} columns")
                st.write(f"- **Target Column**: {entry['target_column']}")
                
                if entry['excluded_values']:
                    st.write(f"- **Excluded Values**: {', '.join(entry['excluded_values'])}")
                    st.write(f"- **Filtered Shape**: {entry['filtered_shape'][0]:,} rows × {entry['filtered_shape'][1]} columns")
                    reduction_pct = (entry['original_shape'][0] - entry['filtered_shape'][0]) / entry['original_shape'][0] * 100
                    st.write(f"- **Data Reduction**: {reduction_pct:.1f}%")
            
            with col2:
                st.write("🎯 **Target Distribution**")
                if entry['excluded_values']:
                    st.write("*After Filtering:*")
                    dist_data = entry['filtered_distribution']
                else:
                    st.write("*Original:*")
                    dist_data = entry['original_distribution']
                
                # Create a simple bar chart
                dist_df = pd.DataFrame(list(dist_data.items()), columns=['Value', 'Count'])
                st.bar_chart(dist_df.set_index('Value'))
                
                for value, count in dist_data.items():
                    percentage = count / sum(dist_data.values()) * 100
                    st.write(f"- **{value}**: {count:,} ({percentage:.1f}%)")
            
            # Training results if available
            if 'training_results' in entry:
                st.write("🏆 **Training Results**")
                training = entry['training_results']
                
                # Training info
                col3, col4 = st.columns(2)
                with col3:
                    st.write(f"- **Training Date**: {training['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- **Models Trained**: {len(training['models_trained'])} ({', '.join(training['models_trained'][:3])}{', ...' if len(training['models_trained']) > 3 else ''})")
                    if training['best_model']:
                        st.write(f"- **Best Model**: {training['best_model']['name']}")
                        st.write(f"- **Best Score**: {training['best_model']['score']:.4f}")
                
                with col4:
                    config = training['training_config']
                    st.write(f"- **Test Size**: {config['test_size']}")
                    st.write(f"- **Random State**: {config['random_state']}")
                    st.write(f"- **Cross Validation**: {config['cross_validation']}")
                
                # Model performance comparison
                if training['comparison_results']:
                    st.write("📉 **Model Performance Comparison**")
                    results_df = pd.DataFrame(training['comparison_results'])
                    
                    # Display metrics with color coding based on dataset type
                    if 'Val R²' in results_df.columns:  # Regression
                        display_cols = ['Model', 'Val R²', 'Test R²', 'Val RMSE', 'Test RMSE', 'Val MAE', 'Test MAE', 'Training Time (s)']
                    else:  # Classification
                        display_cols = ['Model', 'Val Accuracy', 'Test Accuracy', 'Val F1 Score', 'Test F1 Score', 'Training Time (s)']
                    available_cols = [col for col in display_cols if col in results_df.columns]
                    
                    if available_cols:
                        # Highlight best model
                        styled_df = results_df[available_cols].copy()
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Best performing models summary
                        if 'Val R²' in results_df.columns:
                            best_val_model = results_df.loc[results_df['Val R²'].idxmax(), 'Model']
                            st.success(f"🏅 Best Validation R²: **{best_val_model}** ({results_df['Val R²'].max():.4f})")
                        elif 'Val Accuracy' in results_df.columns:
                            best_val_model = results_df.loc[results_df['Val Accuracy'].idxmax(), 'Model']
                            st.success(f"🏅 Best Validation Accuracy: **{best_val_model}** ({results_df['Val Accuracy'].max():.4f})")
                        
                        if 'Test R²' in results_df.columns:
                            best_test_model = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
                            st.success(f"🏅 Best Test R²: **{best_test_model}** ({results_df['Test R²'].max():.4f})")
                        elif 'Test Accuracy' in results_df.columns:
                            best_test_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
                            st.success(f"🏅 Best Test Accuracy: **{best_test_model}** ({results_df['Test Accuracy'].max():.4f})")
            
            # Action buttons
            col5, col6, col7 = st.columns(3)
            
            with col5:
                if st.button(f"🔄 Reload Configuration", key=f"reload_config_{i}"):
                    st.session_state.excluded_values = entry['excluded_values'].copy()
                    st.success(f"✅ Configuration from {entry['filename']} reloaded!")
                    st.rerun()
            
            with col6:
                if 'training_results' in entry and st.button(f"📊 View Training Details", key=f"view_training_{i}"):
                    # Could expand to show more detailed training info
                    st.info("🔍 Training details displayed above. Switch to Model Evaluation page for interactive analysis.")
            
            with col7:
                if st.button(f"🗑️ Remove Entry", key=f"remove_{i}"):
                    st.session_state.upload_history.remove(entry)
                    st.success(f"✅ Entry for {entry['filename']} removed!")
                    st.rerun()
    
    # Bulk operations
    st.subheader("⚙️ Bulk Operations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Export History to CSV"):
            # Create export data
            export_data = []
            for entry in st.session_state.upload_history:
                row = {
                    'filename': entry['filename'],
                    'upload_date': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'target_column': entry['target_column'],
                    'original_rows': entry['original_shape'][0],
                    'original_cols': entry['original_shape'][1],
                    'excluded_values': ', '.join(entry['excluded_values']) if entry['excluded_values'] else 'None',
                    'filtered_rows': entry['filtered_shape'][0],
                    'filtered_cols': entry['filtered_shape'][1],
                    'trained': 'Yes' if 'training_results' in entry else 'No'
                }
                
                if 'training_results' in entry:
                    training = entry['training_results']
                    row['training_date'] = training['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    row['models_trained'] = ', '.join(training['models_trained'])
                    row['best_model'] = training['best_model']['name'] if training['best_model'] else 'N/A'
                    row['best_score'] = training['best_model']['score'] if training['best_model'] else 'N/A'
                else:
                    row.update({'training_date': '', 'models_trained': '', 'best_model': '', 'best_score': ''})
                
                export_data.append(row)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📎 Download CSV",
                data=csv,
                file_name=f"student_dropout_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗑️ Clear All History"):
            if st.session_state.get('confirm_clear', False):
                st.session_state.upload_history = []
                st.session_state.confirm_clear = False
                st.success("✅ All history cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm clearing all history.")
    
    with col3:
        if st.button("🗑️ Clear Untrained Entries"):
            untrained = [entry for entry in st.session_state.upload_history if 'training_results' not in entry]
            st.session_state.upload_history = [entry for entry in st.session_state.upload_history if 'training_results' in entry]
            st.success(f"✅ Removed {len(untrained)} untrained entries!")
            st.rerun()


if __name__ == "__main__":
    main()
