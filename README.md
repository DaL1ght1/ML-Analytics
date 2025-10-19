# 🤖 Universal ML Analytics Platform

A comprehensive, production-ready machine learning web application that works with **ANY dataset** for both classification and regression tasks. Originally built for student dropout prediction, now evolved into a universal ML platform with automatic problem detection, advanced model training, and full interpretability features.

## ✨ Features

### 🚀 **Universal ML Capabilities**
- **🎯 Any Dataset Support**: Works with any CSV/Excel file from any domain (business, healthcare, finance, research)
- **🧠 Smart Problem Detection**: Automatically detects classification vs regression tasks
- **📊 Dynamic Model Selection**: Loads appropriate models based on problem type
- **📊 Interactive Data Upload**: Support for CSV and Excel files with any column names
- **⚙️ Automated Data Preprocessing**: Missing value handling, feature engineering, mixed data types
- **🎯 Target Value Filtering**: Exclude specific classes to focus on binary vs multiclass problems
- **📈 Comprehensive Evaluation**: Problem-appropriate metrics (Accuracy/F1 for classification, R²/RMSE for regression)
- **🔍 Model Interpretability**: SHAP and LIME integration for explainable AI
- **💾 Experiment History**: Complete tracking of datasets, configurations, and training results
- **🎨 Professional UI**: Clean, responsive Streamlit interface that adapts to your data

### 🤖 **Supported Models**

#### Classification Models (Auto-selected for categorical targets):
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree Classifier
- Gradient Boosting Classifier

#### Regression Models (Auto-selected for continuous targets):
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor

### 📊 **Visualization & Analysis**
- Interactive Plotly charts and graphs
- Feature importance analysis
- Model performance comparison
- ROC curves and confusion matrices
- Training time vs accuracy analysis
- Data distribution plots

### 🔬 **Model Interpretability**
- **SHAP (SHapley Additive exPlanations)**: Global and local explanations
- **LIME (Local Interpretable Model-agnostic Explanations)**: Individual prediction explanations
- Waterfall plots for feature contribution analysis
- Model comparison and feature importance ranking

### 🎆 **Advanced Features**
- **🧠 Auto Problem Detection**: Binary classification, multiclass classification, or regression
- **🎯 Smart Filtering**: Exclude target values to create focused datasets (e.g., binary from multiclass)
- **📊 Dynamic Metrics**: Shows relevant metrics based on detected problem type
- **💾 Upload History**: Track all datasets, configurations, and training results
- **🔄 Configuration Reload**: Reuse previous experimental setups
- **📤 Export Results**: Download model comparisons and history
- **🔍 Mixed Data Types**: Automatic handling of numeric + categorical features
- **⚙️ Smart Preprocessing**: Auto-detects stratification needs, handles target encoding

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- [UV package manager](https://github.com/astral-sh/uv) (recommended)

### Quick Setup with UV

1. **Clone or download the project**:
```bash
git clone <repository-url>
cd student_dropout
```

2. **Install dependencies with UV**:
```bash
# Install UV if you haven't already
pip install uv

# Create virtual environment and install dependencies
uv sync
```

3. **Activate the environment**:
```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### Alternative Setup with pip

If you prefer using pip:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Step-by-Step Workflow

#### 1. **Data Upload** 📊
- Navigate to the "Data Upload" page
- Upload your CSV or Excel file containing **any tabular data**
- Select the target variable (what you want to predict)
- **Optional**: Exclude specific target values to focus your analysis
- Review data preview, target distribution, and basic statistics
- Save dataset configuration for future reference

#### 2. **Data Overview** 🔍
- Explore dataset information and statistics
- Review feature types (numeric vs categorical)
- Analyze data quality and distributions

#### 3. **Model Training** 🤖
- **Automatic Problem Detection**: App detects classification vs regression
- **Smart Model Loading**: Appropriate models loaded automatically
- Configure training parameters (test size, random state)
- Select specific models to train (or use all available)
- Adjust cross-validation settings
- Click "Start Training" and monitor progress
- Review training summary with problem-appropriate metrics
- Best model automatically identified using relevant metric

#### 4. **Model Evaluation** 📈
- **Dynamic Metrics Display**: See relevant metrics based on problem type
  - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - **Regression**: R², RMSE, MAE, MAPE
- Select models for detailed evaluation
- Compare model performance with appropriate visualizations
- **Classification**: Confusion matrices, ROC curves, classification reports
- **Regression**: Residual plots, prediction vs actual comparisons
- Examine feature importance (for supported models)
- Compare all models side-by-side with relevant metrics

#### 5. **Model Interpretation** 🔬
- Choose interpretation type:
  - **Global Feature Importance**: SHAP and tree-based importance
  - **Individual Explanations**: SHAP waterfall and LIME plots
  - **Model Comparison**: Side-by-side interpretability analysis
- Explore feature contributions to predictions
- Understand model decision-making process

#### 6. **History & Experiment Tracking** 💾
- **Comprehensive History**: View all uploaded datasets and training results
- **Detailed Experiment Logs**: Dataset info, configurations, model performance
- **Filter & Sort**: Find specific experiments by date, performance, or dataset
- **Configuration Reload**: Reuse previous dataset configurations
- **Export Functionality**: Download history and results as CSV
- **Cleanup Options**: Clear history or remove untrained entries

## 📁 Project Structure

```
student_dropout/
├── app.py                          # Main Streamlit application
├── pyproject.toml                  # UV project configuration
├── README.md                       # This file
├── notebooks/
│   └── Student_Dropout (2).ipynb  # Original research notebook
├── static/
│   └── css/
│       └── style.css               # Custom CSS styling
└── utils/                          # Core utility modules
    ├── __init__.py
    ├── preprocessing.py            # Data preprocessing pipeline
    ├── models.py                   # ML model training and evaluation
    ├── visualization.py            # Interactive plotting and charts
    ├── interpretation.py           # SHAP and LIME model interpretation
    └── {preprocessing,models,visualization,interpretation}/
        └── __init__.py
```

## 🎯 Key Components

### DataPreprocessor
- **Auto Problem Detection**: Determines classification vs regression
- **Smart Data Cleaning**: Handles missing values, duplicates, mixed types
- **Universal Feature Engineering**: Works with any feature names and types
- **Intelligent Splitting**: Auto-detects stratification needs
- **Target Filtering**: Exclude specific values for focused analysis
- **Robust Encoding**: Handles categorical + numeric features seamlessly

### ModelTrainer
- **Dynamic Model Loading**: Selects models based on detected problem type
- **Universal Metrics**: Calculates appropriate metrics (classification/regression)
- **Smart Best Model Selection**: Uses relevant metric (Accuracy/R²/RMSE)
- **Cross-validation**: Adapts to problem type
- **Training Progress**: Real-time updates with time tracking
- **Model Comparison**: Problem-appropriate performance tables

### Visualizer
- Interactive Plotly visualizations
- Matplotlib fallback options
- Confusion matrices and ROC curves
- Feature importance and model comparison plots

### ModelInterpreter
- SHAP explainer initialization and management
- LIME explainer for local interpretations
- Waterfall plots and summary visualizations
- Global and local feature importance analysis

## 📊 Dataset Requirements

The platform works with **ANY tabular dataset**! Here's what's supported:

### 🎯 Target Variable (What You Want to Predict):
- **Classification**: Categorical outcomes (e.g., "Yes/No", "High/Medium/Low", "Class A/B/C")
- **Regression**: Continuous numeric values (e.g., prices, scores, temperatures)
- **Any Column Name**: No naming restrictions - use any column as target
- **Auto-Detection**: App automatically determines if it's classification or regression

### 📈 Feature Types (All Automatically Handled):
- **Numeric Features**: Any continuous or discrete numbers
- **Categorical Features**: Text labels, categories, yes/no values
- **Mixed Data Types**: Combination of numeric and categorical
- **Boolean Features**: True/false, 1/0 values

### Data Quality:
- **Missing Values**: Automatically handled (median for numeric, mode for categorical)
- **Duplicates**: Automatically removed during preprocessing
- **Encoding**: Automatic one-hot encoding for categorical variables

### 🌍 Example Dataset Formats:

**Classification Example (Customer Churn):**
```csv
monthly_charges,total_charges,tenure,contract_length,support_calls,churn
85.50,1200.30,24,Month-to-month,3,Yes
65.20,2400.80,12,One year,1,No
95.10,800.50,36,Two year,0,No
...
```

**Regression Example (House Prices):**
```csv
sqft,bedrooms,bathrooms,age,location_score,garage_size,price
2500,4,3.5,10,8.2,2,350000
1800,3,2,5,7.5,1,280000
3200,5,4,15,9.1,3,450000
...
```

**Mixed Data Types Example (Employee Performance):**
```csv
experience_years,salary,department,education,remote_work,training_hours,performance
5.2,75000,Engineering,Master,Yes,25,Above Average
2.1,45000,Sales,Bachelor,No,10,Average
8.7,95000,Marketing,PhD,Yes,40,Above Average
...
```

## 🎆 Supported Use Cases

The Universal ML Platform works across **all domains and industries**:

### 💼 Business & Finance
- **Customer Churn Prediction** (Classification)
- **Sales Forecasting** (Regression) 
- **Fraud Detection** (Classification)
- **Price Optimization** (Regression)
- **Lead Scoring** (Classification)
- **Demand Forecasting** (Regression)

### 🏭 Healthcare & Medicine
- **Disease Diagnosis** (Classification)
- **Treatment Outcome Prediction** (Classification)
- **Drug Dosage Optimization** (Regression)
- **Patient Risk Assessment** (Classification)
- **Hospital Readmission** (Classification)
- **Medical Cost Estimation** (Regression)

### 🎓 Education & Research
- **Student Performance Prediction** (Regression/Classification)
- **Dropout Risk Assessment** (Classification)
- **Course Recommendation** (Classification)
- **Grade Prediction** (Regression)
- **Scholarship Eligibility** (Classification)
- **Research Outcome Prediction** (Various)

### 🏖️ Real Estate & Property
- **House Price Prediction** (Regression)
- **Property Investment ROI** (Regression)
- **Mortgage Approval** (Classification)
- **Market Trend Analysis** (Classification/Regression)
- **Property Value Assessment** (Regression)

### 📦 E-commerce & Retail
- **Product Recommendation** (Classification)
- **Inventory Management** (Regression)
- **Customer Segmentation** (Classification)
- **Price Elasticity** (Regression)
- **Return Prediction** (Classification)

### 🏭 Manufacturing & IoT
- **Quality Control** (Classification)
- **Predictive Maintenance** (Classification)
- **Production Optimization** (Regression)
- **Defect Detection** (Classification)
- **Energy Consumption** (Regression)

### 🎨 Marketing & Social Media
- **Campaign Performance** (Regression)
- **Sentiment Analysis** (Classification)
- **Influencer Impact** (Regression)
- **Content Engagement** (Classification/Regression)
- **Ad Click Prediction** (Classification)

## ⚙️ Configuration Options

### Training Parameters:
- **Test Size**: Proportion of data reserved for testing (0.1-0.3)
- **Random State**: Seed for reproducible results
- **Cross Validation**: K-fold validation settings
- **Model Selection**: Choose specific algorithms to train

### Visualization Settings:
- **Color Schemes**: Multiple professional color palettes
- **Plot Types**: Interactive Plotly vs static Matplotlib
- **Feature Limits**: Control number of features displayed

### Interpretation Settings:
- **SHAP Samples**: Maximum samples for SHAP computation
- **Explanation Types**: Global vs local interpretations
- **Feature Count**: Number of top features to analyze

## 🔧 Customization

### Adding New Models
To add new ML models, edit `utils/models.py`:

```python
# In ModelTrainer._initialize_models()
self.models['New Model'] = NewModelClass(
    param1=value1,
    random_state=self.random_state
)
```

### Custom Visualizations
Add new visualization functions in `utils/visualization.py`:

```python
def plot_custom_chart(self, data, title="Custom Chart"):
    # Your custom plotting logic
    return fig
```

### Additional Preprocessing
Extend the preprocessing pipeline in `utils/preprocessing.py`:

```python
def custom_feature_engineering(self, df):
    # Your custom feature engineering
    return df
```

## 📈 Performance Optimization

### For Large Datasets:
- **Sampling**: Use subset of data for SHAP computations
- **Model Selection**: Choose faster models for initial exploration
- **Parallel Processing**: Enable multi-core processing where available

### For Production Deployment:
- **Caching**: Enable Streamlit caching for data and models
- **Resource Limits**: Set memory and CPU constraints
- **Model Persistence**: Save and load trained models

## 🐛 Troubleshooting

### Common Issues:

**ImportError: Module not found**
```bash
# Ensure all dependencies are installed
uv sync
# or
pip install -r requirements.txt
```

**SHAP/LIME not available warnings**
```bash
# Install optional dependencies
pip install shap lime
```

**Memory errors with large datasets**
- Reduce SHAP sample size in settings
- Use fewer models for training
- Consider data sampling

**Slow performance**
- Disable cross-validation for faster training
- Reduce number of features
- Use simpler models for exploration

### Getting Help:
1. Check the error messages in the Streamlit interface
2. Review the console output for detailed error traces
3. Ensure your dataset format matches requirements
4. Verify all dependencies are properly installed

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional ML algorithms
- New visualization types
- Enhanced preprocessing features
- Performance optimizations
- UI/UX improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the responsive web interface
- Powered by [scikit-learn](https://scikit-learn.org/) for comprehensive machine learning
- Interactive visualizations with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
- Model interpretability via [SHAP](https://shap.readthedocs.io/) and [LIME](https://lime-ml.readthedocs.io/)
- Modern package management with [UV](https://github.com/astral-sh/uv)
- Robust data processing with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)

---

## 🎆 From Student Dropout to Universal ML Platform

**What started as a student dropout prediction tool has evolved into a comprehensive, universal machine learning platform.** 

**Key Evolution:**
- ✅ **Universal Dataset Support**: Works with any CSV/Excel from any domain
- ✅ **Auto Problem Detection**: Automatically detects classification vs regression
- ✅ **Dynamic Model Selection**: Loads appropriate algorithms based on problem type
- ✅ **Smart Preprocessing**: Handles any feature types and target values
- ✅ **Experiment Tracking**: Complete history and configuration management
- ✅ **Production Ready**: Robust error handling and user experience

*Perfect for data scientists, researchers, business analysts, students, and anyone wanting to apply machine learning to their data without coding complexity.*

**🚀 Ready to analyze your data? Just upload, select your target, and let the AI do the rest!**
