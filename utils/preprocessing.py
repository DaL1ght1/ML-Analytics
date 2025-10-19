import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Universal data preprocessing class for any ML dataset
    """
    
    def __init__(self, target_column=None, test_size=0.2, random_state=42):
        """
        Initialize the preprocessor
        
        Args:
            target_column (str): Name of the target column
            test_size (float): Size of test set (default 0.2)
            random_state (int): Random state for reproducibility
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
        self.problem_type = None  # 'binary_classification', 'multiclass_classification', 'regression'
        self.is_regression = False
        
    def analyze_data(self, df):
        """
        Analyze the dataset and provide summary statistics
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Data analysis summary
        """
        analysis = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'target_distribution': None
        }
        
        if self.target_column and self.target_column in df.columns:
            analysis['target_distribution'] = df[self.target_column].value_counts().to_dict()
            
        return analysis
    
    def detect_problem_type(self, df):
        """
        Automatically detect if this is a classification or regression problem
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Problem type information
        """
        if self.target_column not in df.columns:
            return {'type': 'unknown', 'reason': 'Target column not specified'}
        
        target_series = df[self.target_column]
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # Check if target is numeric
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        
        # Regression indicators
        if is_numeric:
            # Check if it's continuous values (many unique values relative to total)
            unique_ratio = unique_values / total_values
            
            if unique_ratio > 0.05:  # More than 5% unique values suggests continuous
                self.problem_type = 'regression'
                self.is_regression = True
                return {
                    'type': 'regression',
                    'unique_values': unique_values,
                    'unique_ratio': unique_ratio,
                    'reason': f'Continuous numeric target with {unique_values} unique values'
                }
        
        # Classification logic
        if unique_values == 2:
            self.problem_type = 'binary_classification'
            self.is_regression = False
            return {
                'type': 'binary_classification',
                'classes': target_series.unique().tolist(),
                'class_counts': target_series.value_counts().to_dict(),
                'reason': 'Exactly 2 unique values in target'
            }
        elif unique_values <= 20:  # Reasonable limit for multiclass
            self.problem_type = 'multiclass_classification'
            self.is_regression = False
            return {
                'type': 'multiclass_classification',
                'num_classes': unique_values,
                'classes': target_series.unique().tolist(),
                'class_counts': target_series.value_counts().to_dict(),
                'reason': f'{unique_values} unique values suitable for classification'
            }
        else:
            # Too many classes, probably regression even if not numeric
            self.problem_type = 'regression'
            self.is_regression = True
            return {
                'type': 'regression',
                'unique_values': unique_values,
                'reason': f'Too many unique values ({unique_values}) for classification, treating as regression'
            }
    
    def clean_data(self, df):

        df_clean = df.copy()

        # Remove duplicates
        df_clean = df_clean.drop_duplicates()

        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns

        # Fill numeric missing values with median
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        return df_clean

    def create_features(self, df):
        """
        Create additional features based on domain knowledge
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with new features
        """
        df_features = df.copy()
        
        # Example feature engineering based on common student data patterns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        # Create age groups if age column exists
        age_cols = [col for col in numeric_cols if 'age' in col.lower()]
        for age_col in age_cols:
            df_features[f'{age_col}_group'] = pd.cut(
                df_features[age_col], 
                bins=[0, 20, 25, 30, 100], 
                labels=['<20', '20-25', '25-30', '>30'],
                include_lowest=True
            )
        
        # Create GPA categories if GPA-like columns exist
        gpa_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['grade', 'gpa', 'score'])]
        for gpa_col in gpa_cols:
            if df_features[gpa_col].max() > 4:  # Assuming 100-point scale
                df_features[f'{gpa_col}_category'] = pd.cut(
                    df_features[gpa_col],
                    bins=[0, 60, 70, 80, 90, 100],
                    labels=['F', 'D', 'C', 'B', 'A'],
                    include_lowest=True
                )
            else:  # Assuming 4-point scale
                df_features[f'{gpa_col}_category'] = pd.cut(
                    df_features[gpa_col],
                    bins=[0, 2.0, 2.5, 3.0, 3.5, 4.0],
                    labels=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent'],
                    include_lowest=True
                )
        
        return df_features
    
    def prepare_features(self, df):
        """
        Prepare features by creating preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y, preprocessor)
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipelines
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Store feature information
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        return X, y, self.preprocessor
    
    def encode_target(self, y):
        """
        Encode target variable for classification or pass through for regression
        
        Args:
            y (pd.Series): Target variable
            
        Returns:
            np.array: Encoded target variable (or original for regression)
        """
        # Check if this is regression (continuous target) or classification
        unique_ratio = len(np.unique(y)) / len(y)
        
        if unique_ratio > 0.1:  # Likely regression - too many unique values to be classification
            # For regression, return the target as-is but as numpy array
            self.target_classes = None
            return y.values if hasattr(y, 'values') else np.array(y)
        else:
            # For classification, encode the labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.target_classes = self.label_encoder.classes_
            return y_encoded
    
    def split_data(self, X, y, stratify=None):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target variable
            stratify (bool): Whether to use stratified splitting. If None, auto-detect based on problem type.
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Auto-detect if stratification should be used
        if stratify is None:
            # Only stratify for classification (when y has relatively few unique values)
            unique_ratio = len(np.unique(y)) / len(y)
            stratify = unique_ratio < 0.1  # Use stratification if less than 10% unique values
        
        stratify_param = y if stratify else None
        
        try:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.test_size, 
                random_state=self.random_state,
                stratify=stratify_param
            )
            
            # Second split: train vs validation
            val_size = self.test_size / (1 - self.test_size)  # Adjust validation size
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size,
                random_state=self.random_state,
                stratify=stratify_temp
            )
        except ValueError as e:
            # If stratification fails, retry without stratification
            if "least populated class" in str(e):
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=self.test_size, 
                    random_state=self.random_state,
                    stratify=None
                )
                
                val_size = self.test_size / (1 - self.test_size)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size,
                    random_state=self.random_state,
                    stratify=None
                )
            else:
                raise e
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform(self, X_train, X_val, X_test):
        """
        Fit preprocessor on training data and transform all sets
        
        Args:
            X_train: Training features
            X_val: Validation features  
            X_test: Test features
            
        Returns:
            tuple: (X_train_processed, X_val_processed, X_test_processed)
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized. Call prepare_features first.")
        
        # Fit on training data and transform all sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        self.feature_names = self._get_feature_names()
        self.is_fitted = True
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def _get_feature_names(self):
        """
        Get feature names after preprocessing
        
        Returns:
            list: Feature names
        """
        feature_names = []
        
        try:
            # Numeric features
            feature_names.extend(self.numeric_features)
            
            # Categorical features (one-hot encoded)
            if self.categorical_features and len(self.categorical_features) > 0:
                try:
                    if hasattr(self.preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
                        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(
                            self.categorical_features
                        )
                        feature_names.extend(cat_features)
                    else:
                        # Fallback for older sklearn versions
                        categories = self.preprocessor.named_transformers_['cat'].categories_
                        for i, cat_feature in enumerate(self.categorical_features):
                            if i < len(categories):
                                for cat_value in categories[i]:
                                    feature_names.append(f"{cat_feature}_{cat_value}")
                except Exception as cat_error:
                    # Fallback: create generic categorical feature names
                    for cat_feature in self.categorical_features:
                        feature_names.append(f"{cat_feature}_encoded")
            
        except Exception as e:
            # Ultimate fallback: create generic feature names
            num_features = len(self.numeric_features) if hasattr(self, 'numeric_features') else 0
            num_cat_features = len(self.categorical_features) if hasattr(self, 'categorical_features') else 0
            
            feature_names = [f"feature_{i}" for i in range(num_features + num_cat_features)]
        
        return feature_names
    
    def get_preprocessing_summary(self):
        """
        Get summary of preprocessing steps
        
        Returns:
            dict: Preprocessing summary
        """
        if not self.is_fitted:
            return {"status": "Preprocessor not fitted"}
        
        return {
            "numeric_features": len(self.numeric_features),
            "categorical_features": len(self.categorical_features),
            "total_features_after_preprocessing": len(self.feature_names),
            "target_classes": self.target_classes.tolist() if hasattr(self, 'target_classes') else None,
            "preprocessing_steps": [
                "StandardScaler for numeric features",
                "OneHotEncoder for categorical features"
            ]
        }


def quick_preprocess(df, target_column, test_size=0.2, random_state=42):
    """
    Quick preprocessing function for simple use cases
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Target column name
        test_size (float): Test set size
        random_state (int): Random state
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    processor = DataPreprocessor(target_column, test_size, random_state)
    
    # Clean and prepare data
    df_clean = processor.clean_data(df)
    df_features = processor.create_features(df_clean)
    
    # Prepare features and target
    X, y, preprocessor = processor.prepare_features(df_features)
    y_encoded = processor.encode_target(y)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y_encoded)
    
    # Fit and transform
    X_train_processed, X_val_processed, X_test_processed = processor.fit_transform(
        X_train, X_val, X_test
    )
    
    return (X_train_processed, X_val_processed, X_test_processed, 
            y_train, y_val, y_test, processor)