import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class ClusteringPreprocessor:
    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names: Optional[List[str]] = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )
        X = self.preprocessor.fit_transform(df)
        self.feature_names = self._get_feature_names()
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted")
        X = self.preprocessor.transform(df)
        if self.feature_names is None:
            self.feature_names = self._get_feature_names()
        return X

    def _get_feature_names(self) -> List[str]:
        names: List[str] = []
        names.extend(self.numeric_features)
        if self.categorical_features:
            try:
                transformer = self.preprocessor.named_transformers_["cat"]
                if hasattr(transformer, "get_feature_names_out"):
                    cat_names = transformer.get_feature_names_out(
                        self.categorical_features
                    )
                    names.extend(list(cat_names))
                else:
                    categories = transformer.categories_
                    for i, col in enumerate(self.categorical_features):
                        if i < len(categories):
                            for val in categories[i]:
                                names.append(f"{col}_{val}")
            except Exception:
                for col in self.categorical_features:
                    names.append(f"{col}_encoded")
        if not names:
            names = [
                f"feature_{i}"
                for i in range(self.preprocessor.transformers_[0][1].n_features_in_)
            ]
        return names


def reduce_dimensions(
    X: np.ndarray, method: str, n_components: int = 2, random_state: int = 42
) -> np.ndarray:
    m = method.lower()
    if m == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(X)
    if m in ["tsne", "t-sne"]:
        reducer = TSNE(
            n_components=n_components, random_state=random_state, init="random"
        )
        return reducer.fit_transform(X)
    if m == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP is not installed. Install umap-learn to use this reducer."
            )
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(X)
    raise ValueError(f"Unknown reduction method: {method}")


def run_clustering(
    X: np.ndarray, algorithm: str, params: Dict[str, Any], random_state: int = 42
) -> Tuple[np.ndarray, Any]:
    a = algorithm.lower()
    if a == "k-means" or a == "kmeans":
        n_clusters = int(params.get("n_clusters", 3))
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        return labels, model
    if a == "dbscan":
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        return labels, model
    if a == "gmm" or a == "gaussian mixture":
        n_components = int(params.get("n_components", 3))
        model = GaussianMixture(n_components=n_components, random_state=random_state)
        labels = model.fit_predict(X)
        return labels, model
    if a == "optics":
        min_samples = int(params.get("min_samples", 5))
        xi = float(params.get("xi", 0.05))
        model = OPTICS(min_samples=min_samples, xi=xi)
        labels = model.fit_predict(X)
        return labels, model
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def score_clusters(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
        return {
            "n_clusters": len(unique_labels),
            "silhouette": None,
            "davies_bouldin": None,
        }
    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None
    try:
        db = float(davies_bouldin_score(X, labels))
    except Exception:
        db = None
    return {
        "n_clusters": len(unique_labels),
        "silhouette": sil,
        "davies_bouldin": db,
    }


def build_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "Algorithm": name,
                "Clusters": metrics.get("n_clusters"),
                "Silhouette": metrics.get("silhouette"),
                "Davies-Bouldin": metrics.get("davies_bouldin"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Algorithm", "Clusters", "Silhouette", "Davies-Bouldin"]
        )
    df = pd.DataFrame(rows)
    return df


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    data = df.copy()
    data["cluster"] = labels
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if "cluster" in numeric_cols:
        numeric_cols.remove("cluster")
    if not numeric_cols:
        counts = data["cluster"].value_counts().sort_index()
        total = len(data)
        summary = pd.DataFrame(
            {
                "cluster": counts.index,
                "size": counts.values,
                "proportion": counts.values / total,
            }
        )
        return summary
    grouped = data.groupby("cluster")
    size = grouped.size()
    means = grouped[numeric_cols].mean()
    summary = means.copy()
    summary.insert(0, "size", size)
    summary.insert(1, "proportion", size / len(data))
    summary.reset_index(inplace=True)
    return summary
