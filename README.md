# Universal ML Analytics & Clustering Dashboard

A Streamlit web app for **end‑to‑end ML analytics and clustering** on any tabular dataset (CSV/Excel).
Upload your data once, then explore, train, evaluate, interpret, cluster, and persist your experiment history.

---

## Main Features

- **Universal dataset support**
  - Upload any CSV/Excel file with arbitrary column names.
  - Interactive target selection and optional target value filtering.
  - Handles mixed numeric / categorical / boolean features.

- **Automatic problem detection**
  - Detects **classification vs regression** from the target.
  - Chooses appropriate models and metrics for each case.

- **Supervised modeling**
  - Models:
    - Classification: Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, Decision Tree, Gradient Boosting.
    - Regression: Linear, Ridge, Lasso, Random Forest, SVR, KNN, Decision Tree, Gradient Boosting.
  - Configurable test size, random state, cross‑validation.
  - Comparison table with validation/test metrics and CV scores.
  - Best model detection by relevant metric (Accuracy/F1 or R²).

- **Evaluation & visualization**
  - Classification:
    - Accuracy, Precision, Recall, F1, ROC‑AUC.
    - Confusion matrix (raw + normalized), ROC curves (single and multi‑model).
    - Classification report table.
  - Regression:
    - R², RMSE, MAE, MAPE.
  - Training time vs accuracy scatter for model efficiency analysis.

- **Interpretability (Explainable AI)**
  - SHAP:
    - Global feature importance.
    - Per‑sample waterfall plots.
  - LIME:
    - Per‑sample local explanations with feature impact bars.
  - Side‑by‑side model interpretation comparison.

- **AI Clustering & Insight Dashboard**
  - Feature selection for unsupervised analysis.
  - Dimensionality reduction:
    - PCA, t‑SNE, UMAP (if `umap-learn` installed).
  - Clustering algorithms:
    - K‑Means, DBSCAN, Gaussian Mixture (GMM), OPTICS.
  - Cluster quality scoring:
    - Silhouette score, Davies–Bouldin index, effective cluster count.
  - Visualizations:
    - 2D scatter and 3D scatter in reduced space, colored by cluster.
  - Cluster profiles:
    - Per‑cluster size, proportion, and mean of numeric features.
  - Algorithm comparison:
    - Summary table of metrics across clustering algorithms.

- **History & persistence**
  - Uploads:
    - Dataset configuration (target, exclusions, shapes, distributions).
  - Training:
    - Models trained, comparison metrics, best model and config.
  - Clustering:
    - Each run’s features, reducer, algorithm, params, metrics, comparison table, and cluster summary.
  - All history is stored in `upload_history.json` and **persists across app restarts**.

- **Optional Gemini feedback**
  - Sidebar input for a Gemini API key.
  - For each clustering run:
    - Sends cluster summaries + heuristic text to Gemini.
    - Displays concise feedback about clustering quality and tuning suggestions.

---

## 🚀 Quick Start

### 1. Install dependencies

Recommended: use a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
# Optional for UMAP reducer
pip install umap-learn
# Optional for Gemini feedback
pip install requests
```

### 2. Run the app

From the project root:

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 📘 Main Workflows

### Data Upload

1. Go to **Data Upload**.
2. Upload CSV/Excel.
3. Choose the target column.
4. (Optional) Exclude target values (e.g. derive binary from multiclass).
5. Save the dataset configuration to history.

### Supervised Modeling

1. Go to **Model Training**.
2. Adjust test size / random state / cross‑validation.
3. Select models to train.
4. Start training and inspect:
   - Training summary table.
   - Best model message.
   - Persisted training details in history.
5. Use **Model Evaluation** and **Model Interpretation** for deeper analysis.

### Clustering & Insights

1. Go to **Clustering & Insights**.
2. Select features to cluster on.
3. Choose dimensionality reduction (PCA / t‑SNE / UMAP) and 2D/3D.
4. Choose clustering algorithm and hyperparameters.
5. Run clustering to see:
   - Quality metrics (clusters, silhouette, Davies–Bouldin).
   - 2D/3D scatter plots.
   - Cluster profile table.
   - Algorithm comparison table.
   - Automatic text insights.
6. (Optional) Enter a Gemini API key in the sidebar to get AI feedback on clustering.

### History

- Use **History** to:
  - Review all datasets, training runs, and clustering runs.
  - Reload configurations.
  - Export or clean entries as needed.

---

## 🧩 Tech Stack

- **Frontend / app**: Streamlit
- **ML**: scikit‑learn, NumPy, pandas
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: SHAP, LIME
- **Optional**: UMAP (`umap-learn`), Gemini Generative Language API (`requests`)
