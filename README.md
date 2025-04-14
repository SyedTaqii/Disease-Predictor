
# 🧠 Disease Classification with TF-IDF vs One-Hot Encoding

This project explores the impact of **TF-IDF** and **One-Hot Encoding** on medical diagnosis data. It applies dimensionality reduction (PCA/SVD), builds classifiers (KNN, Logistic Regression), and visualizes disease clusters — all with a focus on real-world interpretability.

---

## 📁 Files

- `DSA03_pipeline.ipynb` — Jupyter notebook with full implementation.
- `streamlit_knn_app.py` — Interactive Streamlit app for classification.
- `disease_features.csv` — Dataset with raw features.
- `encoded_output2.csv` — One-hot encoded features provided for comparison.

---

## 🧩 Project Tasks

### ✅ Task 1: Feature Extraction
- Parsed `Risk Factors`, `Symptoms`, `Signs` from stringified lists.
- Applied **TF-IDF Vectorization** to convert them into numerical format.
- Compared TF-IDF vs One-Hot in terms of shape & sparsity.

### ✅ Task 2: Dimensionality Reduction
- Applied **PCA** and **Truncated SVD** on both matrices.
- Visualized clusters in 2D using `matplotlib`.
- Analyzed separability across disease subtypes.

### ✅ Task 3: Classification
- Used **K-Nearest Neighbors** (k=5) and **Logistic Regression**.
- Evaluated with **Leave-One-Out Cross-Validation** due to 1 sample per class.
- Compared accuracy and F1-score across encodings and models.

### ✅ Task 4: Critical Analysis
- TF-IDF captures term importance better, offering more informative clustering.
- One-Hot is simple but sparse, less effective for small datasets.
- Supervised learning was limited due to dataset structure (no class redundancy).

---

## 🚀 Running the App

To launch the interactive Streamlit app:

```bash
pip install streamlit pandas scikit-learn matplotlib
streamlit run streamlit_knn_app.py
```

---

## 🏷️ Tags

`TF-IDF` `One-Hot Encoding` `KNN` `Logistic Regression` `Dimensionality Reduction` `SVD` `PCA` `Streamlit` `Medical Data`
