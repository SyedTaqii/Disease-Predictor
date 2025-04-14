import streamlit as st
import pandas as pd
import ast
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut

# Load data
df = pd.read_csv("disease_features.csv")
df_onehot = pd.read_csv("encoded_output2.csv")

# Streamlit title
st.title("Disease Feature Analysis and Classification")

# --- Task 1: TF-IDF Feature Extraction ---
st.header("Task 1: TF-IDF Feature Extraction")

def parse_and_join(column):
    return df[column].apply(lambda x: " ".join(ast.literal_eval(x)) if pd.notna(x) and x.strip() != "[]" else "")

df['Risk_Factors_str'] = parse_and_join('Risk Factors')
df['Symptoms_str'] = parse_and_join('Symptoms')
df['Signs_str'] = parse_and_join('Signs')

tfidf_risk = TfidfVectorizer()
tfidf_symptoms = TfidfVectorizer()
tfidf_signs = TfidfVectorizer()

X_risk = tfidf_risk.fit_transform(df['Risk_Factors_str'])
X_symptoms = tfidf_symptoms.fit_transform(df['Symptoms_str'])
X_signs = tfidf_signs.fit_transform(df['Signs_str'])

X_tfidf_combined = sp.hstack([X_risk, X_symptoms, X_signs])
st.write("TF-IDF Matrix Shape:", X_tfidf_combined.shape)

X_onehot = df_onehot.drop(columns=['Disease'])
X_onehot_sparse = sp.csr_matrix(X_onehot.values)
st.write("One-Hot Matrix Shape:", X_onehot_sparse.shape)

# --- Task 2: Dimensionality Reduction ---
st.header("Task 2: Dimensionality Reduction")
df['Subtype_Label'] = df['Subtypes'].apply(lambda x: list(ast.literal_eval(x).keys())[0] if pd.notna(x) else "Unknown")
labels = df['Subtype_Label'].values
unique_labels = list(set(labels))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
colors = cm.tab10(np.linspace(0, 1, len(unique_labels)))

def plot_2d(X, title):
    fig, ax = plt.subplots()
    for label in unique_labels:
        idx = np.array(labels) == label
        ax.scatter(X[idx, 0], X[idx, 1], label=label, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

plot_2d(PCA(n_components=2).fit_transform(X_tfidf_combined.toarray()), "PCA - TF-IDF")
plot_2d(PCA(n_components=2).fit_transform(X_onehot_sparse.toarray()), "PCA - One-Hot")
plot_2d(TruncatedSVD(n_components=2).fit_transform(X_tfidf_combined), "SVD - TF-IDF")
plot_2d(TruncatedSVD(n_components=2).fit_transform(X_onehot_sparse), "SVD - One-Hot")

# --- Task 3: Classification ---
st.header("Task 3: Classification with KNN and Logistic Regression")
encoding_type = st.selectbox("Choose Encoding", ["TF-IDF", "One-Hot"])
k_val = st.slider("Choose K (Neighbors)", 1, 10, 5)
metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"])

if encoding_type == "TF-IDF":
    X = X_tfidf_combined
    X_reduced = TruncatedSVD(n_components=15).fit_transform(X_tfidf_combined)
else:
    X = X_onehot_sparse
    X_reduced = TruncatedSVD(n_components=15).fit_transform(X_onehot_sparse)

y = LabelEncoder().fit_transform(df['Subtype_Label'])

# KNN Classification
y_preds, y_trues = [], []
knn = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X):
    knn.fit(X[train_idx], y[train_idx])
    y_pred = knn.predict(X[test_idx])
    y_preds.append(y_pred[0])
    y_trues.append(y[test_idx][0])

acc_knn = accuracy_score(y_trues, y_preds)
f1_knn = f1_score(y_trues, y_preds, average='macro', zero_division=0)

st.subheader("KNN Results")
st.write(f"Accuracy: {acc_knn:.2f}")
st.write(f"F1-Score: {f1_knn:.2f}")

# Logistic Regression on SVD-reduced data
y_preds_lr, y_trues_lr = [], []
logreg = LogisticRegression(solver='liblinear')
for train_idx, test_idx in loo.split(X_reduced):
    logreg.fit(X_reduced[train_idx], y[train_idx])
    y_pred = logreg.predict(X_reduced[test_idx])
    y_preds_lr.append(y_pred[0])
    y_trues_lr.append(y[test_idx][0])

acc_lr = accuracy_score(y_trues_lr, y_preds_lr)
f1_lr = f1_score(y_trues_lr, y_preds_lr, average='macro', zero_division=0)

st.subheader("Logistic Regression Results (SVD-Reduced)")
st.write(f"Accuracy: {acc_lr:.2f}")
st.write(f"F1-Score: {f1_lr:.2f}")

# Prediction Table
if st.checkbox("Show KNN Predictions"):
    subtypes = LabelEncoder().fit(df['Subtype_Label']).classes_
    df_knn = pd.DataFrame({
        "True": [subtypes[i] for i in y_trues],
        "Predicted": [subtypes[i] for i in y_preds]
    })
    st.dataframe(df_knn)

if st.checkbox("Show Logistic Regression Predictions"):
    subtypes = LabelEncoder().fit(df['Subtype_Label']).classes_
    df_lr = pd.DataFrame({
        "True": [subtypes[i] for i in y_trues_lr],
        "Predicted": [subtypes[i] for i in y_preds_lr]
    })
    st.dataframe(df_lr)
