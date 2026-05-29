import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Powered Employee Attrition Prediction System",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("AI Powered Employee Attrition Prediction System")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

df = load_data()

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# DATASET INFO
# ---------------------------------------------------
st.subheader("Dataset Shape")
st.write(df.shape)

# ---------------------------------------------------
# ATTRITION RATE
# ---------------------------------------------------
st.subheader("Attrition Rate")

attrition_rate = (
    df['Attrition']
    .value_counts(normalize=True) * 100
)

st.write(attrition_rate)

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------
st.subheader("Attrition by Monthly Income")

fig1, ax1 = plt.subplots(figsize=(8, 6))

sns.boxplot(
    x='Attrition',
    y='MonthlyIncome',
    data=df,
    ax=ax1
)

st.pyplot(fig1)

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
st.subheader("Data Preprocessing")

df_model = df.copy()

# Convert target column explicitly
df_model["Attrition"] = df_model["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Encode categorical columns
for col in df_model.columns:

    if df_model[col].dtype == 'object':

        le = LabelEncoder()

        df_model[col] = le.fit_transform(
            df_model[col].astype(str)
        )

# Convert all columns to numeric
df_model = df_model.apply(
    pd.to_numeric,
    errors='coerce'
)

# Handle missing values
df_model = df_model.fillna(0)

# Replace infinite values
df_model = df_model.replace(
    [np.inf, -np.inf],
    0
)

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------
X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# Convert datatype
X = X.astype(float)
y = y.astype(int)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.success("Preprocessing Completed Successfully")

# ---------------------------------------------------
# SIDEBAR MODEL SELECTION
# ---------------------------------------------------
st.sidebar.title("Choose Model")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree"]
)

# ---------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------
if model_name == "Logistic Regression":

    st.subheader("Logistic Regression Results")

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )

else:

    st.subheader("Decision Tree Results")

    model = DecisionTreeClassifier(
        random_state=42
    )

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# ---------------------------------------------------
# DISPLAY METRICS
# ---------------------------------------------------
st.subheader("Model Performance")

st.write(f"Accuracy : {accuracy:.4f}")
st.write(f"Precision : {precision:.4f}")
st.write(f"Recall : {recall:.4f}")
st.write(f"F1 Score : {f1:.4f}")

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

st.subheader("Confusion Matrix")

fig2, ax2 = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax2
)

ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

st.pyplot(fig2)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "Developed using Streamlit and Scikit-Learn"
)
