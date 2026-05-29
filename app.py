import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()

# ---------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------
st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Dataset Columns")
st.write(df.columns.tolist())

# ---------------------------------------------------
# ATTRITION RATE
# ---------------------------------------------------
st.subheader("Attrition Rate")

attrition_rate = df['Attrition'].value_counts(normalize=True) * 100

st.write(attrition_rate)

# ---------------------------------------------------
# ATTRITION BY DEPARTMENT
# ---------------------------------------------------
st.subheader("Attrition by Department")

department_attrition = (
    df.groupby('Department')['Attrition']
    .value_counts(normalize=True)
    .unstack() * 100
)

st.dataframe(department_attrition)

# ---------------------------------------------------
# ATTRITION BY GENDER
# ---------------------------------------------------
st.subheader("Attrition by Gender")

gender_attrition = (
    df.groupby('Gender')['Attrition']
    .value_counts(normalize=True)
    .unstack() * 100
)

st.dataframe(gender_attrition)

# ---------------------------------------------------
# MONTHLY INCOME BOXPLOT
# ---------------------------------------------------
st.subheader("Attrition by Monthly Income")

fig1, ax1 = plt.subplots(figsize=(8, 6))

sns.boxplot(
    x='Attrition',
    y='MonthlyIncome',
    data=df,
    ax=ax1
)

ax1.set_title("Monthly Income vs Attrition")

st.pyplot(fig1)

# ---------------------------------------------------
# JOB LEVEL COUNT PLOT
# ---------------------------------------------------
st.subheader("Attrition by Job Level")

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.countplot(
    x='JobLevel',
    hue='Attrition',
    data=df,
    ax=ax2
)

ax2.set_title("Job Level vs Attrition")

st.pyplot(fig2)

# ---------------------------------------------------
# TOTAL WORKING YEARS BOXPLOT
# ---------------------------------------------------
st.subheader("Attrition by Total Working Years")

fig3, ax3 = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x='Attrition',
    y='TotalWorkingYears',
    data=df,
    ax=ax3
)

ax3.set_title("Total Working Years vs Attrition")

st.pyplot(fig3)

# ---------------------------------------------------
# YEARS AT COMPANY BOXPLOT
# ---------------------------------------------------
st.subheader("Attrition by Years At Company")

fig4, ax4 = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x='Attrition',
    y='YearsAtCompany',
    data=df,
    ax=ax4
)

ax4.set_title("Years At Company vs Attrition")

st.pyplot(fig4)

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
st.subheader("Data Preprocessing")

df_model = df.copy()

# Encode categorical columns
for col in df_model.columns:

    if (
        df_model[col].dtype == 'object'
        or str(df_model[col].dtype).startswith("string")
    ):

        le = LabelEncoder()

        df_model[col] = le.fit_transform(
            df_model[col].astype(str)
        )

# Convert boolean columns
for col in df_model.select_dtypes(include=['bool']).columns:
    df_model[col] = df_model[col].astype(int)

# Fill missing values
df_model = df_model.fillna(0)

# Convert all columns safely to numeric
for col in df_model.columns:
    df_model[col] = pd.to_numeric(
        df_model[col],
        errors='coerce'
    )

# Fill any remaining NaN values
df_model = df_model.fillna(0)

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------
X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

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
st.sidebar.title("Choose Machine Learning Model")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree"]
)

# ---------------------------------------------------
# LOGISTIC REGRESSION MODEL
# ---------------------------------------------------
if model_name == "Logistic Regression":

    st.subheader("Logistic Regression Results")

    model = LogisticRegression(
        max_iter=5000,
        solver='liblinear'
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

# ---------------------------------------------------
# DECISION TREE MODEL
# ---------------------------------------------------
else:

    st.subheader("Decision Tree Results")

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"Accuracy : {accuracy:.4f}")
st.write(f"Precision : {precision:.4f}")
st.write(f"Recall : {recall:.4f}")
st.write(f"F1 Score : {f1:.4f}")

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
st.subheader("Confusion Matrix")

fig5, ax5 = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    ax=ax5
)

ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")

st.pyplot(fig5)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "Developed using Streamlit, Pandas, Scikit-Learn and Matplotlib"
)
