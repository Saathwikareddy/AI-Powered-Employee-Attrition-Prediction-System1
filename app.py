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

# -----------------------------------
# PAGE TITLE
# -----------------------------------
st.set_page_config(page_title="IBM HR Attrition Analysis", layout="wide")

st.title("AI Powered Employee Attrition Prediction System")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()

# -----------------------------------
# DATA PREVIEW
# -----------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# DATASET INFORMATION
# -----------------------------------
st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Column Names")
st.write(df.columns.tolist())

# -----------------------------------
# ATTRITION RATE
# -----------------------------------
st.subheader("Attrition Rate")

attrition_rate = df['Attrition'].value_counts(normalize=True) * 100

st.write(attrition_rate)

# -----------------------------------
# ATTRITION BY DEPARTMENT
# -----------------------------------
st.subheader("Attrition by Department")

department_attrition = (
    df.groupby('Department')['Attrition']
    .value_counts(normalize=True)
    .unstack() * 100
)

st.dataframe(department_attrition)

# -----------------------------------
# ATTRITION BY GENDER
# -----------------------------------
st.subheader("Attrition by Gender")

gender_attrition = (
    df.groupby('Gender')['Attrition']
    .value_counts(normalize=True)
    .unstack() * 100
)

st.dataframe(gender_attrition)

# -----------------------------------
# MONTHLY INCOME BOXPLOT
# -----------------------------------
st.subheader("Attrition by Monthly Income")

fig1, ax1 = plt.subplots(figsize=(8, 6))

sns.boxplot(
    x='Attrition',
    y='MonthlyIncome',
    data=df,
    palette='pastel',
    ax=ax1
)

ax1.set_title("Attrition vs Monthly Income")

st.pyplot(fig1)

# -----------------------------------
# JOB LEVEL COUNT PLOT
# -----------------------------------
st.subheader("Attrition by Job Level")

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.countplot(
    x='JobLevel',
    hue='Attrition',
    data=df,
    palette='coolwarm',
    ax=ax2
)

ax2.set_title("Attrition by Job Level")

st.pyplot(fig2)

# -----------------------------------
# TOTAL WORKING YEARS
# -----------------------------------
st.subheader("Attrition by Total Working Years")

fig3, ax3 = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x='Attrition',
    y='TotalWorkingYears',
    data=df,
    palette='viridis',
    ax=ax3
)

ax3.set_title("Attrition by Total Working Years")

st.pyplot(fig3)

# -----------------------------------
# YEARS AT COMPANY
# -----------------------------------
st.subheader("Attrition by Years At Company")

fig4, ax4 = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x='Attrition',
    y='YearsAtCompany',
    data=df,
    palette='plasma',
    ax=ax4
)

ax4.set_title("Attrition by Years At Company")

st.pyplot(fig4)

# -----------------------------------
# PREPROCESSING
# -----------------------------------
st.subheader("Data Preprocessing")

df_model = df.copy()

label_encoders = {}

for col in df_model.columns:
    if df_model[col].dtype == 'object':
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

# Features and target
X = df_model.drop("Attrition", axis=1)
y = df_model["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.success("Preprocessing Completed Successfully")

# -----------------------------------
# SIDEBAR MODEL SELECTION
# -----------------------------------
st.sidebar.title("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a Model",
    ["Logistic Regression", "Decision Tree"]
)

# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------
if model_name == "Logistic Regression":

    st.subheader("Logistic Regression Model")

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    st.subheader("Confusion Matrix")
    st.write(cm)

# -----------------------------------
# DECISION TREE
# -----------------------------------
else:

    st.subheader("Decision Tree Model")

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    st.subheader("Confusion Matrix")
    st.write(cm)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.markdown("Developed using Streamlit and Scikit-Learn")
