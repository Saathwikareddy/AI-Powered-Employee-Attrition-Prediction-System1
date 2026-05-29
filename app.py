import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# STREAMLIT TITLE
# -----------------------------
st.title("IBM HR Attrition Analysis")

# -----------------------------
# LOAD DATASET
# -----------------------------
@st.cache_data

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -----------------------------
# SHOW DATA
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Information")
st.write(df.shape)

# -----------------------------
# ATTRITION RATE
# -----------------------------
st.subheader("Attrition Rate")

attrition_rate = df['Attrition'].value_counts(normalize=True) * 100
st.write(attrition_rate)

# -----------------------------
# ATTRITION BY DEPARTMENT
# -----------------------------
st.subheader("Attrition by Department")

department_attrition = (
    df.groupby('Department')['Attrition']
    .value_counts(normalize=True)
    .unstack() * 100
)

st.dataframe(department_attrition)

# -----------------------------
# ATTRITION BY GENDER
# -----------------------------
st.subheader("Attrition by Gender")

gender_attrition = (
    df.groupby('Gender')['Attrition']
    .value_counts(normalize=True)
    .unstack() * 100
)

st.dataframe(gender_attrition)

# -----------------------------
# BOXPLOT - MONTHLY INCOME
# -----------------------------
st.subheader("Attrition by Monthly Income")

fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(
    x='Attrition',
    y='MonthlyIncome',
    data=df,
    palette='pastel',
    ax=ax
)

st.pyplot(fig)

# -----------------------------
# JOB LEVEL COUNT PLOT
# -----------------------------
st.subheader("Attrition by Job Level")

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.countplot(
    x='JobLevel',
    hue='Attrition',
    data=df,
    palette='coolwarm',
    ax=ax2
)

st.pyplot(fig2)

# -----------------------------
# PREPROCESSING
# -----------------------------
df_model = df.copy()

le = LabelEncoder()

for col in df_model.columns:
    if df_model[col].dtype == 'object':
        df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# MODEL SELECTION
# -----------------------------
st.sidebar.title("Choose Model")

model_name = st.sidebar.selectbox(
    "Select a Model",
    ["Logistic Regression", "Decision Tree"]
)

# -----------------------------
# LOGISTIC REGRESSION
# -----------------------------
if model_name == "Logistic Regression":

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Logistic Regression Accuracy")
    st.write(f"Accuracy: {accuracy:.4f}")

# -----------------------------
# DECISION TREE
# -----------------------------
else:

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Decision Tree Accuracy")
    st.write(f"Accuracy: {accuracy:.4f}")
