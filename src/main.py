from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import streamlit as st
from classifiers import dummy_clf, logistic_clf, decision_tree_clf


def show_df(df: pd.DataFrame):
    st.write(df)
    st.write(f"shape: {df.shape}")


def split_df(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    y = df["target"].values
    X = df.drop(columns="target").values
    return X, y


def load_data() -> Tuple[pd.DataFrame, List[str]]:
    iris = load_iris(as_frame=True)
    df = iris["frame"]
    target_names = iris["target_names"]
    return df, target_names


st.write("# Iris Flower Classification")

df, target_names = load_data()

show_df(df)

random_seed = st.number_input(label="Set Random Seed", value=69)
test_size = st.slider(
    label="Test Size", min_value=0.1, max_value=0.4, step=0.05, value=0.2
)

train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)

st.write("## Training Set")
show_df(train_df)

st.write("## Testing Set")
show_df(test_df)

st.write("## Training & Testing the Model (AKA Machine Learning)")

X_train, y_train = split_df(train_df)
X_test, y_test = split_df(test_df)

CLASSIFIERS: Dict[str, BaseEstimator] = {
    "Dummy": dummy_clf,
    "Logistic": logistic_clf,
    "Decision Tree": decision_tree_clf,
}

clf_name = st.selectbox(
    label="Choose the classifier model",
    options=CLASSIFIERS.keys(),
)

clf = CLASSIFIERS[clf_name](X=X_train, y=y_train, random_seed=random_seed)

st.write(f"#### Test Score: {clf.score(X_test, y_test) * 100:.2f}%")

st.text("Chose Model Input")

sepel_length = st.number_input(label="Sepel Length (cm)")
sepel_width = st.number_input(label="Sepel Width (cm)")
petal_length = st.number_input(label="Petal Length (cm)")
petal_width = st.number_input(label="Petal Width (cm)")

(prediction, *_) = clf.predict([[sepel_length, sepel_width, petal_length, petal_width]])
st.write(f"### Prediction: {prediction} ({target_names[prediction]})")
