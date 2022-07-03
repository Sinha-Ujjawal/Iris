import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def decision_tree_clf(X: np.array, y: np.array, random_seed: int) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(random_state=random_seed)
    return clf.fit(X, y)
