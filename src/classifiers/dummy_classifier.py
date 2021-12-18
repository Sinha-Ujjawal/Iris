import streamlit as st
import numpy as np
from sklearn.dummy import DummyClassifier


def dummy_clf(X: np.array, y: np.array, random_seed: int) -> DummyClassifier:
    clf = DummyClassifier(random_state=random_seed)
    return clf.fit(X, y)
