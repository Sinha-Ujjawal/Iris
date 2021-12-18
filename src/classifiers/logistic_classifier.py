import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegressionCV


def logistic_clf(X: np.array, y: np.array, random_seed: int) -> LogisticRegressionCV:
    clf = LogisticRegressionCV(random_state=random_seed)
    return clf.fit(X, y)
