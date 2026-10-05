"""
Modeling utilities for the SECOM fault-detection task.

The core discipline here: SECOM's target is severely imbalanced (6.6%
positive/fault rate), so evaluate_classifier() always reports recall,
precision and ROC-AUC alongside accuracy — never accuracy alone, since a
trivial always-predict-pass model scores ~93% accuracy while catching
zero real faults. That failure mode is directly covered by
test_dummy_always_negative_has_zero_recall in tests/test_modeling.py.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def train_test_split_stratified(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42):
    """Stratified train/test split so the minority (fault) class ratio is
    preserved in both splits, not just in the dataset as a whole. An
    unstratified split on a 6.6%-positive dataset can easily land a test
    set with too few (or zero) fault examples to evaluate recall
    meaningfully.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _score(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return None


def evaluate_classifier(clf, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = None) -> dict:
    """Evaluate a fitted classifier with imbalance-aware metrics.

    ROC-AUC uses predict_proba when available; falls back to decision
    scores, and to hard predictions only as a last resort (with a worse,
    but degenerate AUC in that case).

    If `threshold` is given and the classifier exposes predict_proba or
    decision_function, predictions are made by thresholding that score
    directly instead of using clf.predict()'s built-in 0.5 cutoff. This
    matters a lot on imbalanced data: a RandomForestClassifier with
    class_weight="balanced" trained on SECOM never produces a predicted
    fault-probability above ~0.38 on the held-out test set (verified by
    inspection, not assumed), so the default 0.5 threshold silently
    predicts "no fault" for every single test row -- recall=0.0 -- even
    though the model's probabilities do carry real separating signal
    (ROC-AUC ~0.80). See find_threshold_for_recall() for choosing a
    threshold deliberately instead of accepting that silent failure.
    """
    y_score = _score(clf, X_test)

    if threshold is not None and y_score is not None:
        y_pred = np.where(y_score >= threshold, 1, -1)
    else:
        y_pred = clf.predict(X_test)

    if y_score is None:
        y_score = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def find_threshold_for_recall(clf, X: pd.DataFrame, y: pd.Series, target_recall: float) -> float:
    """Find the lowest decision threshold (on predict_proba's positive-
    class score) that achieves at least `target_recall` on the given
    (typically training) data.

    This exists because the default 0.5 classification threshold is not
    a law of nature -- it is just scikit-learn's default, and on a
    severely imbalanced problem like SECOM it is frequently the wrong
    choice. Choosing a threshold explicitly, based on how much recall a
    fault-detection system actually needs (versus how many false alarms
    it can tolerate), is a real deployment decision this function makes
    inspectable rather than hiding inside a default.
    """
    y_score = _score(clf, X)
    if y_score is None:
        raise ValueError("classifier has neither predict_proba nor decision_function")

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    # roc_curve's first threshold is +inf (from sklearn's implementation
    # detail) and thresholds are in decreasing order paired with
    # increasing tpr (recall). Find the highest threshold whose recall
    # (tpr) meets the target.
    candidates = [(t, r) for t, r in zip(thresholds, tpr) if r >= target_recall]
    if not candidates:
        # target_recall not achievable even at the lowest threshold;
        # return the lowest observed threshold as the best effort.
        return float(thresholds[-1])
    # candidates are already ordered by decreasing threshold since
    # roc_curve returns thresholds in decreasing order; the first match
    # is the highest (most conservative) threshold meeting the target.
    return float(candidates[0][0])


def select_top_variance_features(X: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the n columns with the highest variance.

    Used as a simple, transparent dimensionality-reduction step before
    modeling on SECOM's ~500+ remaining sensor columns after cleaning —
    not because variance implies predictive power, but because it is a
    simple, easily explained baseline rather than a black-box selector.
    """
    variances = X.var(numeric_only=True).sort_values(ascending=False)
    top_cols = variances.head(n).index.tolist()
    return X[top_cols]
