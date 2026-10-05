"""
Tests for the modeling module.

The target is severely imbalanced (104 fails / 1463 passes, 6.6% positive
rate), so plain accuracy is close to meaningless here — a model that
always predicts "pass" scores ~93.4% accuracy while catching zero real
faults. These tests exist specifically to force the evaluation code to
report imbalance-aware metrics (recall, precision, ROC-AUC) rather than
letting accuracy stand in as the headline number.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from modeling import evaluate_classifier, select_top_variance_features, train_test_split_stratified


@pytest.fixture
def imbalanced_data():
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=5, weights=[0.9, 0.1],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    y = pd.Series(y)
    return X, y


class TestTrainTestSplitStratified:
    def test_preserves_class_ratio_in_both_splits(self, imbalanced_data):
        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        overall_ratio = y.mean()
        # Stratified split should keep the minority-class ratio close in
        # both splits, not just in the overall dataset.
        assert abs(train_ratio - overall_ratio) < 0.05
        assert abs(test_ratio - overall_ratio) < 0.05

    def test_split_sizes_sum_to_total(self, imbalanced_data):
        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)


class TestEvaluateClassifier:
    def test_reports_recall_precision_and_roc_auc_not_just_accuracy(self, imbalanced_data):
        from sklearn.ensemble import RandomForestClassifier
        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        clf = RandomForestClassifier(random_state=42, n_estimators=50)
        clf.fit(X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)
        for key in ("accuracy", "recall", "precision", "roc_auc", "confusion_matrix"):
            assert key in metrics

    def test_dummy_always_negative_has_zero_recall(self, imbalanced_data):
        # A model that always predicts the majority class ("no fault")
        # must be caught by recall=0, even though accuracy looks fine.
        # This is the exact failure mode the imbalance-aware evaluation
        # in this project is meant to surface.
        from sklearn.dummy import DummyClassifier
        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(X_train, y_train)
        metrics = evaluate_classifier(clf, X_test, y_test)
        assert metrics["recall"] == 0.0
        assert metrics["accuracy"] > 0.7  # deceptively high despite recall=0


class TestSelectTopVarianceFeatures:
    def test_selects_requested_number_of_features(self, imbalanced_data):
        X, _ = imbalanced_data
        selected = select_top_variance_features(X, n=5)
        assert len(selected.columns) == 5

    def test_selects_highest_variance_columns(self):
        df = pd.DataFrame({
            "low_var": [1.0, 1.01, 1.02, 1.0],
            "high_var": [1.0, 50.0, -30.0, 20.0],
            "mid_var": [1.0, 3.0, 2.0, 4.0],
        })
        selected = select_top_variance_features(df, n=1)
        assert selected.columns.tolist() == ["high_var"]


class TestFindThresholdForRecall:
    def test_finds_threshold_achieving_target_recall_on_same_data(self, imbalanced_data):
        # By construction, a threshold chosen by find_threshold_for_recall
        # on a given (X, y) must achieve >= target_recall ON THAT SAME
        # DATA -- that is what roc_curve's tpr values directly measure.
        from sklearn.ensemble import RandomForestClassifier
        from modeling import find_threshold_for_recall
        from sklearn.metrics import recall_score

        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        clf = RandomForestClassifier(random_state=42, n_estimators=50, class_weight="balanced")
        clf.fit(X_train, y_train)

        threshold = find_threshold_for_recall(clf, X_train, y_train, target_recall=0.5)
        proba_train = clf.predict_proba(X_train)[:, 1]
        pred_train = (proba_train >= threshold).astype(int)
        assert recall_score(y_train, pred_train) >= 0.5

    def test_threshold_tuned_on_train_can_fail_to_generalize_to_test(self, imbalanced_data):
        # This is an important finding from building this project,
        # kept here as a regression test as well as in the README:
        # a threshold tuned against a RandomForest's OWN training-set
        # probabilities overfits badly on this fixture (0.72, chosen from
        # a near-memorized training set) and achieves 0% recall on the
        # held-out test set despite targeting 50%. That is exactly why
        # demo_secom.py tunes its threshold on a validation split carved
        # out of the training data, never on the test set or on the same
        # data the classifier was fit on. This test pins that failure
        # mode down so it cannot silently regress into "looks fine."
        from sklearn.ensemble import RandomForestClassifier
        from modeling import find_threshold_for_recall
        from sklearn.metrics import recall_score

        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        clf = RandomForestClassifier(random_state=42, n_estimators=50, class_weight="balanced")
        clf.fit(X_train, y_train)

        threshold = find_threshold_for_recall(clf, X_train, y_train, target_recall=0.5)
        proba_test = clf.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= threshold).astype(int)
        # Documented, verified overfitting behavior, which is why the demo script uses a
        # held-out validation split for threshold tuning instead.
        assert recall_score(y_test, pred_test) == 0.0

    def test_threshold_decreases_as_target_recall_increases(self, imbalanced_data):
        # "High recall always needs threshold < 0.5" does not hold on
        # this fixture -- a RandomForest evaluated on its OWN training
        # data separates this synthetic dataset well enough that 80%
        # recall is reachable above 0.5 (verified: threshold ~0.62).
        # The property that holds universally, and is what this function
        # is for, is monotonicity: asking for more recall can never
        # require a HIGHER threshold than asking for less.
        from sklearn.ensemble import RandomForestClassifier
        from modeling import find_threshold_for_recall

        X, y = imbalanced_data
        X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.25)
        clf = RandomForestClassifier(random_state=42, n_estimators=50, class_weight="balanced")
        clf.fit(X_train, y_train)

        low_target_threshold = find_threshold_for_recall(clf, X_train, y_train, target_recall=0.3)
        high_target_threshold = find_threshold_for_recall(clf, X_train, y_train, target_recall=0.8)
        assert high_target_threshold <= low_target_threshold
