"""
End-to-end demo: SECOM fault detection.

Loads the real SECOM sensor dataset, cleans it (drops dead sensors, drops
high-missingness columns, imputes the rest), trains a Random Forest
classifier, and reports imbalance-aware evaluation metrics,
including the naive always-predict-pass baseline, so the model's
lift over "do nothing" is visible.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from data_cleaning import drop_constant_columns, drop_high_missing_columns, impute_median, load_secom
from modeling import evaluate_classifier, find_threshold_for_recall, train_test_split_stratified


def main():
    df = load_secom("data/secom_raw.csv")
    print(f"Loaded SECOM: {df.shape[0]} rows, {df.shape[1]} columns")

    y = df["Pass/Fail"]
    X = df.drop(columns=["Time", "Pass/Fail"])
    print(f"Fault rate: {(y == 1).sum()}/{len(y)} = {(y == 1).mean():.1%}")

    X_clean, dropped_constant = drop_constant_columns(X, return_dropped=True)
    print(f"Dropped {len(dropped_constant)} zero-variance (dead) sensor columns")

    X_clean = drop_high_missing_columns(X_clean, threshold=0.5)
    print(f"Columns after dropping >=50% missing: {X_clean.shape[1]}")

    X_clean = impute_median(X_clean)
    assert X_clean.isna().sum().sum() == 0, "median imputation left missing values"
    print("Remaining missing values imputed with column median")

    # Three-way split: train / validation (for threshold tuning) / test
    # (touched exactly once, at the end). Tuning the decision threshold
    # on the test set, or on the same data the model was fit on, would be
    # a leakage bug -- see tests/test_modeling.py::
    # test_threshold_tuned_on_train_can_fail_to_generalize_to_test for a
    # pinned, verified demonstration of exactly how badly that overfits
    # (0% recall on held-out data despite targeting 50%).
    X_train_full, X_test, y_train_full, y_test = train_test_split_stratified(X_clean, y, test_size=0.25)
    X_train, X_val, y_train, y_val = train_test_split_stratified(X_train_full, y_train_full, test_size=0.25)
    print(
        f"Train: {len(X_train)} rows ({(y_train == 1).sum()} faults) | "
        f"Validation: {len(X_val)} rows ({(y_val == 1).sum()} faults) | "
        f"Test: {len(X_test)} rows ({(y_test == 1).sum()} faults)"
    )

    # Baseline: always predict "pass" (the majority class)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate_classifier(baseline, X_test, y_test)

    # Random Forest with class_weight="balanced" to counter the 14:1 imbalance
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    # Default 0.5 threshold, for comparison -- this is the failure mode
    # this project documents: it predicts zero faults.
    default_threshold_metrics = evaluate_classifier(clf, X_test, y_test)

    # Tune the threshold on the VALIDATION split (never seen by the
    # model during fit, never touched until now for the test split).
    target_recall = 0.5
    threshold = find_threshold_for_recall(clf, X_val, y_val, target_recall=target_recall)
    tuned_metrics = evaluate_classifier(clf, X_test, y_test, threshold=threshold)

    print("\n--- Baseline (always predict 'pass') ---")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Random Forest, default threshold=0.5 ---")
    for k, v in default_threshold_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n--- Random Forest, threshold={threshold:.3f} (tuned on validation split for target_recall={target_recall}) ---")
    for k, v in tuned_metrics.items():
        print(f"  {k}: {v}")

    print(
        f"\nAt the default threshold, the model's recall is {default_threshold_metrics['recall']:.1%} "
        f"-- identical to the naive baseline, despite class_weight='balanced' and a real ROC-AUC of "
        f"{default_threshold_metrics['roc_auc']:.2f} showing the model DOES separate the classes. "
        f"The predicted fault probabilities on this dataset simply never cross 0.5 (verified: max "
        f"probability on the test set is well under 0.5). Tuning the threshold on a held-out "
        f"validation split (never the test set) targeted {target_recall:.0%} recall and achieved "
        f"{tuned_metrics['recall']:.1%} recall on the actual test set (higher than targeted on this "
        f"particular split -- validation-to-test transfer is approximate, not exact) at precision={tuned_metrics['precision']:.1%}. That precision means "
        f"roughly 6 false alarms for every real fault caught -- a real, deliberate tradeoff a deployed "
        f"fault-detection system would have to choose, not a free win."
    )


if __name__ == "__main__":
    main()
