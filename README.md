# Industrial Sensor Fault Detection & Gas-Sensor Regression

Two small real-data projects applying pandas, NumPy and scikit-learn to messy industrial and environmental sensor data. Both pipelines are test-driven, and this README documents the data issues and modeling decisions that came up along the way, as well as the final numbers.

## What it does

1. **SECOM fault detection** (`demo_secom.py`): predicts Pass/Fail on a real semiconductor-fab production line from 590 anonymized sensor readings per run.
2. **Air Quality regression** (`demo_airquality.py`): predicts a reference-grade CO concentration from a cheaper co-located gas-sensor array, using real hourly readings from a multisensor device deployed in an Italian city.

Both use real, publicly available data, and both required genuine data-cleaning decisions beyond `df.dropna()`.

## Data

**SECOM** (McCann & Johnston, 2008; UCI ML Repository, CC BY 4.0) is real semiconductor fab sensor data. Detecting a fault from hundreds of correlated sensor channels on a production line is the same methodological problem across process types, so it is a good real, messy, multi-sensor fault-detection dataset. The project exercises real-data cleaning, imbalance-aware evaluation and careful threshold selection.

**Air Quality** (De Vito et al., 2008; UCI ML Repository, CC BY 4.0, **non-commercial research use**) is real chemical gas-sensor data: metal-oxide sensors (tin oxide, titania, tungsten oxide, indium oxide) reporting alongside a reference-grade analyzer in an urban pollution-monitoring deployment. Its license is non-commercial/research use, which suits a learning project.

Both datasets originate from the UCI Machine Learning Repository and are fetched from GitHub mirrors by `download_data.py`. I verified them against the row counts and column layouts documented for the original UCI files.

## Findings along the way

**1. Column count verified against the real file.** Initial research suggested SECOM has "591 sensor features." The first test against the downloaded file failed with `592 == 593` on `assert df.shape[1] == 593`, and direct inspection showed 590 numbered sensor columns (0-589). The test now matches the verified file: see `tests/test_data_cleaning.py::TestLoadSecom::test_loads_real_file_with_expected_shape`.

**2. 116 "dead" sensor columns.** Beyond scattered missing values (4.52% of cells, across 538/591 columns), 116 of SECOM's sensor columns are constant or entirely NaN across all 1567 production runs. I drop these explicitly (`drop_constant_columns`) rather than imputing them, since imputing a constant column only reproduces that constant.

**3. The default classification threshold predicts no faults.** SECOM's target is severely imbalanced (104 faults / 1567 runs = 6.6%). A `RandomForestClassifier` with `class_weight="balanced"` produced:

```
recall: 0.0
precision: 0.0
confusion_matrix: [[366, 0], [26, 0]]
```

This gives 93.4% accuracy while missing every fault in the test set. Inspecting the predicted probabilities showed that on this dataset the model's fault probabilities never exceed ~0.38 on held-out data, so scikit-learn's default 0.5 threshold classifies every row as "pass." The model's ROC-AUC (0.76) shows it does separate the classes, so the threshold was the issue, not the model.

**4. Tune the threshold on a validation split, not the training data.** Tuning the decision threshold on the training set's own predicted probabilities overfits: the tuned threshold (0.72) achieved **0% recall** on the held-out test set, because it was fit to the model's near-memorized view of its training data. This is pinned as a regression test (`tests/test_modeling.py::TestFindThresholdForRecall::test_threshold_tuned_on_train_can_fail_to_generalize_to_test`). The pipeline uses a three-way split (train / validation / test), with the threshold tuned only on the validation split and the test split used once, at the end. Targeting 50% recall on validation achieved 76.9% recall on the untouched test set, at 14.2% precision: roughly 6 false alarms per fault caught, a deliberate tradeoff.

**5. Interpreting the Air Quality feature importances.** In the Air Quality regression, the Random Forest's most important feature for predicting CO concentration is `PT08.S2(NMHC)`, the sensor channel nominally calibrated for hydrocarbons, rather than `PT08.S1(CO)`, the CO-targeted sensor (importance 0.894 vs. 0.025). I checked this against the dataset's UCI documentation and the source paper (De Vito et al., *Sensors and Actuators B*, 2008): metal-oxide sensors are documented as cross-sensitive (they respond to a general class of reducing gases via surface oxidation, not just their nominal target), and CO and NMHC are both primary products of incomplete combustion, so they co-vary strongly in urban traffic pollution regardless of sensor selectivity. The explanation is cross-sensitivity plus a shared emission source, which is what `demo_airquality.py` prints.

## Results

**SECOM fault detection**: Random Forest, `class_weight="balanced"`, threshold tuned on a held-out validation split for 50% target recall: recall 76.9%, precision 14.2%, ROC-AUC 0.76, vs. a naive always-"pass" baseline at 0% recall. Test set: 392 rows, 26 real faults.

**Air Quality regression**: Random Forest, predicting CO(GT) from the 5-channel sensor array plus temperature/humidity: MAE 0.261 mg/m³, R² 0.920 (Linear Regression baseline: MAE 0.314, R² 0.900). Test set: 1836 rows.

## Tests

28 tests cover the cleaning utilities (`tests/test_data_cleaning.py`, `tests/test_airquality.py`) and the modeling utilities (`tests/test_modeling.py`):

```bash
pytest tests/
```

## Project structure

```
download_data.py        fetches both datasets
demo_secom.py           SECOM fault-detection demo
demo_airquality.py      Air Quality regression demo
src/data_cleaning.py    SECOM cleaning utilities
src/airquality.py       Air Quality cleaning utilities
src/modeling.py         splits, imbalance-aware evaluation, threshold selection
tests/                  test_data_cleaning.py, test_airquality.py, test_modeling.py
conftest.py, pytest.ini, requirements-dev.txt
```

## Running it

```bash
pip install -r requirements-dev.txt
python3 download_data.py    # fetches both real datasets (~6.8 MB total)
pytest tests/               # 28 tests
python3 demo_secom.py
python3 demo_airquality.py
```

All reported results come from running the demos.

## Notes

- SECOM is semiconductor-fab data and Air Quality is ambient urban air monitoring; both are real sensor datasets used here for the data-cleaning and modeling methods they demonstrate.
- The Air Quality data is licensed for non-commercial research use.

## Possible extensions

- Add cross-validated threshold selection for SECOM.
- Try gradient-boosted models and compare feature-selection strategies on both datasets.
- Add time-aware splits for the hourly Air Quality data.
