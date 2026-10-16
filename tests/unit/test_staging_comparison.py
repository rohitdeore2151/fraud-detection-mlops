import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ml.compare_staging import evaluate_model, load_model


FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
NUMERIC_COLUMNS = ["step", "amount", "oldbalanceOrg", "oldbalanceDest"]


def build_preprocessor(numeric_columns=NUMERIC_COLUMNS):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                ["type"],
            ),
        ],
        sparse_threshold=1.0,
    )


def as_dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def test_candidate_and_staging_artifact_pairs_are_feature_compatible(
    tmp_path,
):
    X_train = pd.DataFrame(
        {
            "step": range(1, 11),
            "type": [
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER",
            ]
            * 2,
            "amount": [100.0 * value for value in range(1, 11)],
            "oldbalanceOrg": [200.0 * value for value in range(1, 11)],
            "oldbalanceDest": [300.0 * value for value in range(1, 11)],
        }
    )
    y_train = np.array([0, 1] * 5)
    X_val = X_train[FEATURE_COLUMNS].copy()
    y_val = pd.Series(y_train)

    candidate_preprocessor = build_preprocessor()
    candidate_train = candidate_preprocessor.fit_transform(
        X_train[NUMERIC_COLUMNS + ["type"]]
    )
    staging_preprocessor = build_preprocessor()
    staging_preprocessor.fit(X_train[FEATURE_COLUMNS])

    model = XGBClassifier(
        n_estimators=3,
        max_depth=2,
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(candidate_train, y_train)

    candidate_model_path = tmp_path / "candidate.json"
    staging_model_path = tmp_path / "staging.json"
    candidate_preprocessor_path = tmp_path / "candidate.joblib"
    staging_preprocessor_path = tmp_path / "staging.joblib"
    model.save_model(str(candidate_model_path))
    shutil.copyfile(candidate_model_path, staging_model_path)
    joblib.dump(candidate_preprocessor, candidate_preprocessor_path)
    joblib.dump(staging_preprocessor, staging_preprocessor_path)

    candidate_loaded = load_model(candidate_model_path)
    staging_loaded = load_model(staging_model_path)
    candidate_prep_loaded = joblib.load(candidate_preprocessor_path)
    staging_prep_loaded = joblib.load(staging_preprocessor_path)
    candidate_features = candidate_prep_loaded.transform(X_val)
    staging_features = staging_prep_loaded.transform(X_val)
    expected_features = candidate_loaded.get_booster().num_features()

    assert candidate_features.shape[1] == expected_features
    assert staging_features.shape[1] == staging_loaded.get_booster().num_features()
    np.testing.assert_array_equal(
        as_dense(candidate_features),
        as_dense(staging_features),
    )
    np.testing.assert_array_equal(
        candidate_loaded.predict_proba(candidate_features),
        staging_loaded.predict_proba(staging_features),
    )

    candidate_metrics = evaluate_model(
        "Candidate",
        candidate_model_path,
        candidate_preprocessor_path,
        X_val,
        y_val,
    )
    staging_metrics = evaluate_model(
        "Staging",
        staging_model_path,
        staging_preprocessor_path,
        X_val,
        y_val,
    )
    assert {
        metric: value
        for metric, value in candidate_metrics.items()
        if metric != "model"
    } == {
        metric: value
        for metric, value in staging_metrics.items()
        if metric != "model"
    }


def test_evaluate_model_rejects_incompatible_preprocessor(tmp_path):
    X_train = pd.DataFrame(
        {
            "step": range(1, 11),
            "type": ["TRANSFER", "PAYMENT"] * 5,
            "amount": [100.0 * value for value in range(1, 11)],
            "oldbalanceOrg": [200.0 * value for value in range(1, 11)],
            "oldbalanceDest": [300.0 * value for value in range(1, 11)],
        }
    )
    y_train = np.array([0, 1] * 5)
    compatible_preprocessor = build_preprocessor()
    transformed = compatible_preprocessor.fit_transform(X_train)
    model = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(transformed, y_train)

    model_path = tmp_path / "model.json"
    incompatible_preprocessor_path = tmp_path / "incompatible.joblib"
    model.save_model(str(model_path))
    incompatible_preprocessor = build_preprocessor(NUMERIC_COLUMNS[:-1])
    joblib.dump(
        incompatible_preprocessor.fit(X_train),
        incompatible_preprocessor_path,
    )

    with pytest.raises(ValueError, match="preprocessor produced .* model expects"):
        evaluate_model(
            "Incompatible",
            model_path,
            incompatible_preprocessor_path,
            X_train[FEATURE_COLUMNS],
            pd.Series(y_train),
        )
