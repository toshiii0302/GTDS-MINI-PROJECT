import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# ============================================================
# NSL-KDD FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

CATEGORICAL_FEATURES = [
    "protocol_type",
    "service",
    "flag"
]

NUMERICAL_FEATURES = [
    feature
    for feature in FEATURE_NAMES
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================
# DEMO DATA
# ============================================================

def create_demo_data(n: int = 1500) -> pd.DataFrame:
    """
    Generates synthetic network traffic when the NSL-KDD
    dataset is unavailable.

    This is only a fallback/demo dataset.
    """

    rng = np.random.default_rng(42)

    data = pd.DataFrame()

    data["duration"] = rng.exponential(
        scale=40,
        size=n
    ).astype(int).clip(0, 300)

    data["protocol_type"] = rng.choice(
        ["tcp", "udp", "icmp"],
        size=n,
        p=[0.70, 0.25, 0.05]
    )

    data["service"] = rng.choice(
        ["http", "ftp", "ssh", "smtp", "private"],
        size=n,
        p=[0.35, 0.15, 0.15, 0.10, 0.25]
    )

    data["flag"] = rng.choice(
        ["SF", "S0", "REJ", "RSTR"],
        size=n,
        p=[0.55, 0.20, 0.15, 0.10]
    )

    data["src_bytes"] = rng.exponential(
        scale=1000,
        size=n
    ).astype(int).clip(0, 10000)

    data["dst_bytes"] = rng.exponential(
        scale=1200,
        size=n
    ).astype(int).clip(0, 10000)

    # Remaining numerical features
    for column in NUMERICAL_FEATURES:
        if column not in data.columns:
            data[column] = rng.integers(
                0,
                100,
                size=n
            )

    attack_types = [
        "normal",
        "DoS",
        "Probe",
        "R2L",
        "U2R"
    ]

    probabilities = [
        0.45,
        0.25,
        0.15,
        0.10,
        0.05
    ]

    data["label"] = rng.choice(
        attack_types,
        size=n,
        p=probabilities
    )

    return data


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset():

    base_directory = Path(__file__).resolve().parent

    train_file = base_directory / "KDDTrain+.txt"

    if not train_file.exists():

        print("NSL-KDD dataset not found.")
        print("Using automatically generated demo data.")

        return create_demo_data(), False

    print(f"Loading dataset: {train_file}")

    # --------------------------------------------------------
    # Read TAB-separated NSL-KDD dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        train_file,
        header=None,
        sep="\t"
    )

    print(f"Dataset shape detected: {df.shape}")

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if df.shape[1] < 42:

        raise ValueError(
            f"""
            Dataset format problem.

            Detected columns: {df.shape[1]}
            Expected at least 42 columns.

            Make sure KDDTrain+.txt is the actual
            NSL-KDD dataset and is tab-separated.
            """
        )

    # --------------------------------------------------------
    # NSL-KDD structure:
    #
    # 41 network features
    # 1 attack label
    # 1 difficulty level
    #
    # We don't need difficulty level.
    # --------------------------------------------------------

    df = df.iloc[:, :42].copy()

    df.columns = FEATURE_NAMES + ["label"]

    return df, True




# ============================================================
# ATTACK CATEGORY MAPPING
# ============================================================

def simplify_label(label: str) -> str:

    label = str(label).lower().strip()

    # Remove trailing period if present
    label = label.rstrip(".")

    if label == "normal":
        return "Normal"

    dos = [
        "back",
        "land",
        "neptune",
        "pod",
        "smurf",
        "teardrop",
        "apache2",
        "udpstorm",
        "processtable",
        "mailbomb"
    ]

    probe = [
        "ipsweep",
        "nmap",
        "portsweep",
        "satan",
        "mscan",
        "saint"
    ]

    r2l = [
        "ftp_write",
        "guess_passwd",
        "imap",
        "multihop",
        "phf",
        "spy",
        "warezclient",
        "warezmaster",
        "xlock",
        "xsnoop",
        "snmpguess",
        "snmpgetattack",
        "httptunnel",
        "sendmail",
        "named"
    ]

    u2r = [
        "buffer_overflow",
        "loadmodule",
        "perl",
        "rootkit",
        "ps",
        "sqlattack",
        "xterm"
    ]

    if label in dos:
        return "DoS"

    if label in probe:
        return "Probe"

    if label in r2l:
        return "R2L"

    if label in u2r:
        return "U2R"

    return "Other Attack"


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model() -> Dict[str, Any]:

    df, using_real_dataset = load_dataset()

    # Categorize attacks
    df["attack_type"] = df["label"].apply(
        simplify_label
    )

    # Binary IDS target
    df["target"] = (
        df["attack_type"] != "Normal"
    ).astype(int)

    X = df[FEATURE_NAMES]
    y = df["target"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            ),
            (
                "numeric",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing
            ),
            (
                "classifier",
                model
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Classification report
    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Attack"
        ],
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    # Attack distribution
    attack_distribution = (
        df["attack_type"]
        .value_counts()
        .to_dict()
    )

    # Feature importance
    classifier = pipeline.named_steps[
        "classifier"
    ]

    transformed_feature_names = (
        pipeline
        .named_steps["preprocessing"]
        .get_feature_names_out()
    )

    feature_importance = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Importance": classifier.feature_importances_
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )

    return {
        "model": pipeline,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "attack_distribution": attack_distribution,
        "feature_importance": feature_importance,
        "data": df,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "using_real_dataset": using_real_dataset,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
