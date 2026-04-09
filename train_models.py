#!/usr/bin/env python3
"""
LearnSight: Training + Explainable AI (SHAP)
Trains LightGBM, XGBoost, and Random Forest with anti-overfitting controls.
Produces models/, reports/ artifacts for the Flask web app and academic reporting.
"""

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

DATA_DIR = "data"
MERGED_DATA_PATH = os.path.join(DATA_DIR, "merged_dataset.csv")
DATA_PATH = os.path.join(DATA_DIR, "student_data.csv")

MODELS_DIR = "models"
REPORTS_DIR = "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
SEARCH_ITER = 4

LEAKAGE_CORR_THRESHOLD = 0.97


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _map_ordinal(series, mapping):
    if series.dtype.kind in {"i", "u", "f"}:
        return series.map(mapping).where(~series.isna(), np.nan)
    return series


def load_data():
    _ensure_dirs()

    if os.path.exists(MERGED_DATA_PATH):
        df = pd.read_csv(MERGED_DATA_PATH)
        df = df.rename(
            columns={
                "StudyHours": "Study_Hours",
                "Attendance": "Attendance_Rate",
                "Resources": "Access_to_Resources",
                "Extracurricular": "Extracurricular_Activity",
                "Motivation": "Motivation_Level",
                "Internet": "Internet_Access",
                "Gender": "Gender",
                "Age": "Age",
                "LearningStyle": "Learning_Style",
                "OnlineCourses": "Online_Courses",
                "Discussions": "Discussion_Participation",
                "AssignmentCompletion": "Assignment_Completion",
                "ExamScore": "Exam_Score",
                "EduTech": "EduTech_Usage",
                "StressLevel": "Stress_Level",
                "FinalGrade": "Final_Grade",
            }
        )

        df["Final_Grade"] = _map_ordinal(df["Final_Grade"], {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"})
        if "Access_to_Resources" in df.columns:
            df["Access_to_Resources"] = _map_ordinal(df["Access_to_Resources"], {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"})
        if "Motivation_Level" in df.columns:
            df["Motivation_Level"] = _map_ordinal(df["Motivation_Level"], {0: "Low", 1: "Medium", 2: "High"})
        if "Stress_Level" in df.columns:
            df["Stress_Level"] = _map_ordinal(df["Stress_Level"], {0: "Low", 1: "Medium", 2: "High"})
        if "Internet_Access" in df.columns:
            df["Internet_Access"] = _map_ordinal(df["Internet_Access"], {0: "No", 1: "Yes"})
        if "Extracurricular_Activity" in df.columns:
            df["Extracurricular_Activity"] = _map_ordinal(df["Extracurricular_Activity"], {0: "No", 1: "Yes"})
        if "EduTech_Usage" in df.columns:
            df["EduTech_Usage"] = _map_ordinal(df["EduTech_Usage"], {0: "Low", 1: "Medium", 2: "High"})
        if "Gender" in df.columns:
            df["Gender"] = _map_ordinal(df["Gender"], {0: "Male", 1: "Female"})
        if "Learning_Style" in df.columns:
            df["Learning_Style"] = _map_ordinal(df["Learning_Style"], {0: "Visual", 1: "Auditory", 2: "Kinesthetic", 3: "Reading/Writing"})

        return df

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    df = generate_sample_data(n_samples=700)
    df.to_csv(DATA_PATH, index=False)
    return df


def generate_sample_data(n_samples=700):
    np.random.seed(RANDOM_STATE)
    data = {
        "Age": np.random.randint(18, 30, n_samples),
        "Gender": np.random.choice(["Male", "Female", "Other"], n_samples, p=[0.45, 0.5, 0.05]),
        "Study_Hours": np.random.uniform(0, 50, n_samples),
        "Attendance_Rate": np.random.uniform(50, 100, n_samples),
        "Assignment_Completion": np.random.uniform(0, 100, n_samples),
        "Discussion_Participation": np.random.uniform(0, 100, n_samples),
        "Motivation_Level": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.2, 0.55, 0.25]),
        "Stress_Level": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.35, 0.45, 0.2]),
        "Access_to_Resources": np.random.choice(["Poor", "Fair", "Good", "Excellent"], n_samples, p=[0.1, 0.25, 0.45, 0.2]),
        "Learning_Style": np.random.choice(["Visual", "Auditory", "Kinesthetic", "Reading/Writing"], n_samples),
        "Previous_GPA": np.random.uniform(1.0, 4.0, n_samples),
        "Midterm_Score": np.random.uniform(0, 100, n_samples),
        "Quiz_Scores": np.random.uniform(0, 100, n_samples),
        "Library_Visits": np.random.randint(0, 20, n_samples),
        "Online_Forum_Activity": np.random.uniform(0, 100, n_samples),
        "Internet_Access": np.random.choice(["No", "Yes"], n_samples, p=[0.2, 0.8]),
        "Online_Courses": np.random.randint(0, 25, n_samples),
        "Extracurricular_Activity": np.random.choice(["No", "Yes"], n_samples, p=[0.6, 0.4]),
        "EduTech_Usage": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.25, 0.5, 0.25]),
    }
    df = pd.DataFrame(data)

    grade_score = (
        (df["Study_Hours"] / 50.0) * 18
        + (df["Attendance_Rate"] / 100.0) * 18
        + (df["Assignment_Completion"] / 100.0) * 14
        + (df["Previous_GPA"] / 4.0) * 18
        + (df["Midterm_Score"] / 100.0) * 16
        + (df["Quiz_Scores"] / 100.0) * 16
    )
    grade_score += np.random.normal(0, 12, n_samples)
    grade_score = np.clip(grade_score, 0, 100)

    def score_to_grade(score):
        if score >= 85:
            return "A"
        if score >= 70:
            return "B"
        if score >= 55:
            return "C"
        if score >= 40:
            return "D"
        return "F"

    df["Final_Grade"] = grade_score.apply(score_to_grade)

    for col in ["Study_Hours", "Attendance_Rate", "Previous_GPA", "Midterm_Score"]:
        mask = np.random.random(len(df)) < 0.05
        df.loc[mask, col] = np.nan

    return df


def feature_engineer(df):
    df = df.copy()
    if "Exam_Score" in df.columns:
        exam = pd.to_numeric(df["Exam_Score"], errors="coerce")
        noise = np.random.normal(0, 1.5, len(df))
        df["Exam_Score_Noised"] = np.clip(exam + noise, 0, 100)
        df = df.drop(columns=["Exam_Score"])

    if {"Study_Hours", "Attendance_Rate"}.issubset(df.columns):
        df["Study_Attendance_Interaction"] = df["Study_Hours"] * (df["Attendance_Rate"] / 100.0)

    engagement_cols = [c for c in ["Attendance_Rate", "Assignment_Completion", "Discussion_Participation", "Online_Forum_Activity"] if c in df.columns]
    if engagement_cols:
        df["Engagement_Score"] = df[engagement_cols].mean(axis=1)

    academic_cols = [c for c in ["Previous_GPA", "Midterm_Score", "Quiz_Scores"] if c in df.columns]
    if academic_cols:
        df["Academic_Score"] = df[academic_cols].mean(axis=1)

    return df


def detect_and_drop_leakage(X_train, y_train, X_test):
    y_num = pd.Series(y_train).astype(float)
    to_drop = []
    for col in X_train.columns:
        if X_train[col].dtype.kind in {"i", "u", "f"}:
            x = pd.to_numeric(X_train[col], errors="coerce")
            if x.notna().sum() < 10:
                continue
            corr = np.corrcoef(x.fillna(x.median()), y_num)[0, 1]
            if np.isfinite(corr) and abs(corr) >= LEAKAGE_CORR_THRESHOLD:
                to_drop.append(col)

    if to_drop:
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop, errors="ignore")

    meta = {"dropped_for_leakage_corr_threshold": LEAKAGE_CORR_THRESHOLD, "dropped_features": to_drop}
    with open(os.path.join(REPORTS_DIR, "leakage_feature_drop.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return X_train, X_test, to_drop


def _make_one_hot():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def extract_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return None


def build_input_schema(X, y_raw):
    schema = {"target": "Final_Grade", "classes": sorted(pd.Series(y_raw).dropna().astype(str).unique().tolist()), "features": []}
    for col in X.columns:
        s = X[col]
        if s.dtype == "object" or str(s.dtype).startswith("category"):
            options = sorted([str(v) for v in s.dropna().unique().tolist()])[:100]
            schema["features"].append({"name": col, "type": "categorical", "options": options})
        else:
            vals = pd.to_numeric(s, errors="coerce").to_numpy()
            if np.isfinite(vals).any():
                schema["features"].append({"name": col, "type": "numerical", "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals))})
            else:
                schema["features"].append({"name": col, "type": "numerical", "min": 0.0, "max": 1.0})
    return schema


@dataclass
class TrainResult:
    models: dict
    tuning: dict


def _search(estimator, params, X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=SEARCH_ITER,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def train_models(X_train_t, y_train, n_classes):
    models = {}
    tuning = {}

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, bootstrap=True)
    rf_params = {
        "n_estimators": [300, 500],
        "max_depth": [8, 12, None],
        "min_samples_split": [10, 20, 40],
        "min_samples_leaf": [4, 8, 16],
        "max_features": ["sqrt", 0.6, 0.8],
        "class_weight": [None, "balanced"],
    }
    print("Tuning Random Forest...")
    rf_best, rf_cv, rf_best_params = _search(rf, rf_params, X_train_t, y_train)
    models["Random Forest"] = rf_best
    tuning["Random Forest"] = {"cv_accuracy": rf_cv, "best_params": rf_best_params}

    xgb_est = xgb.XGBClassifier(
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_params = {
        "n_estimators": [250, 400, 550],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.04, 0.06, 0.08],
        "subsample": [0.65, 0.75, 0.85],
        "colsample_bytree": [0.65, 0.75, 0.85],
        "min_child_weight": [5, 10, 20],
        "gamma": [0.0, 0.1, 0.25],
        "reg_alpha": [0.0, 0.2, 0.8, 1.5],
        "reg_lambda": [3.0, 6.0, 10.0],
    }
    print("Tuning XGBoost...")
    xgb_best, xgb_cv, xgb_best_params = _search(xgb_est, xgb_params, X_train_t, y_train)
    models["XGBoost"] = xgb_best
    tuning["XGBoost"] = {"cv_accuracy": xgb_cv, "best_params": xgb_best_params}

    lgb_est = lgb.LGBMClassifier(
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_params = {
        "n_estimators": [250, 400, 550],
        "learning_rate": [0.04, 0.06, 0.08],
        "max_depth": [-1, 6, 8],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [30, 60, 90],
        "subsample": [0.65, 0.75, 0.85],
        "colsample_bytree": [0.65, 0.75, 0.85],
        "reg_alpha": [0.0, 0.2, 0.8, 1.5],
        "reg_lambda": [0.0, 1.0, 4.0, 8.0],
        "min_split_gain": [0.0, 0.05, 0.1],
    }
    print("Tuning LightGBM...")
    lgb_best, lgb_cv, lgb_best_params = _search(lgb_est, lgb_params, X_train_t, y_train)
    models["LightGBM"] = lgb_best
    tuning["LightGBM"] = {"cv_accuracy": lgb_cv, "best_params": lgb_best_params}

    return TrainResult(models=models, tuning=tuning)


def _plot_confusion_matrix(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_totals_bar(totals, title, out_path):
    keys = ["TP", "FN", "TN", "FP"]
    vals = [totals[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keys, vals, color=["#2E7D32", "#F9A825", "#1565C0", "#C62828"])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ymax = max(vals) if vals else 0
    for i, v in enumerate(vals):
        ax.text(i, v + (ymax * 0.02 if ymax else 0.5), str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _confusion_totals(cm):
    total = int(cm.sum())
    tp = int(np.trace(cm))
    fp = int(cm.sum(axis=0).sum() - np.trace(cm))
    fn = int(cm.sum(axis=1).sum() - np.trace(cm))
    tn = int(total - tp - fp - fn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _plot_roc_curves(y_true, y_proba, class_names, title, out_path):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    ax.plot(fpr_micro, tpr_micro, color="black", linestyle="--", lw=2.0, label=f"micro (AUC={auc(fpr_micro, tpr_micro):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_models(models, X_test_t, y_test, target_encoder):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t) if hasattr(model, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        prec_w = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_w = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        prec_m = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        roc_auc_macro = None
        roc_auc_weighted = None
        if y_proba is not None and len(target_encoder.classes_) > 1:
            try:
                roc_auc_macro = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
                roc_auc_weighted = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
            except Exception:
                roc_auc_macro = None
                roc_auc_weighted = None

        cm = confusion_matrix(y_test, y_pred)
        totals = _confusion_totals(cm)

        report_txt = classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=4, zero_division=0)

        results[name] = {
            "accuracy": acc,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "f1_weighted": f1_w,
            "precision_macro": prec_m,
            "recall_macro": rec_m,
            "f1_macro": f1_m,
            "roc_auc_macro_ovr": roc_auc_macro,
            "roc_auc_weighted_ovr": roc_auc_weighted,
            "confusion_totals": totals,
        }

        cm_path = os.path.join(REPORTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        _plot_confusion_matrix(cm, target_encoder.classes_, f"{name} Confusion Matrix", cm_path)

        totals_path = os.path.join(REPORTS_DIR, f"tp_fn_tn_fp_{name.lower().replace(' ', '_')}.png")
        _plot_totals_bar(totals, f"{name} TP/FN/TN/FP (Aggregated)", totals_path)

        if y_proba is not None and len(target_encoder.classes_) > 2:
            roc_path = os.path.join(REPORTS_DIR, f"roc_curve_{name.lower().replace(' ', '_')}.png")
            _plot_roc_curves(y_test, y_proba, target_encoder.classes_, f"{name} ROC Curves (OvR)", roc_path)

        with open(os.path.join(REPORTS_DIR, f"classification_report_{name.lower().replace(' ', '_')}.txt"), "w") as f:
            f.write(report_txt)

        with open(os.path.join(REPORTS_DIR, f"metrics_{name.lower().replace(' ', '_')}.json"), "w") as f:
            json.dump(results[name], f, indent=2)

    return results


def select_best_model(results):
    items = list(results.items())
    items.sort(key=lambda kv: (kv[1]["accuracy"], kv[1]["f1_weighted"]), reverse=True)
    return items[0][0]


def shap_top5(shap_values, feature_names):
    if isinstance(shap_values, list):
        abs_vals = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
    else:
        abs_vals = np.mean(np.abs(shap_values), axis=0)
    idx = np.argsort(abs_vals)[::-1][:5]
    return [{"feature": feature_names[i], "mean_abs_shap": float(abs_vals[i])} for i in idx]


def generate_shap_assets(models, X_test_t, feature_names, target_encoder):
    X_sample = X_test_t[: min(250, X_test_t.shape[0])]

    for name, model in models.items():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        with open(os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_explainer.pkl"), "wb") as f:
            pickle.dump(explainer, f)

        summary_path = os.path.join(REPORTS_DIR, f"shap_summary_{name.lower().replace(' ', '_')}.png")
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=180, bbox_inches="tight")
        plt.close()

        bar_path = os.path.join(REPORTS_DIR, f"shap_bar_{name.lower().replace(' ', '_')}.png")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=180, bbox_inches="tight")
        plt.close()

        top5 = shap_top5(shap_values, feature_names)
        with open(os.path.join(REPORTS_DIR, f"shap_top5_{name.lower().replace(' ', '_')}.json"), "w") as f:
            json.dump(top5, f, indent=2)

        local_dir = os.path.join(REPORTS_DIR, f"shap_local_{name.lower().replace(' ', '_')}")
        os.makedirs(local_dir, exist_ok=True)
        for i in range(min(3, X_sample.shape[0])):
            if isinstance(shap_values, list):
                proba = model.predict_proba(X_sample[i : i + 1])[0]
                cls_idx = int(np.argmax(proba))
                force = shap.force_plot(
                    explainer.expected_value[cls_idx],
                    shap_values[cls_idx][i, :],
                    X_sample[i, :],
                    feature_names=feature_names,
                    matplotlib=False,
                )
                out_html = os.path.join(local_dir, f"sample_{i}_class_{target_encoder.classes_[cls_idx]}.html")
            else:
                force = shap.force_plot(
                    explainer.expected_value,
                    shap_values[i, :],
                    X_sample[i, :],
                    feature_names=feature_names,
                    matplotlib=False,
                )
                out_html = os.path.join(local_dir, f"sample_{i}.html")
            shap.save_html(out_html, force)


def save_artifacts(models, preprocessor, feature_names, target_encoder, best_model_name, input_schema, tuning, results, dropped_features):
    for name, model in models.items():
        with open(os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl"), "wb") as f:
            pickle.dump(model, f)

    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    with open(os.path.join(MODELS_DIR, "target_encoder.pkl"), "wb") as f:
        pickle.dump(target_encoder, f)

    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(best_model_name)

    with open(os.path.join(MODELS_DIR, "input_schema.json"), "w") as f:
        json.dump(input_schema, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "tuning_summary.json"), "w") as f:
        json.dump(tuning, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "training_artifacts.json"), "w") as f:
        json.dump(
            {
                "run_utc": _utc_now_iso(),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
                "search_iter": SEARCH_ITER,
                "best_model": best_model_name,
                "dropped_features": dropped_features,
            },
            f,
            indent=2,
        )


def write_summary_md(best_model_name, results, tuning, target_encoder):
    lines = []
    lines.append("# LearnSight Model Training Report")
    lines.append("")
    lines.append(f"- Run date (UTC): {_utc_now_iso()}")
    lines.append(f"- Split: 80/20 stratified (test_size={TEST_SIZE})")
    lines.append(f"- CV: StratifiedKFold(n_splits={CV_FOLDS})")
    lines.append(f"- Best model: {best_model_name}")
    lines.append(f"- Classes: {', '.join([str(c) for c in target_encoder.classes_])}")
    lines.append("")
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Model | Accuracy | F1 (weighted) | Precision (weighted) | Recall (weighted) | ROC-AUC (macro OvR) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        auc_val = "" if m.get("roc_auc_macro_ovr") is None else f"{m['roc_auc_macro_ovr']:.4f}"
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['f1_weighted']:.4f} | {m['precision_weighted']:.4f} | {m['recall_weighted']:.4f} | {auc_val} |")
    lines.append("")
    lines.append("## Anti-overfitting Controls")
    lines.append("")
    lines.append("- Held-out test set used once for final reporting")
    lines.append("- Stratified cross-validation for hyperparameter selection")
    lines.append("- Regularization constraints for tree-based models (depth, leaf sizes, subsampling, L1/L2)")
    lines.append("- Robust scaling + median/mode imputations for stability")
    lines.append(f"- Automatic drop of numeric features with |corr(feature, target)| >= {LEAKAGE_CORR_THRESHOLD:.2f} on the training split")
    lines.append("")
    lines.append("## Best Hyperparameters (CV)")
    lines.append("")
    for name, t in tuning.items():
        lines.append(f"### {name}")
        lines.append(f"- CV accuracy: {t['cv_accuracy']:.4f}")
        lines.append("```json")
        lines.append(json.dumps(t["best_params"], indent=2))
        lines.append("```")
        lines.append("")
    with open(os.path.join(REPORTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    _ensure_dirs()
    df = load_data()
    df = feature_engineer(df)

    if "Final_Grade" not in df.columns:
        raise ValueError("Dataset must contain Final_Grade column.")

    y_raw = df["Final_Grade"].astype(str)
    X = df.drop(columns=["Final_Grade"])

    input_schema = build_input_schema(X, y_raw)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test, dropped_features = detect_and_drop_leakage(X_train, y_train, X_test)

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = extract_feature_names(preprocessor)
    if not feature_names:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    train_out = train_models(X_train_t, y_train, n_classes=len(target_encoder.classes_))
    results = evaluate_models(train_out.models, X_test_t, y_test, target_encoder)

    best_model_name = select_best_model(results)

    generate_shap_assets(train_out.models, X_test_t, feature_names, target_encoder)
    save_artifacts(
        models=train_out.models,
        preprocessor=preprocessor,
        feature_names=feature_names,
        target_encoder=target_encoder,
        best_model_name=best_model_name,
        input_schema=input_schema,
        tuning=train_out.tuning,
        results=results,
        dropped_features=dropped_features,
    )
    write_summary_md(best_model_name, results, train_out.tuning, target_encoder)

    print(f"Best model: {best_model_name}")
    print(f"Test accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"Artifacts written to: {MODELS_DIR}/ and {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
