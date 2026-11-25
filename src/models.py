"""Training + evaluation for the three modules.

Module 1 — JE anomaly detection (classification on injected labels):
  rule baseline vs Isolation Forest (unsupervised) vs Logistic
  Regression vs Random Forest. Hold-out: Nov-Dec 2025.
Module 2a — close cycle-time forecast (regression):
  seasonal-naive (same month last year proxy = median) vs Linear
  Regression vs Random Forest. Hold-out: Nov-Dec 2025.
Module 2b — recon-break prediction (classification):
  majority baseline vs Logistic Regression vs Random Forest.
Module 3 — tax-planning estimator (regression on the scenario engine):
  prior-year-liability baseline vs Linear Regression vs Random
  Forest. Grouped split by client (70/30).

Every metric the report cites is written to models/*.csv here.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def _cls_metrics(y_true, y_pred, y_score=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             zero_division=0),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def _reg_metrics(y_true, y_pred, mape_floor=0.0):
    """MAPE is computed only on rows with |y_true| >= mape_floor: a
    zero (or near-zero) true value makes MAPE meaningless (division by
    ~0). Module 3 passes floor=1000 because 23 of 180 synthetic
    liabilities are exactly $0 (credits exceed the invented tax);
    those rows still count in MAE and R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= mape_floor
    return {
        "mape": float(mean_absolute_percentage_error(y_true[mask],
                                                     y_pred[mask])),
        "mape_n": int(mask.sum()),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def module1(train, test, features, target="is_anomaly"):
    Xtr, ytr = train[features], train[target]
    Xte, yte = test[features], test[target]
    rows = []

    # Rule baseline (computed upstream, passed via test frame column).
    rule_pred = test["rule_flag"].values
    m = _cls_metrics(yte, rule_pred)
    m["model"] = "Rule baseline"
    rows.append(m)

    # Isolation Forest (unsupervised; fit on train features only).
    iso = IsolationForest(n_estimators=200, contamination=0.03,
                          random_state=42)
    iso.fit(Xtr)
    iso_pred = (iso.predict(Xte) == -1).astype(int)
    iso_score = -iso.decision_function(Xte)
    m = _cls_metrics(yte, iso_pred, iso_score)
    m["model"] = "Isolation Forest"
    rows.append(m)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr_s, ytr)
    lr_pred = lr.predict(Xte_s)
    m = _cls_metrics(yte, lr_pred, lr.predict_proba(Xte_s)[:, 1])
    m["model"] = "Logistic Regression"
    rows.append(m)

    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    m = _cls_metrics(yte, rf_pred, rf.predict_proba(Xte)[:, 1])
    m["model"] = "Random Forest"
    rows.append(m)

    metrics = pd.DataFrame(rows)
    scored = test.copy()
    scored["rf_score"] = rf.predict_proba(Xte)[:, 1]
    scored["rf_pred"] = rf_pred
    importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return metrics, scored, importance, rf


def module2a(train, test, features, target="days_to_close"):
    Xtr, ytr = train[features], train[target]
    Xte, yte = test[features], test[target]
    rows = []

    naive_pred = np.full(len(yte), ytr.median())
    m = _reg_metrics(yte, naive_pred)
    m["model"] = "Median baseline"
    rows.append(m)

    lin = LinearRegression()
    lin.fit(Xtr, ytr)
    m = _reg_metrics(yte, lin.predict(Xte))
    m["model"] = "Linear Regression"
    rows.append(m)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    m = _reg_metrics(yte, rf.predict(Xte))
    m["model"] = "Random Forest"
    rows.append(m)

    metrics = pd.DataFrame(rows)
    importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return metrics, importance, rf


def module2b(train, test, features, target="recon_break"):
    Xtr, ytr = train[features], train[target]
    Xte, yte = test[features], test[target]
    rows = []

    maj = int(ytr.mode().iloc[0])
    m = _cls_metrics(yte, np.full(len(yte), maj))
    m["model"] = "Majority baseline"
    rows.append(m)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr_s, ytr)
    m = _cls_metrics(yte, lr.predict(Xte_s),
                     lr.predict_proba(Xte_s)[:, 1])
    m["model"] = "Logistic Regression"
    rows.append(m)

    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)
    m = _cls_metrics(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])
    m["model"] = "Random Forest"
    rows.append(m)

    return pd.DataFrame(rows), rf


def module3(train, test, features, target="projected_liability"):
    Xtr, ytr = train[features], train[target]
    Xte, yte = test[features], test[target]
    rows = []

    base_pred = test["prior_year_liability"].values
    m = _reg_metrics(yte, base_pred, mape_floor=1000.0)
    m["model"] = "Prior-year baseline"
    rows.append(m)

    lin = LinearRegression()
    lin.fit(Xtr, ytr)
    m = _reg_metrics(yte, lin.predict(Xte), mape_floor=1000.0)
    m["model"] = "Linear Regression"
    rows.append(m)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    m = _reg_metrics(yte, rf.predict(Xte), mape_floor=1000.0)
    m["model"] = "Random Forest"
    rows.append(m)

    metrics = pd.DataFrame(rows)
    importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return metrics, importance, rf
