"""Rule-based baselines for the Senior Accountant project.

Module 1 — JE anomaly rule: flag an entry when ANY of these fires:
  * amount is round (>= $5,000 and a multiple of $1,000)
  * no supporting document attached AND amount > $5,000
  * tie_out_diff != 0 (unbalanced entry)
  * posted on a weekend
The rule is deliberately transparent: it is the checklist a senior
accountant runs by hand before the model exists.
"""

import numpy as np


def je_rule_flag(df):
    round_flag = (df["amount"] % 1000 == 0) & (df["amount"] >= 5000)
    missing_doc = (~df["doc_attached"]) & (df["amount"] > 5000)
    unbalanced = df["tie_out_diff"] != 0
    weekend = df["date"].dt.weekday >= 5
    return (round_flag | missing_doc | unbalanced | weekend).astype(int)


def zscore_flag(series, threshold=4.0):
    z = (series - series.mean()) / series.std()
    return (z.abs() > threshold).astype(int)


def top_n_queue(scores, n=20):
    """Indices of the n highest scores, highest first."""
    return np.argsort(-np.asarray(scores))[:n]
