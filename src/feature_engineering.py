"""Feature engineering + splits for the Senior Accountant project.

Module 1 (JE anomaly): time-based hold-out, last 2 months of 2025.
Module 2 (close analytics): time-based hold-out, last 2 months.
Module 3 (tax estimator): grouped random split by client (70/30) so no
client appears in both train and test.
"""

import numpy as np
import pandas as pd

JE_FEATURES = [
    "log_amount", "is_weekend", "doc_missing", "is_round",
    "amount_vs_client_median", "entries_same_day", "month_end",
    "is_adjusting", "account_rarity",
]

CLOSE_FEATURES = [
    "complexity", "staff_assigned", "entries_count", "docs_missing",
    "busy_season", "prior_month_late", "month_num",
]

# Module 2a (cycle-time forecast) may use the recon break as a driver —
# it is discovered during the close. Module 2b PREDICTS the break, so
# the break itself must never be a feature (leakage).
CLOSE_FEATURES_A = CLOSE_FEATURES + ["recon_break"]

TAX_FEATURES = [
    "taxable_income", "deductions", "credits", "prior_year_liability",
    "is_bear", "is_bull", "state_rate",
]


def je_features(je):
    df = je.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    df["doc_missing"] = (~df["doc_attached"]).astype(int)
    df["is_round"] = ((df["amount"] % 1000 == 0)
                      & (df["amount"] >= 5000)).astype(int)
    med = df.groupby("client_id")["amount"].transform("median")
    df["amount_vs_client_median"] = df["amount"] / med
    same_day = df.groupby(["client_id", "date"])["entry_id"].transform("count")
    df["entries_same_day"] = same_day
    df["month_end"] = (df["date"].dt.day >= 25).astype(int)
    df["is_adjusting"] = (df["entry_type"] == "Adjusting").astype(int)
    acct_counts = df["account"].value_counts(normalize=True)
    df["account_rarity"] = 1.0 - df["account"].map(acct_counts)
    return df


def je_time_split(df, months=2):
    cut = pd.Timestamp("2025-11-01")
    train = df[df["date"] < cut].reset_index(drop=True)
    test = df[df["date"] >= cut].reset_index(drop=True)
    return train, test, cut


def close_features(close):
    df = close.copy()
    df["month_num"] = pd.PeriodIndex(df["month"], freq="M").month
    return df


def close_time_split(df, months=2):
    cut = "2025-11"
    train = df[df["month"] < cut].reset_index(drop=True)
    test = df[df["month"] >= cut].reset_index(drop=True)
    return train, test, cut


def tax_features(tax, state_rates):
    df = tax.copy()
    df["is_bear"] = (df["scenario"] == "bear").astype(int)
    df["is_bull"] = (df["scenario"] == "bull").astype(int)
    df["state_rate"] = df["state"].map(state_rates)
    return df


def tax_group_split(df, seed=42):
    rng = np.random.default_rng(seed)
    clients = list(df["client_id"].unique())
    rng.shuffle(clients)
    n_test = int(round(0.30 * len(clients)))
    test_clients = set(clients[:n_test])
    test = df[df["client_id"].isin(test_clients)].reset_index(drop=True)
    train = df[~df["client_id"].isin(test_clients)].reset_index(drop=True)
    return train, test
