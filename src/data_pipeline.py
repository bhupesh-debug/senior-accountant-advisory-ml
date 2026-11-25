"""Validation + cleaning for the Senior Accountant advisory project.

Loads the synthetic CSVs, checks schema and ranges, and returns a
findings dict. Nothing here touches real data — all inputs are
synthetic by construction.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")


def load():
    clients = pd.read_csv(os.path.join(SYN, "clients.csv"))
    je = pd.read_csv(os.path.join(SYN, "journal_entries.csv"),
                     parse_dates=["date"])
    close = pd.read_csv(os.path.join(SYN, "close_calendar.csv"))
    tax = pd.read_csv(os.path.join(SYN, "tax_scenarios.csv"))
    return clients, je, close, tax


def validate(clients, je, close, tax):
    findings = {}
    findings["clients_rows"] = len(clients)
    findings["clients_missing"] = int(clients.isna().sum().sum())
    findings["je_rows"] = len(je)
    findings["je_missing"] = int(je.isna().sum().sum())
    findings["je_negative_amount"] = int((je["amount"] <= 0).sum())
    findings["je_anomaly_rate"] = round(
        float(je["is_anomaly"].mean()), 4)
    findings["je_date_range"] = (
        str(je["date"].min().date()), str(je["date"].max().date()))
    findings["close_rows"] = len(close)
    findings["close_missing"] = int(close.isna().sum().sum())
    findings["close_break_rate"] = round(
        float(close["recon_break"].mean()), 4)
    findings["close_late_rate"] = round(
        float(close["late_close"].mean()), 4)
    findings["tax_rows"] = len(tax)
    findings["tax_missing"] = int(tax.isna().sum().sum())
    findings["tax_negative_liability"] = int(
        (tax["projected_liability"] < 0).sum())
    findings["tax_scenarios"] = sorted(tax["scenario"].unique().tolist())
    return findings


def daily_totals(je):
    """Daily entry counts + amounts (for the pipeline health view)."""
    d = je.groupby(je["date"].dt.date).agg(
        entries=("entry_id", "count"),
        total_amount=("amount", "sum"),
        anomalies=("is_anomaly", "sum"),
    ).reset_index().rename(columns={"date": "day"})
    return d
