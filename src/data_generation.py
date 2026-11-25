"""Synthetic data generation for the Senior Accountant advisory project.

ALL DATA IS SYNTHETIC. No real client, no real tax return, no real
ledger is used anywhere. The generator mimics the structure of the JD
(client tax returns, AR/AP/GL accounting, month-end close, tax
planning scenarios) so the models can be built and tested honestly.
Magnitudes, anomaly rates, break rates and tax rates are design
choices, not facts about any firm or client.

IMPORTANT — tax: the scenario engine in this file uses INVENTED,
simplified rates (a flat federal schedule and a flat state rate per
state). It is an analytics/education prototype, NEVER a tax
preparation or tax advice tool. Real tax law is far more complex;
nothing here may be used to prepare a return or advise a client.

Deterministic: seed 42 everywhere.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "synthetic")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(42)

ACCOUNTS = [
    "Cash", "Accounts Receivable", "Accounts Payable", "Inventory",
    "Prepaid Expenses", "Fixed Assets", "Accumulated Depreciation",
    "Notes Payable", "Sales Revenue", "Service Revenue",
    "Cost of Goods Sold", "Payroll Expense", "Rent Expense",
    "Utilities Expense", "Marketing Expense", "Insurance Expense",
    "Professional Fees", "Interest Expense", "Tax Expense",
    "Owner Draw", "Retained Earnings", "Unearned Revenue",
    "Accrued Liabilities", "Depreciation Expense", "Bad Debt Expense",
    "Office Supplies", "Travel Expense", "Software Subscriptions",
    "Equipment", "Long-Term Debt",
]

INDUSTRIES = [
    "Retail", "Manufacturing", "Professional Services", "Real Estate",
    "Healthcare", "Hospitality", "Construction", "E-commerce",
]

ENTITY_TYPES = ["Individual", "Corporation", "Partnership", "S-Corp", "LLC"]

STATES = ["NY", "NJ", "CT", "PA", "FL", "TX", "CA", "OH"]
# INVENTED flat state rates for the scenario prototype only.
STATE_RATES = {"NY": 0.055, "NJ": 0.052, "CT": 0.049, "PA": 0.031,
               "FL": 0.0, "TX": 0.0, "CA": 0.083, "OH": 0.035}

PREPARERS = ["P1", "P2", "P3", "P4", "P5", "P6"]


def make_clients(n=60):
    rows = []
    for i in range(n):
        entity = ENTITY_TYPES[i % len(ENTITY_TYPES)]
        rows.append({
            "client_id": f"C{i + 1:03d}",
            "entity_type": entity,
            "industry": INDUSTRIES[RNG.integers(0, len(INDUSTRIES))],
            "state": STATES[RNG.integers(0, len(STATES))],
            "complexity": int(RNG.integers(1, 6)),
            "staff_assigned": int(RNG.integers(1, 4)),
            "years_with_firm": int(RNG.integers(1, 11)),
            "annual_revenue": float(np.round(RNG.lognormal(13.5, 1.1), 2)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "clients.csv"), index=False)
    return df


def make_journal_entries(clients, n_per_month=1000):
    """One row per journal entry, Jan-Dec 2025. ~3% injected anomalies."""
    rows = []
    entry_id = 0
    months = pd.period_range("2025-01", "2025-12", freq="M")
    client_median = {}
    for _, c in clients.iterrows():
        client_median[c["client_id"]] = float(
            np.exp(RNG.normal(8.0, 0.5)))
    for m in months:
        days = pd.date_range(m.start_time, m.end_time, freq="D")
        weekdays = [d for d in days if d.weekday() < 5]
        for _ in range(n_per_month):
            entry_id += 1
            client = clients.iloc[RNG.integers(0, len(clients))]
            date = weekdays[RNG.integers(0, len(weekdays))]
            amount = float(np.round(RNG.lognormal(8.0, 1.0), 2))
            acct = ACCOUNTS[RNG.integers(0, len(ACCOUNTS))]
            rows.append({
                "entry_id": f"JE{entry_id:06d}",
                "date": date.strftime("%Y-%m-%d"),
                "client_id": client["client_id"],
                "account": acct,
                "amount": amount,
                "debit_account": acct,
                "credit_account": ACCOUNTS[RNG.integers(0, len(ACCOUNTS))],
                "preparer": PREPARERS[RNG.integers(0, len(PREPARERS))],
                "entry_type": RNG.choice(
                    ["Standard", "Adjusting", "Reversing", "Accrual"],
                    p=[0.70, 0.15, 0.05, 0.10]),
                "doc_attached": bool(RNG.random() < 0.93),
                "tie_out_diff": 0.0,
                "is_anomaly": 0,
                "anomaly_type": "none",
            })
    df = pd.DataFrame(rows)
    # Inject anomalies (~3% of rows).
    n_anom = int(0.03 * len(df))
    idx = RNG.choice(df.index, size=n_anom, replace=False)
    types = ["duplicate", "round_number", "weekend", "missing_doc",
             "unbalanced", "unusual_amount"]
    for j, i in enumerate(idx):
        t = types[j % len(types)]
        df.at[i, "is_anomaly"] = 1
        df.at[i, "anomaly_type"] = t
        if t == "duplicate":
            src = df.iloc[RNG.integers(0, len(df))]
            df.at[i, "client_id"] = src["client_id"]
            df.at[i, "account"] = src["account"]
            df.at[i, "amount"] = src["amount"]
            df.at[i, "date"] = src["date"]
        elif t == "round_number":
            df.at[i, "amount"] = float(RNG.integers(5, 60) * 1000)
        elif t == "weekend":
            d = pd.Timestamp(df.at[i, "date"])
            while d.weekday() < 5:
                d += pd.Timedelta(days=1)
            if d > pd.Timestamp("2025-12-31"):
                # Stay inside 2025: use the preceding Saturday instead.
                d = pd.Timestamp(df.at[i, "date"])
                while d.weekday() != 5:
                    d -= pd.Timedelta(days=1)
            df.at[i, "date"] = d.strftime("%Y-%m-%d")
        elif t == "missing_doc":
            df.at[i, "doc_attached"] = False
            df.at[i, "amount"] = float(np.round(
                RNG.lognormal(9.5, 0.5), 2))
        elif t == "unbalanced":
            df.at[i, "tie_out_diff"] = float(np.round(
                RNG.uniform(50, 5000), 2))
        elif t == "unusual_amount":
            med = client_median[df.at[i, "client_id"]]
            df.at[i, "amount"] = float(np.round(med * RNG.uniform(6, 12), 2))
    df.to_csv(os.path.join(OUT, "journal_entries.csv"), index=False)
    return df


def make_close_calendar(clients):
    """One row per client-month: the month-end close process."""
    rows = []
    months = pd.period_range("2025-01", "2025-12", freq="M")
    for _, c in clients.iterrows():
        prior_late = 0
        for m in months:
            busy = 1 if m.month in (1, 2, 3, 4) else 0  # tax season load
            entries = int(RNG.integers(120, 900) * (0.4 + 0.2 * c["complexity"]))
            docs_missing = int(RNG.poisson(1.5 + 0.8 * c["complexity"]))
            recon_break = int(RNG.random() < (0.04 + 0.03 * c["complexity"]))
            base = (3.0 + 1.1 * c["complexity"] + 0.004 * entries
                    + 0.9 * docs_missing + 1.6 * recon_break
                    + 1.2 * busy + 1.5 * prior_late
                    - 0.8 * c["staff_assigned"]
                    + RNG.normal(0, 0.8))
            days = float(np.clip(round(base), 2, 25))
            rows.append({
                "client_id": c["client_id"],
                "month": str(m),
                "entity_type": c["entity_type"],
                "complexity": c["complexity"],
                "staff_assigned": c["staff_assigned"],
                "entries_count": entries,
                "docs_missing": docs_missing,
                "recon_break": recon_break,
                "busy_season": busy,
                "prior_month_late": prior_late,
                "days_to_close": days,
                "late_close": int(days > 10),
            })
            prior_late = int(days > 10)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "close_calendar.csv"), index=False)
    return df


def _invented_liability(income, deductions, credits, state_rate, scenario):
    """INVENTED simplified tax arithmetic — prototype only, NOT tax law.

    Federal: flat invented brackets. Scenario shifts rates/credits.
    """
    adj = {"base": (0.0, 0.0), "bear": (0.02, 0.0), "bull": (-0.02, 0.10)}
    rate_shift, credit_boost = adj[scenario]
    taxable = max(income - deductions, 0.0)
    if taxable <= 50000:
        fed_rate = 0.12 + rate_shift
    elif taxable <= 200000:
        fed_rate = 0.22 + rate_shift
    else:
        fed_rate = 0.30 + rate_shift
    fed = taxable * fed_rate
    state = taxable * state_rate
    total = max(fed + state - credits * (1.0 + credit_boost), 0.0)
    return float(np.round(total, 2))


def make_tax_scenarios(clients):
    """One row per client per scenario (base/bear/bull)."""
    rows = []
    for _, c in clients.iterrows():
        income = float(np.round(RNG.lognormal(12.0, 0.9), 2))
        deductions = float(np.round(income * RNG.uniform(0.15, 0.45), 2))
        credits = float(np.round(RNG.uniform(0, 15000), 2))
        prior_year = float(np.round(income * RNG.uniform(0.10, 0.28), 2))
        for scenario in ("base", "bear", "bull"):
            liability = _invented_liability(
                income, deductions, credits, STATE_RATES[c["state"]], scenario)
            rows.append({
                "client_id": c["client_id"],
                "entity_type": c["entity_type"],
                "state": c["state"],
                "scenario": scenario,
                "taxable_income": float(np.round(max(income - deductions, 0), 2)),
                "deductions": deductions,
                "credits": credits,
                "prior_year_liability": prior_year,
                "projected_liability": liability,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tax_scenarios.csv"), index=False)
    return df


if __name__ == "__main__":
    clients = make_clients()
    je = make_journal_entries(clients)
    close = make_close_calendar(clients)
    tax = make_tax_scenarios(clients)
    print(f"clients: {len(clients)}")
    print(f"journal entries: {len(je)} "
          f"(anomalies: {int(je['is_anomaly'].sum())})")
    print(f"close rows: {len(close)} "
          f"(recon breaks: {int(close['recon_break'].sum())}, "
          f"late closes: {int(close['late_close'].sum())})")
    print(f"tax scenario rows: {len(tax)}")
