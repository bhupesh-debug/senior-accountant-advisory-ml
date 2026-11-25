"""Test suite for the Senior Accountant advisory ML project.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python tests/run_tests.py

Prints the runtime "N passed" line — the test count is that line,
never a count of check( call sites (checks inside loops execute
multiple times).
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import feature_engineering as fe  # noqa: E402

PASSED = 0
FAILED = []


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILED.append(f"{name} {detail}")


# --- File existence -------------------------------------------------------
for f in ("README.md", "requirements.txt", "run_all.py", "build_pdf.py",
          "build_deck.py", "make_notebooks.py"):
    check(f"file exists: {f}", os.path.exists(os.path.join(ROOT, f)))
for f in ("data_generation.py", "data_pipeline.py",
          "feature_engineering.py", "anomaly.py", "models.py", "charts.py"):
    check(f"src exists: {f}",
          os.path.exists(os.path.join(ROOT, "src", f)))
for f in ("clients.csv", "journal_entries.csv", "close_calendar.csv",
          "tax_scenarios.csv"):
    check(f"synthetic data exists: {f}",
          os.path.exists(os.path.join(ROOT, "data", "synthetic", f)))
for f in ("anomaly_metrics.csv", "anomaly_holdout_scored.csv",
          "anomaly_feature_importance.csv", "review_queue_top20.csv",
          "close_cycle_metrics.csv", "recon_break_metrics.csv",
          "close_feature_importance.csv", "tax_metrics.csv",
          "tax_feature_importance.csv"):
    check(f"model output exists: {f}",
          os.path.exists(os.path.join(ROOT, "models", f)))
for f in ("anomaly_importance.png", "anomaly_f1.png", "close_mape.png",
          "close_dist.png", "tax_scenarios.png", "review_queue.png"):
    check(f"chart exists: {f}",
          os.path.exists(os.path.join(ROOT, "reports", "charts", f)))
check("model report exists",
      os.path.exists(os.path.join(ROOT, "reports", "model_report.md")))

# --- Data schema + validation ----------------------------------------------
clients = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                   "clients.csv"))
je = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                             "journal_entries.csv"), parse_dates=["date"])
close = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                 "close_calendar.csv"))
tax = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                               "tax_scenarios.csv"))

check("clients: 60 rows", len(clients) == 60, f"got {len(clients)}")
check("clients: 5 entity types",
      clients["entity_type"].nunique() == 5)
check("clients: no missing values", int(clients.isna().sum().sum()) == 0)
check("JE: 12000 rows", len(je) == 12000, f"got {len(je)}")
check("JE: anomaly rate ~3%",
      0.02 < je["is_anomaly"].mean() < 0.04,
      f"got {je['is_anomaly'].mean():.4f}")
check("JE: dates inside 2025",
      je["date"].min() >= pd.Timestamp("2025-01-01")
      and je["date"].max() <= pd.Timestamp("2025-12-31"))
check("JE: no negative amounts", int((je["amount"] <= 0).sum()) == 0)
check("JE: no missing values", int(je.isna().sum().sum()) == 0)
check("JE: 6 anomaly types present",
      je.loc[je["is_anomaly"] == 1, "anomaly_type"].nunique() == 6)
check("close: 720 rows (60 clients x 12 months)", len(close) == 720)
check("close: no missing values", int(close.isna().sum().sum()) == 0)
check("close: break rate 5-25%",
      0.05 < close["recon_break"].mean() < 0.25,
      f"got {close['recon_break'].mean():.3f}")
check("close: days_to_close in 2..25",
      close["days_to_close"].min() >= 2
      and close["days_to_close"].max() <= 25)
check("tax: 180 rows (60 clients x 3 scenarios)", len(tax) == 180)
check("tax: exactly base/bear/bull",
      sorted(tax["scenario"].unique().tolist())
      == ["base", "bear", "bull"])
check("tax: no negative liabilities",
      int((tax["projected_liability"] < 0).sum()) == 0)
check("tax: no missing values", int(tax.isna().sum().sum()) == 0)

# --- Hold-out design --------------------------------------------------------
je_f = fe.je_features(je)
train1, test1, _ = fe.je_time_split(je_f)
check("module 1: hold-out starts 2025-11-01",
      test1["date"].min() >= pd.Timestamp("2025-11-01"))
check("module 1: hold-out is Nov-Dec only",
      test1["date"].max() <= pd.Timestamp("2025-12-31"))
check("module 1: train/hold-out disjoint",
      len(set(train1["entry_id"]) & set(test1["entry_id"])) == 0)
close_f = fe.close_features(close)
train2, test2, _ = fe.close_time_split(close_f)
check("module 2: hold-out is Nov-Dec 2025",
      sorted(test2["month"].unique().tolist()) == ["2025-11", "2025-12"])
import data_generation as dg  # noqa: E402
tax_f = fe.tax_features(tax, dg.STATE_RATES)
train3, test3 = fe.tax_group_split(tax_f)
check("module 3: no client in both train and test",
      len(set(train3["client_id"]) & set(test3["client_id"])) == 0)
check("module 3: 18 clients held out",
      test3["client_id"].nunique() == 18,
      f"got {test3['client_id'].nunique()}")
check("module 2b: recon_break NOT in feature list",
      "recon_break" not in fe.CLOSE_FEATURES)
check("module 2a: recon_break IS a driver",
      "recon_break" in fe.CLOSE_FEATURES_A)

# --- Metrics sanity ----------------------------------------------------------
m1 = pd.read_csv(os.path.join(ROOT, "models", "anomaly_metrics.csv"))
check("module 1: 4 models scored", len(m1) == 4, f"got {len(m1)}")
for _, r in m1.iterrows():
    check(f"module 1: {r['model']} F1 in [0,1]",
          0.0 <= r["f1"] <= 1.0, f"got {r['f1']}")
rf_f1 = m1.loc[m1["model"] == "Random Forest", "f1"].iloc[0]
rule_f1 = m1.loc[m1["model"] == "Rule baseline", "f1"].iloc[0]
check("module 1: rule baseline F1 beats RF (honest finding)",
      rule_f1 > rf_f1, f"rule {rule_f1:.3f} vs RF {rf_f1:.3f}")
q = pd.read_csv(os.path.join(ROOT, "models", "review_queue_top20.csv"))
check("review queue: 20 rows", len(q) == 20, f"got {len(q)}")
check("review queue: all top-20 are true anomalies",
      float(q["is_anomaly"].mean()) == 1.0,
      f"got {q['is_anomaly'].mean():.2f}")
check("review queue: hit rate beats base rate",
      float(q["is_anomaly"].mean()) > 0.10)

m2a = pd.read_csv(os.path.join(ROOT, "models", "close_cycle_metrics.csv"))
rf_mape = m2a.loc[m2a["model"] == "Random Forest", "mape"].iloc[0]
lin_mape = m2a.loc[m2a["model"] == "Linear Regression", "mape"].iloc[0]
check("module 2a: linear MAPE < 15%", lin_mape < 0.15, f"got {lin_mape:.3f}")
check("module 2a: RF MAPE < 20%", rf_mape < 0.20, f"got {rf_mape:.3f}")
check("module 2a: linear beats RF (honest finding)", lin_mape < rf_mape)

m2b = pd.read_csv(os.path.join(ROOT, "models", "recon_break_metrics.csv"))
rf_macro = m2b.loc[m2b["model"] == "Random Forest", "macro_f1"].iloc[0]
maj_macro = m2b.loc[m2b["model"] == "Majority baseline", "macro_f1"].iloc[0]
check("module 2b: RF macro-F1 beats majority baseline",
      rf_macro > maj_macro, f"RF {rf_macro:.3f} vs maj {maj_macro:.3f}")
check("module 2b: RF macro-F1 reported honestly (weak, ~0.5)",
      0.45 < rf_macro < 0.65, f"got {rf_macro:.3f}")

m3 = pd.read_csv(os.path.join(ROOT, "models", "tax_metrics.csv"))
lin_r2 = m3.loc[m3["model"] == "Linear Regression", "r2"].iloc[0]
rf_mae = m3.loc[m3["model"] == "Random Forest", "mae"].iloc[0]
base_mae = m3.loc[m3["model"] == "Prior-year baseline", "mae"].iloc[0]
check("module 3: linear R2 > 0.9", lin_r2 > 0.9, f"got {lin_r2:.3f}")
check("module 3: RF MAE beats prior-year baseline",
      rf_mae < base_mae, f"RF {rf_mae:.0f} vs base {base_mae:.0f}")
check("module 3: MAPE uses the $1,000 floor (mape_n < hold-out)",
      int(m3.loc[m3["model"] == "Random Forest", "mape_n"].iloc[0]) < 54)

# --- Honesty guards -----------------------------------------------------------
report = open(os.path.join(ROOT, "reports", "model_report.md")).read()
check("report: states synthetic data", "synthetic" in report.lower())
check("report: states prototype is not tax advice",
      "never tax advice" in report.lower()
      or "not tax advice" in report.lower())
check("report: no CPA Candidate claim", "CPA Candidate" not in report)
check("report: no CCH/Axcess/QuickBooks claim",
      "CCH" not in report and "Axcess" not in report
      and "QuickBooks" not in report)

print(f"{PASSED} passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print("FAIL:", f)
    sys.exit(1)
print("ALL TESTS PASSED")
