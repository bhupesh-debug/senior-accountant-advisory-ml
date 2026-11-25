"""Orchestrator: generate -> validate -> features -> models -> metrics.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python run_all.py

Deterministic: seed 42 everywhere. Module 1 and 2 use a time-based
hold-out (Nov-Dec 2025); module 3 uses a grouped split by client
(70/30) so no client appears in both train and test. Every metric in
the PDF and deck comes from models/*.csv written here.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import anomaly as an  # noqa: E402
import charts as ch  # noqa: E402
import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import models as md  # noqa: E402

MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)
os.makedirs(os.path.join(ROOT, "data", "processed"), exist_ok=True)

# 1. Generate + validate -------------------------------------------------
syn_je = os.path.join(ROOT, "data", "synthetic", "journal_entries.csv")
if not os.path.exists(syn_je):
    subprocess.run([sys.executable, os.path.join(ROOT, "src",
                                                 "data_generation.py")],
                   check=True)
clients, je, close, tax = dp.load()
findings = dp.validate(clients, je, close, tax)
print("validation:", findings)

# 2. Module 1 — JE anomaly detection ------------------------------------
je_f = fe.je_features(je)
je_f["rule_flag"] = an.je_rule_flag(je_f)
train1, test1, cut1 = fe.je_time_split(je_f)
print(f"module 1: train {len(train1)}, hold-out {len(test1)} "
      f"(from {cut1.date()})")
m1, scored1, imp1, _ = md.module1(train1, test1, fe.JE_FEATURES)
m1.to_csv(os.path.join(MODELS, "anomaly_metrics.csv"), index=False)
scored1.to_csv(os.path.join(MODELS, "anomaly_holdout_scored.csv"),
               index=False)
imp1.to_csv(os.path.join(MODELS, "anomaly_feature_importance.csv"),
            index=False)
top20_idx = an.top_n_queue(scored1["rf_score"].values, 20)
top20 = scored1.iloc[top20_idx][
    ["entry_id", "date", "client_id", "account", "amount",
     "anomaly_type", "is_anomaly", "rf_score"]]
top20.to_csv(os.path.join(MODELS, "review_queue_top20.csv"), index=False)
base_rate = float(test1["is_anomaly"].mean())
top20_hit = float(top20["is_anomaly"].mean())
print(f"module 1: RF F1 "
      f"{m1.loc[m1['model'] == 'Random Forest', 'f1'].iloc[0]:.3f}; "
      f"review queue top-20 hit rate {top20_hit:.2f} vs base {base_rate:.3f}")

# 3. Module 2 — close analytics ------------------------------------------
close_f = fe.close_features(close)
train2, test2, cut2 = fe.close_time_split(close_f)
print(f"module 2: train {len(train2)}, hold-out {len(test2)} "
      f"(from {cut2})")
m2a, imp2a, _ = md.module2a(train2, test2, fe.CLOSE_FEATURES_A)
m2a.to_csv(os.path.join(MODELS, "close_cycle_metrics.csv"), index=False)
m2b, _ = md.module2b(train2, test2, fe.CLOSE_FEATURES)
m2b.to_csv(os.path.join(MODELS, "recon_break_metrics.csv"), index=False)
imp2a.to_csv(os.path.join(MODELS, "close_feature_importance.csv"),
             index=False)
print(f"module 2a: RF MAPE "
      f"{m2a.loc[m2a['model'] == 'Random Forest', 'mape'].iloc[0]:.3f}")
print(f"module 2b: RF macro-F1 "
      f"{m2b.loc[m2b['model'] == 'Random Forest', 'macro_f1'].iloc[0]:.3f}")

# 4. Module 3 — tax-planning estimator ------------------------------------
tax_f = fe.tax_features(tax, dg.STATE_RATES)
train3, test3 = fe.tax_group_split(tax_f)
print(f"module 3: train {len(train3)}, hold-out {len(test3)} "
      f"(grouped by client)")
m3, imp3, _ = md.module3(train3, test3, fe.TAX_FEATURES)
m3.to_csv(os.path.join(MODELS, "tax_metrics.csv"), index=False)
imp3.to_csv(os.path.join(MODELS, "tax_feature_importance.csv"),
            index=False)
print(f"module 3: RF MAPE "
      f"{m3.loc[m3['model'] == 'Random Forest', 'mape'].iloc[0]:.3f}")

# 5. Charts + model report ------------------------------------------------
ch.feature_importance(imp1, os.path.join(CHARTS, "anomaly_importance.png"),
                      "Module 1 — what flags a bad journal entry")
ch.model_comparison(m1, "f1", os.path.join(CHARTS, "anomaly_f1.png"),
                    "Module 1 — anomaly detection F1 by model", "F1")
ch.model_comparison(m2a, "mape", os.path.join(CHARTS, "close_mape.png"),
                    "Module 2a — close cycle-time MAPE by model", "MAPE")
ch.close_distribution(close, os.path.join(CHARTS, "close_dist.png"))
ch.scenario_liability(tax, os.path.join(CHARTS, "tax_scenarios.png"))
ch.review_queue_concentration(scored1,
                              os.path.join(CHARTS, "review_queue.png"))

report = []
report.append("# Tax Preparation & Client Accounting Analytics — model report\n")
report.append("All data synthetic. Tax rates invented for the prototype; "
              "this is an analytics/education prototype, never tax advice.\n")
report.append("## Module 1 — return-readiness & accuracy checker\n")
report.append(m1.to_markdown(index=False))
report.append(f"\nHold-out: {len(test1)} journal entries, Nov-Dec 2025. "
              f"Anomaly base rate {base_rate:.3f}. Review queue top-20 "
              f"hit rate {top20_hit:.2f}.\n")
report.append("\n## Module 2a — close cycle-time forecast\n")
report.append(m2a.to_markdown(index=False))
report.append("\n## Module 2b — reconciliation-break prediction\n")
report.append(m2b.to_markdown(index=False))
report.append(f"\nHold-out: {len(test2)} client-months, Nov-Dec 2025.\n")
report.append("\n## Module 3 — tax-planning estimator\n")
report.append(m3.to_markdown(index=False))
report.append(f"\nHold-out: {len(test3)} client-scenarios, grouped by "
              f"client (18 of 60 clients held out).\n")
with open(os.path.join(ROOT, "reports", "model_report.md"), "w") as fh:
    fh.write("\n".join(report))
print("done — metrics in models/, charts in reports/charts/")
