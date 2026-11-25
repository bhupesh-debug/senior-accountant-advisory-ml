"""Generate the 4 notebooks as valid nbformat-4 JSON.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python make_notebooks.py

The notebooks are code templates that run the project's own src/
modules; they are not pre-executed (nbformat is not installed here),
and the README says so.
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [text]}


def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": [text]}


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


NOTEBOOKS = {
    "01_eda.ipynb": [
        md("# 01 — Exploratory Data Analysis\n\nSynthetic advisory-practice "
           "data: 60 clients, 12,000 journal entries in 2025 (3% injected "
           "anomalies, 6 types), 720 client-months of close data, 180 "
           "tax scenarios. ALL DATA IS SYNTHETIC."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import pandas as pd\n"
             "clients, je, close, tax = __import__('data_pipeline').load()\n"
             "print(clients.shape, je.shape, close.shape, tax.shape)\n"
             "print(je['is_anomaly'].mean(), close['recon_break'].mean())"),
        code("import matplotlib.pyplot as plt\n"
             "je['amount'].hist(bins=50)\n"
             "plt.title('Journal entry amounts (log scale coming)')\n"
             "plt.show()"),
        code("print(close.groupby('month')['days_to_close'].mean())\n"
             "print(tax.groupby('scenario')['projected_liability']"
             ".describe())"),
    ],
    "02_feature_engineering.ipynb": [
        md("# 02 — Feature Engineering\n\nJE features (amount vs client "
           "median, weekend, missing doc, round number, same-day count, "
           "month-end, adjusting, account rarity); close features; tax "
           "features. Splits: time-based Nov-Dec 2025 for modules 1-2, "
           "grouped by client for module 3."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import feature_engineering as fe\n"
             "import data_pipeline as dp\n"
             "clients, je, close, tax = dp.load()\n"
             "je_f = fe.je_features(je)\n"
             "print(je_f[fe.JE_FEATURES].describe().T[['mean','max']])"),
        code("close_f = fe.close_features(close)\n"
             "train2, test2, cut = fe.close_time_split(close_f)\n"
             "print(len(train2), len(test2), cut)\n"
             "print('recon_break is NOT a feature for module 2b:',\n"
             "      'recon_break' not in fe.CLOSE_FEATURES)"),
    ],
    "03_modeling.ipynb": [
        md("# 03 — Modeling\n\nRule baseline vs Isolation Forest vs "
           "Logistic Regression vs Random Forest (module 1); Linear vs "
           "Random Forest (modules 2a and 3); majority vs Logistic vs "
           "Random Forest (module 2b)."),
        code("import sys\nsys.path.insert(0, '../src')\n"
             "import feature_engineering as fe\n"
             "import data_pipeline as dp\n"
             "import anomaly as an\n"
             "import models as md\n"
             "clients, je, close, tax = dp.load()\n"
             "je_f = fe.je_features(je)\n"
             "je_f['rule_flag'] = an.je_rule_flag(je_f)\n"
             "train1, test1, _ = fe.je_time_split(je_f)\n"
             "m1, scored, imp, _ = md.module1(train1, test1, fe.JE_FEATURES)\n"
             "print(m1)"),
        code("print(imp)\n"
             "print('Review queue top-20 hit rate:',\n"
             "      scored.sort_values('rf_score', ascending=False)"
             "      .head(20)['is_anomaly'].mean())"),
    ],
    "04_evaluation.ipynb": [
        md("# 04 — Evaluation\n\nHold-out metrics from models/*.csv; the "
           "honest findings (rule beats the model in module 1, linear "
           "beats RF in module 2a, break prediction is weak in 2b) and "
           "the MAPE floor note for module 3."),
        code("import pandas as pd\n"
             "for f in ('anomaly_metrics.csv', 'close_cycle_metrics.csv',\n"
             "          'recon_break_metrics.csv', 'tax_metrics.csv'):\n"
             "    print(f)\n"
             "    print(pd.read_csv(f'../models/{f}').to_string(index=False))\n"
             "    print()"),
        code("q = pd.read_csv('../models/review_queue_top20.csv')\n"
             "print('Top-20 queue hit rate:', q['is_anomaly'].mean())\n"
             "print('Base rate in hold-out: 0.033')"),
    ],
}

if __name__ == "__main__":
    os.makedirs(NB, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        with open(os.path.join(NB, name), "w") as fh:
            json.dump(nb(cells), fh, indent=1)
        print("wrote", name)
