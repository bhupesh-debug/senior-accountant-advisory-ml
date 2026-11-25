# Tax Preparation & Client Accounting Analytics

**ML portfolio project for the Senior Accountant role (advisory practice, Newburgh NY)**

Author: Bhupesh Chandra Dimri · November 2026

> **ALL DATA IS SYNTHETIC.** No real client, no real tax return, no real
> ledger is used anywhere. The generator mimics the structure of the job
> description (client tax returns, AR/AP/GL accounting, month-end close,
> tax planning scenarios) so the models can be built and tested honestly.
> Magnitudes, anomaly rates, break rates and tax rates are design
> choices, not facts about any firm or client.
>
> **TAX WARNING — read first.** The Module 3 scenario engine uses
> INVENTED, simplified rates. It is an analytics/education prototype,
> NEVER a tax preparation or tax advice tool. Nothing here may be used
> to prepare a return or advise a client.

## The problem (from the JD)

A Senior Accountant in an advisory practice juggles three things at
once: returns that must be right before they are filed, a dozen
clients' books that must close on time every month, and tax planning
that has to survive a client's "what if?" questions. This project
builds the three checks a senior runs every cycle — as reproducible,
testable code.

## Three modules

| # | Module | What it does | Model |
|---|--------|--------------|-------|
| 1 | Return-readiness & accuracy checker | Flags bad journal entries (duplicates, round numbers, weekend postings, missing docs, unbalanced, out-of-pattern amounts) with a ranked review queue | Rule baseline vs Isolation Forest vs Logistic Regression vs Random Forest |
| 2 | Close-process analytics | Forecasts month-end close cycle time; predicts reconciliation breaks | Median baseline vs Linear Regression vs Random Forest (cycle time); majority vs Logistic vs Random Forest (breaks) |
| 3 | Tax-planning estimator | Scenario-based projected liability (base/bear/bull) on synthetic client data — prototype only | Prior-year baseline vs Linear Regression vs Random Forest |

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py          # generate -> validate -> features -> models -> charts
python tests/run_tests.py  # test suite (exit 0 = all pass)
python build_pdf.py        # 10-page project report -> ~/workspace/your_files/
python build_deck.py       # 10-slide deck PDF -> ~/workspace/your_files/
```

Deterministic: seed 42 everywhere. Modules 1 and 2 use a time-based
hold-out (Nov-Dec 2025); module 3 uses a grouped split by client
(70/30, 18 of 60 clients held out) so no client appears in both
train and test.

## Results (hold-out)

See `reports/model_report.md` for the full numbers. Headline results
from the run that produced this README:

- **Module 1:** the transparent rule baseline (round-number + missing-doc
  + unbalanced + weekend) beats the models on F1 — the injected
  anomalies are rule-shaped by construction, and the honest finding is
  stated, not hidden. The Random Forest review queue still concentrates
  the real errors: top-20 ranked entries are 20/20 true anomalies vs
  a 3.3% base rate.
- **Module 2a:** Linear Regression forecasts close cycle time at 7.8%
  MAPE (vs 43.7% median baseline); Random Forest 10.9%. The cycle-time
  drivers are near-linear by construction — the simpler model wins.
- **Module 2b:** reconciliation-break prediction is the weakest module
  (Random Forest macro-F1 0.505 vs 0.459 majority baseline). Breaks in
  this synthetic data are near-random; the report says so and names
  the richer features a real deployment would need.
- **Module 3:** Linear Regression R² 0.984, MAE $5,235 vs $16,952
  prior-year baseline. MAPE is reported only on liabilities over
  $1,000 (45 of 54 hold-out rows) because 23 synthetic liabilities
  are exactly $0.

## Project structure

```
ml-project-senior-accountant-advisory/
├── README.md
├── requirements.txt
├── run_all.py                 # orchestrator
├── build_pdf.py               # 10-page report builder
├── build_deck.py              # 10-slide deck builder (PDF)
├── make_notebooks.py          # generates notebooks/01-04
├── src/
│   ├── data_generation.py   # synthetic clients / JEs / close / tax
│   ├── data_pipeline.py     # validation + cleaning
│   ├── feature_engineering.py
│   ├── anomaly.py           # rule-based baselines
│   ├── models.py            # training + evaluation
│   └── charts.py            # matplotlib figures
├── tests/run_tests.py
├── data/synthetic/          # generated CSVs (regenerate with run_all.py)
├── data/processed/          # validated copies
├── models/*.csv             # every metric the report cites
├── reports/charts/*.png
├── reports/model_report.md
└── notebooks/01_eda.ipynb … 04_evaluation.ipynb
```

## Limits — read before citing any number

- Synthetic data proves the **pipeline design**, not performance on real
  books. Injected anomalies are deliberately detectable; real errors
  are subtler — expect lower recall on live data and keep a human in
  the review queue.
- The tax scenario engine uses invented rates. It demonstrates scenario
  modeling, not tax law. Never cite its liabilities as real.
- The candidate does **not** claim tax-return preparation experience,
  CPA candidacy, or experience with CCH Prosystem, Axcess, Thomson
  Reuters Fixed Assets CS, Checkpoint, Sage Intacct, Bill.com,
  QuickBooks or Xero — all are [CONFIRM WITH CANDIDATE] items for the
  role, not facts in this repo. NetSuite IS in his background.
- The recon-break predictor is weak by design of the synthetic data;
  do not present it as production-ready.
