# Tax Preparation & Client Accounting Analytics — model report

All data synthetic. Tax rates invented for the prototype; this is an analytics/education prototype, never tax advice.

## Module 1 — return-readiness & accuracy checker

|   accuracy |   precision |   recall |       f1 |   macro_f1 | model               |    roc_auc |
|-----------:|------------:|---------:|---------:|-----------:|:--------------------|-----------:|
|   0.967936 |    0.505263 | 0.738462 | 0.6      |   0.791649 | Rule baseline       | nan        |
|   0.957916 |    0.306122 | 0.230769 | 0.263158 |   0.620749 | Isolation Forest    |   0.72281  |
|   0.97996  |    1        | 0.384615 | 0.555556 |   0.772652 | Logistic Regression |   0.810927 |
|   0.973948 |    0.632653 | 0.476923 | 0.54386  |   0.765225 | Random Forest       |   0.816659 |

Hold-out: 1996 journal entries, Nov-Dec 2025. Anomaly base rate 0.033. Review queue top-20 hit rate 1.00.


## Module 2a — close cycle-time forecast

|      mape |   mape_n |      mae |          r2 | model             |
|----------:|---------:|---------:|------------:|:------------------|
| 0.436934  |      120 | 3.56667  | -0.00189896 | Median baseline   |
| 0.0779812 |      120 | 0.649464 |  0.962737   | Linear Regression |
| 0.108518  |      120 | 0.9175   |  0.926306   | Random Forest     |

## Module 2b — reconciliation-break prediction

|   accuracy |   precision |   recall |       f1 |   macro_f1 | model               |    roc_auc |
|-----------:|------------:|---------:|---------:|-----------:|:--------------------|-----------:|
|   0.85     |    0        | 0        | 0        |   0.459459 | Majority baseline   | nan        |
|   0.85     |    0        | 0        | 0        |   0.459459 | Logistic Regression |   0.602941 |
|   0.783333 |    0.166667 | 0.111111 | 0.133333 |   0.504762 | Random Forest       |   0.535675 |

Hold-out: 120 client-months, Nov-Dec 2025.


## Module 3 — tax-planning estimator

|     mape |   mape_n |      mae |       r2 | model               |
|---------:|---------:|---------:|---------:|:--------------------|
| 1.1982   |       45 | 16952.1  | 0.758306 | Prior-year baseline |
| 0.409705 |       45 |  5234.57 | 0.983855 | Linear Regression   |
| 0.504995 |       45 |  6792.78 | 0.947972 | Random Forest       |

Hold-out: 54 client-scenarios, grouped by client (18 of 60 clients held out).
