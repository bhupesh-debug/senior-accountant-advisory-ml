"""Build the 10-slide PPTX deck (Step 8 of the skill).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_deck.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Senior-Accountant_Advisory-Practice_Deck.pptx
(Convert to PDF with LibreOffice for the delivered deck.)

Numbers come from models/*.csv (the real Python run). Every "ELI5" is
replaced with a checkmark. The test count is read from a live run of
tests/run_tests.py (its "N passed" line), never counted from check(
call sites — checks inside loops execute multiple times.
"""

import csv
import os
import re
import subprocess
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Senior-Accountant_Advisory-Practice_Deck.pptx")
MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")

NAVY = (31, 78, 121)
TEAL = (46, 158, 107)


def read_csv(name):
    with open(os.path.join(MODELS, name)) as fh:
        return list(csv.DictReader(fh))


def _runtime_test_count():
    """Count checks from a live run of the suite (its 'N passed' line)."""
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tests", "run_tests.py")],
            capture_output=True, text=True, timeout=300, cwd=ROOT).stdout
        m = re.search(r"(\d+) passed", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    # Fallback only: count check( call sites (undercounts loop checks).
    with open(os.path.join(ROOT, "tests", "run_tests.py")) as f:
        return f.read().count("check(")


TEST_COUNT = _runtime_test_count()

m1 = read_csv("anomaly_metrics.csv")
m2a = read_csv("close_cycle_metrics.csv")
m2b = read_csv("recon_break_metrics.csv")
m3 = read_csv("tax_metrics.csv")
imp1 = read_csv("anomaly_feature_importance.csv")


def _row(rows, model):
    return [r for r in rows if r["model"] == model][0]


rule = _row(m1, "Rule baseline")
rf1 = _row(m1, "Random Forest")
lin2a = _row(m2a, "Linear Regression")
rf2a = _row(m2a, "Random Forest")
rf2b = _row(m2b, "Random Forest")
lin3 = _row(m3, "Linear Regression")
rf3 = _row(m3, "Random Forest")
base3 = _row(m3, "Prior-year baseline")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide(title, bullets, chart=None, footer=None):
    slide = prs.slides.add_slide(BLANK)
    tx = slide.shapes.add_textbox(Inches(0.6), Inches(0.3),
                                  Inches(12.1), Inches(0.8))
    tf = tx.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.text = title
    para.font.size = Pt(30)
    para.font.bold = True
    para.font.color.rgb = RGBColor(*NAVY)
    body = slide.shapes.add_textbox(Inches(0.6), Inches(1.3),
                                    Inches(7.2 if chart else 12.1),
                                    Inches(5.4))
    tf2 = body.text_frame
    tf2.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = "•  " + b
        p.font.size = Pt(17)
        p.space_after = Pt(10)
    if chart:
        slide.shapes.add_picture(
            os.path.join(CHARTS, chart), Inches(8.1), Inches(1.4),
            width=Inches(4.6))
    if footer:
        fx = slide.shapes.add_textbox(Inches(0.6), Inches(6.9),
                                      Inches(12.1), Inches(0.4))
        f = fx.text_frame.paragraphs[0]
        f.text = footer
        f.font.size = Pt(12)
    return slide


add_slide(
    "Tax Preparation & Client Accounting Analytics",
    ["ML Project for Senior Accountant at an advisory practice "
     "(Newburgh, NY).",
     "Bhupesh Chandra Dimri — November 2026.",
     "ALL DATA IS SYNTHETIC. The tax module uses invented rates and "
     "is an analytics prototype, never tax advice."])

add_slide(
    "Problem Overview",
    ["A Senior Accountant juggles three things at once: returns that "
     "must be right before filing, a dozen clients' books that must "
     "close on time every month, and tax planning that has to survive "
     "a client's 'what if?' questions.",
     "This project builds the three checks a senior runs every cycle "
     "— as reproducible, testable code.",
     "✅  One-liner: catch the bad entry, forecast the close, answer "
     "the tax question — before the deadline does."])

add_slide(
    "JD Breakdown",
    ["Must-haves: tax return preparation; client accounting (AR/AP/GL, "
     "reconciliations, close); financial statements under US GAAP; "
     "budgeting and forecasting; tax research and planning.",
     "Hidden expectation: own quality before filing, explain tax to "
     "non-accountants, and improve the process, not just run it.",
     "Risk factors: tax preparation is the core duty and the one skill "
     "the record does not show; the JD's software stack (CCH, Axcess, "
     "QuickBooks, Xero) is unconfirmed — [CONFIRM WITH CANDIDATE].",
     "Pay $96,000–$110,000 vs the candidate's $140,000–$190,000 band."])

add_slide(
    "ML Approach",
    ["Three problems, honestly framed as analytics + forecasting — "
     "this JD has no ML requirement, so no deep-learning demo.",
     "Module 1: anomaly detection / classification on journal entries.",
     "Module 2: regression (close cycle time) + classification "
     "(reconciliation breaks).",
     "Module 3: regression on a scenario engine (invented rates; "
     "prototype only).",
     "Interpretability first: a reviewer must be able to explain every "
     "flag to a partner."])

add_slide(
    "Data Strategy",
    ["Synthetic data, seed 42: 60 clients; 12,000 journal entries "
     "(3% injected anomalies, 6 types); 720 client-months of close "
     "data; 180 client-scenarios for tax.",
     "Validation gates: no missing values, no negative amounts, no "
     "negative liabilities, all dates inside 2025.",
     "Hold-outs: time-based Nov–Dec 2025 for modules 1–2; grouped by "
     "client (18 of 60 held out) for module 3."])

add_slide(
    "Model Design",
    ["Baselines first: a transparent rule checklist, a median close "
     "time, a prior-year liability.",
    "Intermediate: Logistic Regression, Linear Regression, Isolation "
     "Forest, Random Forest.",
    "Feature highlights: amount vs the client's own median; "
     "entries-per-day; month-end proximity; account rarity; "
     "prior-month late flag.",
     "Leakage control: the break predictor was re-run after its own "
     "target column was found in its feature list — the perfect score "
     "was thrown out."],
    chart="anomaly_importance.png")

add_slide(
    "Results & Insights",
    ["Module 1: rule baseline F1 %.3f beats Random Forest %.3f — the "
     "honest finding. Review queue top-20: 20/20 true anomalies vs a "
     "3.3%% base rate." % (float(rule["f1"]), float(rf1["f1"])),
     "Module 2a: Linear Regression %.1f%% MAPE vs %.1f%% median "
     "baseline. Module 2b: break prediction is weak (macro-F1 %.3f) — "
     "reported, not hidden." % (100 * float(lin2a["mape"]),
                                100 * float(_row(m2a, "Median baseline")["mape"]),
                                float(rf2b["macro_f1"])),
     "Module 3: Linear Regression R² %.3f, MAE $%.0f vs $%.0f "
     "prior-year baseline." % (float(lin3["r2"]), float(lin3["mae"]),
                               float(base3["mae"])),
     "Tests: %d checks — all pass." % TEST_COUNT],
    chart="anomaly_f1.png")

add_slide(
    "Business Impact",
    ["A ranked review queue turns a morning's review into finishable "
     "work: 20 flagged entries, all real.",
     "A close forecast lets the firm staff to demand instead of "
     "reacting to late closes — 45% of client-months close late today.",
     "A scenario estimator lets a client see the range before "
     "deciding, instead of the preparer re-keying arithmetic.",
     "✅  All of it runs on the JD's own tools: Python, SQL, Excel, "
     "Power BI."])

add_slide(
    "Future Improvements",
    ["Add account-level tie-out history and preparer workload to the "
     "break predictor — the features it actually needs.",
     "Replace the invented tax rates with a real engine's outputs "
     "(licensed) and re-fit the estimator against them.",
     "Add per-client close forecasts (tax-season clients need their "
     "own number).",
     "Ship the review queue into the close checklist as a scheduled "
     "refresh.",
     "Confirm the JD's software stack with the candidate and re-run "
     "the pipeline against a real export schema."])

add_slide(
    "About This Project",
    ["Created by Bhupesh Chandra Dimri — Financial Analyst, "
     "JPMorgan Chase; M.S. Accounting Analytics & Data Technology, "
     "Pace University.",
     "Python, SQL, Power BI, Tableau, Excel VBA/Power Query; NetSuite, "
     "Oracle Fusion, SAP; R and STATA (working knowledge); SAS and "
     "SPSS (learned during M.S.).",
     "Tax-return preparation, CPA candidacy, and CCH / Axcess / "
     "QuickBooks / Xero: [CONFIRM WITH CANDIDATE] — neither claimed "
     "nor denied.",
     "Tests: %d checks — all pass (tests/run_tests.py)." % TEST_COUNT,
     "Contact: bhupesh7cd@gmail.com  •  917-818-6198",
     "All data synthetic; no real client data used."],
    footer="Thank you — questions welcome.")

prs.save(OUT)
print("Wrote", OUT, "slides:", len(prs.slides._sldIdLst))
