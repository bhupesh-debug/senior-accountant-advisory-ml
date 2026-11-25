"""Charts for the Senior Accountant project (matplotlib, PNG output)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feature_importance(df, path, title, n=8):
    top = df.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["feature"], top["importance"], color="#1f4e79")
    ax.set_title(title)
    ax.set_xlabel("Permutation-free importance (RF)")
    _save(fig, path)


def model_comparison(metrics, value_col, path, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics["model"], metrics[value_col], color="#2e9e6b")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    _save(fig, path)


def close_distribution(close, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(close["days_to_close"], bins=range(0, 27, 2),
            color="#1f4e79", edgecolor="white")
    ax.axvline(10, color="#c0392b", linestyle="--",
               label="Late close (>10 days)")
    ax.set_title("Month-end close cycle time (days, all client-months)")
    ax.set_xlabel("Days to close")
    ax.set_ylabel("Client-months")
    ax.legend()
    _save(fig, path)


def scenario_liability(tax, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scen, color in (("base", "#1f4e79"), ("bear", "#c0392b"),
                        ("bull", "#2e9e6b")):
        vals = tax[tax["scenario"] == scen]["projected_liability"]
        ax.hist(vals, bins=20, alpha=0.6, label=scen, color=color)
    ax.set_title("Projected liability by scenario (synthetic, invented rates)")
    ax.set_xlabel("Projected liability ($)")
    ax.set_ylabel("Clients")
    ax.legend()
    _save(fig, path)


def review_queue_concentration(scored, path):
    """Share of true anomalies captured in the top-N of the ranked queue."""
    ranked = scored.sort_values("rf_score", ascending=False)
    cum = ranked["is_anomaly"].cumsum() / ranked["is_anomaly"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(cum) + 1), cum.values, color="#1f4e79")
    ax.axhline(1.0, color="#c0392b", linestyle="--", label="All anomalies")
    ax.set_title("Review queue: cumulative anomalies captured vs queue depth")
    ax.set_xlabel("Queue depth (ranked by model score)")
    ax.set_ylabel("Share of hold-out anomalies captured")
    ax.legend()
    _save(fig, path)
