"""Visual matrix output: macro sentiment through time, surprise matrix, permutation x regime
heatmap and walk-forward equity curves. All figures are saved as PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features.macro_features import add_surprise_z


def plot_sentiment_timeline(daily: pd.DataFrame, events: pd.DataFrame, path: Path) -> Path:
    d = daily.set_index("date")
    d = d.loc[: pd.Timestamp.now().normalize()] if d.index.max() > pd.Timestamp.now().normalize() else d
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(d.index, d["viz_risk_sentiment"], color="tab:green", lw=1.6)
    axes[0].axhline(0, color="grey", lw=0.8); axes[0].set_ylabel("Risk sentiment\n(+ risk-on)")
    axes[1].plot(d.index, d["viz_hawk_dove"], color="tab:red", lw=1.6)
    axes[1].axhline(0, color="grey", lw=0.8); axes[1].set_ylabel("Hawkish(+) / Dovish(-)\nnews tone")
    axes[2].bar(d.index, d["viz_surprise_ewm"], color=np.where(d["viz_surprise_ewm"] >= 0, "tab:orange", "tab:blue"), width=1.0)
    axes[2].axhline(0, color="grey", lw=0.8); axes[2].set_ylabel("Data surprise\n(EWM z-score)")
    if len(events):
        ev = add_surprise_z(events)
        colors = {"CPI": "tab:red", "FOMC": "black", "EMPLOYMENT": "tab:purple", "PCE": "tab:brown", "GDP": "tab:cyan"}
        for _, r in ev.iterrows():
            if r["event_date"] in d.index:
                axes[0].axvline(r["event_date"], color=colors.get(r["event_type"], "grey"), alpha=0.25, lw=1)
        handles = [plt.Line2D([0], [0], color=c, lw=2, label=t) for t, c in colors.items()]
        axes[0].legend(handles=handles, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Macro sentiment & data surprises over time", fontsize=14, y=0.995)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def plot_surprise_matrix(events: pd.DataFrame, path: Path) -> Path:
    """Matrix: event type (rows) x ISO week (columns), colour = standardised surprise."""
    ev = add_surprise_z(events).dropna(subset=["surprise_z"])
    fig, ax = plt.subplots(figsize=(13, 4))
    if ev.empty:
        ax.text(0.5, 0.5, "No realised releases with forecasts", ha="center"); ax.axis("off")
    else:
        ev["week"] = ev["event_date"].dt.to_period("W").dt.start_time
        piv = ev.pivot_table(index="event_type", columns="week", values="surprise_z", aggfunc="mean")
        piv = piv.reindex(columns=pd.date_range(piv.columns.min(), piv.columns.max(), freq="W-MON"))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        step = max(1, len(piv.columns) // 12)
        ax.set_xticks(range(0, len(piv.columns), step))
        ax.set_xticklabels([c.strftime("%b %d") for c in piv.columns[::step]], rotation=45, ha="right")
        fig.colorbar(im, ax=ax, label="surprise z (red = hotter / stronger than forecast)")
    ax.set_title("Macro surprise matrix (event type x week)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def plot_permutation_regime_heatmap(panel: pd.DataFrame, path: Path) -> Path:
    """Matrix: strategy permutation (rows) x macro regime (columns), colour = mean daily P&L."""
    h = panel[panel["is_future"] == 0].copy()
    h["regime"] = "No event"
    for t in config.EVENT_TYPES:
        h.loc[h[f"is_{t.lower()}_day"] == 1, "regime"] = f"{t} day"
    tilt = h["hawk_dove_ewm"]
    h["tilt"] = np.where(tilt > 0.1, "Hawkish tilt", np.where(tilt < -0.1, "Dovish tilt", "Neutral tilt"))
    cols = pd.concat([h.assign(col=h["regime"]), h[h["regime"] == "No event"].assign(col=h["tilt"])])
    piv = cols.pivot_table(index="permutation_id", columns="col", values="y", aggfunc="mean")
    order = [c for c in ["No event"] + [f"{t} day" for t in config.EVENT_TYPES] + ["Hawkish tilt", "Neutral tilt", "Dovish tilt"] if c in piv.columns]
    piv = piv[order].loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(piv))))
    lim = np.nanmax(np.abs(piv.to_numpy())) or 1
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="mean daily P&L (%)")
    ax.set_title("Permutation x macro-regime performance matrix\n(tilt columns use no-event days only)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def plot_equity_curves(returns: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in returns.columns:
        ax.plot(returns.index, returns[col].cumsum(), label=col, lw=2.2 if col == "model_topk" else 1.2)
    ax.axhline(0, color="grey", lw=0.8); ax.legend(); ax.set_ylabel("Cumulative P&L (%)")
    ax.set_title("Walk-forward out-of-sample equity curves")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def make_all(daily: pd.DataFrame, events: pd.DataFrame, panel: pd.DataFrame, returns: pd.DataFrame,
             out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or config.FIGURES_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_sentiment_timeline(daily, events, out_dir / "macro_sentiment_timeline.png"),
        plot_surprise_matrix(events, out_dir / "macro_surprise_matrix.png"),
        plot_permutation_regime_heatmap(panel, out_dir / "permutation_regime_matrix.png"),
        plot_equity_curves(returns, out_dir / "walk_forward_equity.png"),
    ]
