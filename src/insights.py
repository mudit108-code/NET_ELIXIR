"""
insights.py
-----------
"AI-assisted" causal-summary and anomaly-interpretation layer.
  
  As per an explicit project requirement, this
  submission avoids ALL external/paid APIs so the prototype has zero
  network dependency and zero per-call cost. Instead, this module
  implements a deterministic "reasoning layer": it runs statistical
  attribution (which factor moved the number and by how much), z-score
  anomaly detection, and a template-based natural-language generator to
  produce business-readable causal summaries.

  This keeps the required *shape* of the deliverable (causal explanation +
  anomaly interpretation + operational risk flags, in plain English) while
  satisfying the "no external APIs" constraint. The architecture is written
  so a real LLM call (e.g. Claude via the Anthropic API) can be swapped in
  behind the same `generate_causal_summary()` function signature with no
  other code changes -- see the "Future LLM Integration" note in the docs.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _fmt_money(x: float) -> str:
    return f"₹{x:,.0f}"


def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}%"


def zscore_anomalies(weekly: pd.DataFrame, value_col: str = "revenue", z_thresh: float = 1.8) -> pd.DataFrame:
    """Flag weeks whose value deviates > z_thresh standard deviations from
    the trailing 8-week rolling mean -- a lightweight, explainable anomaly
    detector (no model training required)."""
    w = weekly.sort_values("week").copy()
    w["roll_mean"] = w[value_col].rolling(8, min_periods=3).mean()
    w["roll_std"] = w[value_col].rolling(8, min_periods=3).std().replace(0, np.nan)
    w["z"] = (w[value_col] - w["roll_mean"]) / w["roll_std"]
    w["is_anomaly"] = w["z"].abs() > z_thresh
    return w


def segment_causal_summary(segment_name: str, weekly: pd.DataFrame, model: dict, forecast: dict) -> str:
    """Generate a plain-English causal narrative for one channel / campaign-type
    segment, combining: trend direction, spend elasticity, seasonality,
    and any detected anomalies. Purely rule-based / statistical -- no LLM call."""
    lines = []
    trend_pct = model.get("trend_weekly_pct", 0.0)
    elasticity = model.get("elasticity", 0.6)
    roas = model.get("avg_roas", 0.0)

    # 1. Trend narrative
    if abs(trend_pct) < 0.01:
        lines.append(f"{segment_name} revenue has been roughly flat week-over-week over the observed history.")
    elif trend_pct > 0:
        lines.append(f"{segment_name} revenue has been growing at an average of {_fmt_pct(trend_pct)} per week, "
                      f"driven by the underlying trend independent of any budget change.")
    else:
        lines.append(f"{segment_name} revenue has been declining at an average of {_fmt_pct(trend_pct)} per week, "
                      f"which the forecast projects forward unless budget or targeting changes.")

    # 2. Elasticity / diminishing returns narrative
    if elasticity >= 0.9:
        lines.append(f"Spend-response elasticity is high ({elasticity:.2f}): revenue scales nearly "
                      f"proportionally with spend in this segment, so incremental budget is currently efficient.")
    elif elasticity >= 0.5:
        lines.append(f"Spend-response elasticity is moderate ({elasticity:.2f}): expect diminishing but still "
                      f"positive returns from additional budget.")
    else:
        lines.append(f"Spend-response elasticity is low ({elasticity:.2f}), indicating this segment is close to "
                      f"saturation -- additional budget is likely to erode blended ROAS faster than it adds revenue.")

    # 3. ROAS context
    lines.append(f"Trailing average ROAS for this segment is {roas:.2f}x.")

    # 4. Anomalies
    anomalies = zscore_anomalies(weekly)
    recent_anomalies = anomalies[anomalies["is_anomaly"]].tail(3)
    if len(recent_anomalies):
        wk = recent_anomalies.iloc[-1]
        direction = "spike" if wk["z"] > 0 else "drop"
        lines.append(f"An unusual revenue {direction} was detected in the week of {pd.Timestamp(wk['week']).date()} "
                      f"({_fmt_money(wk['revenue'])}, z-score {wk['z']:.1f}) -- worth a manual sanity check "
                      f"against tracking or promo activity before trusting the forecast baseline.")

    # 5. Forward risk flag
    if model.get("low_data_flag"):
        lines.append("Caution: this segment has limited history, so the forecast band is wider and less reliable "
                      "than segments with a full seasonal cycle of data.")
    if forecast["budget_multiplier"] < 0.9 and forecast["roas_p50"] > roas * 1.05:
        lines.append("Reducing budget in this segment is projected to improve blended ROAS, at the cost of "
                      "absolute revenue -- a trade-off worth flagging to the account lead.")
    elif forecast["budget_multiplier"] > 1.1 and forecast["roas_p50"] < roas * 0.95:
        lines.append("Increasing budget here is projected to add revenue but dilute ROAS -- confirm this segment "
                      "still clears the account's minimum ROAS threshold before scaling.")

    return " ".join(lines)


def portfolio_summary(segment_results: list[dict]) -> str:
    """Roll up all segment causal summaries into a top-line executive summary."""
    if not segment_results:
        return "No segments were forecastable from the supplied data."
    total_p50 = sum(s["forecast"]["revenue_p50"] for s in segment_results)
    total_spend = sum(s["forecast"]["spend_scenario_total"] for s in segment_results)
    blended_roas = total_p50 / total_spend if total_spend else 0.0
    best = max(segment_results, key=lambda s: s["model"].get("trend_weekly_pct", 0))
    worst = min(segment_results, key=lambda s: s["model"].get("trend_weekly_pct", 0))

    parts = [
        f"Across {len(segment_results)} channel/campaign-type segments, the median forecast is "
        f"{_fmt_money(total_p50)} in revenue against {_fmt_money(total_spend)} of planned spend "
        f"(blended ROAS {blended_roas:.2f}x).",
        f"{best['segment']} shows the strongest organic growth trend "
        f"({_fmt_pct(best['model'].get('trend_weekly_pct', 0))}/week).",
    ]
    if worst["model"].get("trend_weekly_pct", 0) < 0:
        parts.append(f"{worst['segment']} is the primary drag, declining at "
                      f"{_fmt_pct(worst['model'].get('trend_weekly_pct', 0))}/week and warrants review.")
    return " ".join(parts)
