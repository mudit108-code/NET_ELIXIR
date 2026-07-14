"""
utils.py
--------
Shared utilities for the AIgnition Probabilistic Revenue Forecasting utility.

Responsibilities:
  1. Ingest raw per-channel CSVs (Google Ads, Microsoft/Bing Ads, Meta Ads) of
     whatever row-count / date-range is dropped into data/, and normalise them
     into one canonical long-format table.
  2. Derive a `campaign_type` for every row (funnel/format grouping) even for
     sources such as Meta Ads that do not natively expose one.
  3. Run campaign-consistency validation checks and produce a structured
     validation report (used by both the CLI pipeline and the Streamlit demo).

No network calls are made anywhere in this module (fully offline, as required
by the submission guide, Section 6/8).
"""

from __future__ import annotations
import os
import glob
import re
import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "date", "channel", "campaign_id", "campaign_name", "campaign_type",
    "spend", "revenue", "conversions", "clicks", "impressions", "daily_budget",
]

# ---------------------------------------------------------------------------
# Campaign-type inference for sources that don't expose one natively (Meta)
# ---------------------------------------------------------------------------
_META_TYPE_RULES = [
    (r"prospecting", "Prospecting"),
    (r"remarketing|retarget", "Remarketing"),
    (r"brand", "Brand"),
    (r"dpa", "Catalog/DPA"),
    (r"generic", "Generic"),
]


def infer_meta_campaign_type(campaign_name: str) -> str:
    name = str(campaign_name).lower()
    for pattern, label in _META_TYPE_RULES:
        if re.search(pattern, name):
            return label
    return "Other"


def _find_file(data_dir: str, patterns: list[str]) -> str | None:
    """Locate a file in data_dir matching any of the given glob patterns
    (case-insensitive). Returns the first match or None."""
    for pattern in patterns:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return matches[0]
        # case-insensitive fallback
        for f in glob.glob(os.path.join(data_dir, "*")):
            if re.fullmatch(pattern.replace("*", ".*"), os.path.basename(f), re.IGNORECASE):
                return f
    return None


def load_bing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "CampaignId": "campaign_id", "TimePeriod": "date", "Revenue": "revenue",
        "Spend": "spend", "Clicks": "clicks", "Impressions": "impressions",
        "Conversions": "conversions", "CampaignType": "campaign_type",
        "DailyBudget": "daily_budget", "CampaignName": "campaign_name",
    })
    df["channel"] = "Microsoft Ads"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name", "campaign_type",
               "spend", "revenue", "conversions", "clicks", "impressions", "daily_budget"]]


def load_google(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "campaign_id": "campaign_id", "segments_date": "date",
        "metrics_clicks": "clicks", "metrics_conversions": "conversions",
        "metrics_cost_micros": "spend_micros", "metrics_impressions": "impressions",
        "metrics_conversions_value": "revenue",
        "campaign_advertising_channel_type": "campaign_type",
        "campaign_budget_amount": "daily_budget", "campaign_name": "campaign_name",
    })
    df["spend"] = df["spend_micros"].astype(float) / 1_000_000.0
    df["channel"] = "Google Ads"
    df["campaign_type"] = df["campaign_type"].str.title().str.replace("_", " ")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name", "campaign_type",
               "spend", "revenue", "conversions", "clicks", "impressions", "daily_budget"]]


def load_meta(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(columns={
        "campaign_id": "campaign_id", "date_start": "date", "spend": "spend",
        "clicks": "clicks", "impressions": "impressions",
        "conversion": "revenue",  # Meta's "conversion" field is a conversion *value* (currency), not a count
        "daily_budget": "daily_budget", "campaign_name": "campaign_name",
    })
    df["channel"] = "Meta Ads"
    df["campaign_type"] = df["campaign_name"].apply(infer_meta_campaign_type)
    # Meta feed has no separate conversion-count column in this export; we treat
    # the "conversion" values field as revenue and approximate a conversion count
    # using CPA-neutral fallback of NaN (not used by the model).
    df["conversions"] = np.nan
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "channel", "campaign_id", "campaign_name", "campaign_type",
               "spend", "revenue", "conversions", "clicks", "impressions", "daily_budget"]]


LOADERS = {
    "bing": (["*bing*campaign*stat*.csv", "*bing*.csv", "*microsoft*.csv"], load_bing),
    "google": (["*google*campaign*stat*.csv", "*google*.csv"], load_google),
    "meta": (["*meta*campaign*stat*.csv", "*meta*.csv", "*facebook*.csv"], load_meta),
}


def ingest_all(data_dir: str) -> pd.DataFrame:
    """Discover and load whatever channel CSVs exist in data_dir (by filename
    pattern, not hardcoded to specific files), union them into one canonical
    long table. Missing a channel simply omits it, rather than failing."""
    frames = []
    found = {}
    for key, (patterns, loader) in LOADERS.items():
        path = _find_file(data_dir, patterns)
        if path:
            frames.append(loader(path))
            found[key] = path
    if not frames:
        raise FileNotFoundError(
            f"No recognised channel CSVs (google/meta/bing) found in {data_dir}"
        )
    df = pd.concat(frames, ignore_index=True)
    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0.0)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0.0)
    df = df.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Campaign-consistency validation
# ---------------------------------------------------------------------------

def validate_campaign_consistency(df: pd.DataFrame) -> dict:
    """Runs a battery of consistency checks required before forecasting.
    Returns a JSON-serialisable report; never raises -- issues are surfaced
    as warnings so the pipeline degrades gracefully on messy real-world data."""
    report = {"n_rows": int(len(df)), "n_campaigns": int(df["campaign_id"].nunique()),
               "channels": sorted(df["channel"].unique().tolist()), "issues": []}

    # 1. A campaign_id should map to exactly one campaign_name / campaign_type / channel
    grp = df.groupby("campaign_id")
    multi_name = grp["campaign_name"].nunique()
    multi_type = grp["campaign_type"].nunique()
    bad_ids = multi_name[multi_name > 1].index.tolist() + multi_type[multi_type > 1].index.tolist()
    if bad_ids:
        report["issues"].append({
            "check": "campaign_id_maps_to_single_name_and_type",
            "severity": "warning",
            "n_affected": len(set(bad_ids)),
            "detail": "Some campaign_ids are associated with more than one name/type across rows; "
                      "the most frequent value is used for forecasting.",
        })

    # 2. Duplicate (campaign_id, date) rows
    dupes = df.duplicated(subset=["campaign_id", "date"]).sum()
    if dupes:
        report["issues"].append({
            "check": "duplicate_campaign_date_rows", "severity": "warning",
            "n_affected": int(dupes),
            "detail": "Duplicate rows for the same campaign/date were found and summed.",
        })

    # 3. Negative spend or revenue
    neg_spend = int((df["spend"] < 0).sum())
    neg_rev = int((df["revenue"] < 0).sum())
    if neg_spend or neg_rev:
        report["issues"].append({
            "check": "negative_values", "severity": "warning",
            "n_affected": neg_spend + neg_rev,
            "detail": f"{neg_spend} rows with negative spend, {neg_rev} rows with negative revenue "
                      "were clipped to zero.",
        })

    # 4. Spend with zero impressions/clicks (possible tracking gap)
    ghost = int(((df["spend"] > 0) & (df["impressions"] == 0)).sum())
    if ghost:
        report["issues"].append({
            "check": "spend_without_impressions", "severity": "info",
            "n_affected": ghost,
            "detail": "Rows with recorded spend but zero impressions (delivery lag or tracking gap).",
        })

    # 5. Date gaps per campaign (missing days within the campaign's active window)
    gap_campaigns = 0
    for cid, g in df.groupby("campaign_id"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        if len(full_range) > len(g["date"].unique()):
            gap_campaigns += 1
    if gap_campaigns:
        report["issues"].append({
            "check": "date_gaps_within_campaign_window", "severity": "info",
            "n_affected": gap_campaigns,
            "detail": "Campaigns with missing daily rows inside their active date range "
                      "(treated as zero-spend/zero-revenue days).",
        })

    # 6. Missing daily_budget
    missing_budget = int(df["daily_budget"].isna().sum())
    if missing_budget:
        report["issues"].append({
            "check": "missing_daily_budget", "severity": "info",
            "n_affected": missing_budget,
            "detail": "Rows missing a declared daily budget; excluded from budget-elasticity fitting "
                      "but still included in historical actuals.",
        })

    report["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    report["is_valid_for_forecasting"] = True  # informational-only checks above; nothing blocks the run
    return report


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the fixes implied by the validation report: clip negatives,
    collapse duplicate rows, resolve campaign_id -> single name/type mapping."""
    df = df.copy()
    df["spend"] = df["spend"].clip(lower=0)
    df["revenue"] = df["revenue"].clip(lower=0)

    # resolve campaign_id -> most frequent name/type
    name_map = df.groupby("campaign_id")["campaign_name"].agg(lambda s: s.mode().iat[0])
    type_map = df.groupby("campaign_id")["campaign_type"].agg(lambda s: s.mode().iat[0])
    df["campaign_name"] = df["campaign_id"].map(name_map)
    df["campaign_type"] = df["campaign_id"].map(type_map)

    # collapse duplicate (campaign_id, date) rows by summing metrics
    agg = {
        "spend": "sum", "revenue": "sum", "conversions": "sum",
        "clicks": "sum", "impressions": "sum",
        "channel": "first", "campaign_name": "first", "campaign_type": "first",
        "daily_budget": "mean",
    }
    df = df.groupby(["campaign_id", "date"], as_index=False).agg(agg)
    return df.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)
