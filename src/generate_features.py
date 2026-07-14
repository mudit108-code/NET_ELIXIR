#!/usr/bin/env python3
"""
generate_features.py
---------------------
Pipeline stage 1 (called by run.sh).

  1. Ingests whatever channel CSVs are present in --data-dir (Google/Meta/Bing,
     detected by filename pattern -- see src/utils.py).
  2. Runs campaign-consistency validation and writes a JSON report.
  3. Cleans the data (dedup, clip negatives, resolve id->name/type mapping).
  4. Builds weekly-aggregated feature tables at three grains:
       - channel level
       - channel x campaign_type level
       - channel x campaign_type x campaign_id level
  5. Serialises everything needed by predict.py into a single pickle file
     (kept as one artifact per the "combine data updated together" guidance;
     avoids adding a parquet dependency purely for an intermediate hand-off).

Usage:
    python src/generate_features.py --data-dir ./data --out features.pkl
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import ingest_all, validate_campaign_consistency, clean_for_modeling
from forecasting import weekly_aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[generate_features] Ingesting channel CSVs from {args.data_dir} ...")
    raw = ingest_all(args.data_dir)
    print(f"[generate_features] Ingested {len(raw):,} rows across "
          f"{raw['channel'].nunique()} channel(s), {raw['campaign_id'].nunique():,} campaigns, "
          f"{raw['date'].min().date()} -> {raw['date'].max().date()}")

    print("[generate_features] Validating campaign consistency ...")
    report = validate_campaign_consistency(raw)
    report_path = os.path.join(os.path.dirname(args.out) or ".", "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[generate_features] Validation report written to {report_path} "
          f"({len(report['issues'])} issue type(s) flagged)")

    clean = clean_for_modeling(raw)

    weekly_channel = weekly_aggregate(clean, ["channel"])
    weekly_channel_type = weekly_aggregate(clean, ["channel", "campaign_type"])
    weekly_campaign = weekly_aggregate(clean, ["channel", "campaign_type", "campaign_id", "campaign_name"])

    payload = {
        "clean_daily": clean,
        "weekly_channel": weekly_channel,
        "weekly_channel_type": weekly_channel_type,
        "weekly_campaign": weekly_campaign,
        "validation_report": report,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"[generate_features] Feature bundle written to {args.out}")


if __name__ == "__main__":
    main()
