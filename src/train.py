#!/usr/bin/env python3
"""
train.py
--------
OFFLINE training script -- NOT part of the graded run.sh pipeline.

Per the Hackathon Submission Guide (Section 5): "The model must be already
trained and committed -- we do not retrain. The test run only generates
features and predicts." This script is what was used to produce the
committed pickle/model.pkl from the sample data at data/, and is kept in the
repo purely for reproducibility / transparency (so a reviewer can retrain
from scratch on the same or new historical data if desired).

Usage:
    python src/train.py --data-dir ./data --model-out ./pickle/model.pkl
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils import ingest_all, validate_campaign_consistency, clean_for_modeling
from forecasting import weekly_aggregate, fit_segment_model


def build_models(clean_daily):
    models = {}

    # Channel-level models
    weekly_channel = weekly_aggregate(clean_daily, ["channel"])
    for channel, g in weekly_channel.groupby("channel"):
        models[("channel", channel)] = fit_segment_model(g)

    # Channel x campaign_type level models
    weekly_ct = weekly_aggregate(clean_daily, ["channel", "campaign_type"])
    for (channel, ctype), g in weekly_ct.groupby(["channel", "campaign_type"]):
        models[("channel_type", channel, ctype)] = fit_segment_model(g)

    # Channel x campaign_type x campaign_id level models
    weekly_camp = weekly_aggregate(clean_daily, ["channel", "campaign_type", "campaign_id", "campaign_name"])
    for (channel, ctype, cid, cname), g in weekly_camp.groupby(
            ["channel", "campaign_type", "campaign_id", "campaign_name"]):
        models[("campaign", channel, ctype, cid, cname)] = fit_segment_model(g)

    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model-out", default="./pickle/model.pkl")
    args = ap.parse_args()

    print(f"[train] Loading data from {args.data_dir} ...")
    raw = ingest_all(args.data_dir)
    _ = validate_campaign_consistency(raw)  # sanity check only, not persisted here
    clean = clean_for_modeling(raw)

    print("[train] Fitting per-segment trend / seasonality / elasticity / residual models ...")
    models = build_models(clean)
    print(f"[train] Fitted {len(models)} segment models "
          f"(channel-level, channel x campaign-type, and campaign-level).")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "models": models,
            "trained_on_date_range": [str(clean["date"].min().date()), str(clean["date"].max().date())],
            "n_rows_trained_on": int(len(clean)),
            "model_version": "aignition-v1.0-elasticity-montecarlo",
        }, f)
    print(f"[train] Model artifact written to {args.model_out}")


if __name__ == "__main__":
    main()
