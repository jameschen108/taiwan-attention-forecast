"""P3 prospective prediction: immutable ledger, model snapshots, mature scoring."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.forecast.config import load_forecast_config, resolve_path
from src.forecast.dataset import build_forecast_tables, join_xy
from src.forecast.evaluate import weekly_rank_ic
from src.forecast.features import feature_lists
from src.forecast.readiness import default_readiness, write_readiness
from src.forecast.splits import as_of_list, filter_mature_fast, refit_schedule
from src.forecast.train import (
    constant_predictor,
    fit_ridge,
    predict_frame,
    select_alpha_inner,
)
from src.forecast.validate_config import assert_data_paths, validate_forecast_config

PREDICTION_KEYS = ["run_id", "as_of", "ticker", "horizon", "model_version"]


def _normalize_as_of(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).normalize()


def _ledger_has_as_of(existing: pd.DataFrame, as_of: pd.Timestamp) -> bool:
    if existing.empty:
        return False
    target = _normalize_as_of(as_of)
    return (pd.to_datetime(existing["as_of"]).dt.normalize() == target).any()


def _prospective_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("prospective") or {}


def predictions_path(cfg: dict[str, Any]) -> Path:
    rel = _prospective_cfg(cfg).get(
        "predictions_path", "data/forecast/prospective/predictions.parquet",
    )
    return Path(cfg["_root"]) / rel


def skips_path(cfg: dict[str, Any]) -> Path:
    rel = _prospective_cfg(cfg).get(
        "skips_path", "data/forecast/prospective/skips.csv",
    )
    return Path(cfg["_root"]) / rel


def scores_path(cfg: dict[str, Any]) -> Path:
    rel = _prospective_cfg(cfg).get(
        "scores_path", "data/forecast/prospective/scores.parquet",
    )
    return Path(cfg["_root"]) / rel


def model_store(cfg: dict[str, Any]) -> Path:
    rel = _prospective_cfg(cfg).get("model_store", "data/forecast/models")
    return Path(cfg["_root"]) / rel


def watchlist_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "report_dir") / "watchlists"


def load_or_build_features(cfg: dict[str, Any], rebuild: bool = False) -> pd.DataFrame:
    feat_path = resolve_path(cfg, "output_dir") / "features.parquet"
    if rebuild or not feat_path.exists():
        build_forecast_tables(cfg)
    return pd.read_parquet(feat_path)


def inference_slice(features: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    return features[features["as_of"] == as_of].copy()


def validate_inference_slice(slice_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    min_names = int(cfg["validation"].get("min_names_per_ic_week", 30))
    n = len(slice_df)
    tiers = cfg["features"].get("train_sparsity_tiers")
    if tiers and "sparsity_tier" in slice_df.columns:
        n_scorable = int(slice_df["sparsity_tier"].isin(tiers).sum())
    else:
        n_scorable = n
    return {
        "n_tickers": n,
        "n_scorable": n_scorable,
        "sufficient_universe": n_scorable >= min_names,
        "min_names_required": min_names,
    }


def should_skip(as_of: pd.Timestamp, quality: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    if not quality["sufficient_universe"]:
        return "insufficient_universe"
    path = predictions_path(cfg)
    if path.exists():
        existing = pd.read_parquet(path, columns=["as_of"])
        if _ledger_has_as_of(existing, as_of):
            return "duplicate_as_of"
    return None


def _model_tag(as_of: pd.Timestamp) -> str:
    return pd.Timestamp(as_of).strftime("%Y-%m-%d")


def save_model_bundle(
    as_of: pd.Timestamp,
    fit_a: Any,
    fit_b: Any,
    alpha_a: float,
    alpha_b: float,
    cfg: dict[str, Any],
) -> Path:
    store = model_store(cfg)
    store.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(as_of)
    manifest = {
        "as_of": str(pd.Timestamp(as_of).date()),
        "model_version": cfg["forecast"]["version"],
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "fit_cutoff": str(pd.Timestamp(as_of).date()),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": _prospective_cfg(cfg).get("baseline_model", "A"),
    }
    man_path = store / f"manifest_{tag}.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    joblib.dump(fit_a, store / f"ridge_a_{tag}.joblib")
    joblib.dump(fit_b, store / f"ridge_b_{tag}.joblib")
    return man_path


def load_model_bundle(as_of: pd.Timestamp, cfg: dict[str, Any]) -> tuple[Any, Any, dict]:
    store = model_store(cfg)
    tag = _model_tag(as_of)
    man_path = store / f"manifest_{tag}.json"
    if not man_path.exists():
        raise FileNotFoundError(f"no model bundle for {tag}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    fit_a = joblib.load(store / f"ridge_a_{tag}.joblib")
    fit_b = joblib.load(store / f"ridge_b_{tag}.joblib")
    return fit_a, fit_b, manifest


def needs_refit(as_of: pd.Timestamp, cfg: dict[str, Any], all_as_ofs: list[pd.Timestamp]) -> bool:
    tag = _model_tag(as_of)
    if (model_store(cfg) / f"manifest_{tag}.json").exists():
        return False
    scheduled = refit_schedule(
        all_as_ofs, every=int(cfg["validation"]["refit_every_prediction_weeks"]),
    )
    return as_of in scheduled


def fit_production_models(
    ja: pd.DataFrame,
    jb: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: dict[str, Any],
) -> tuple[Any, Any, float, float]:
    val = cfg["validation"]
    models_cfg = cfg["models"]
    tiers = cfg["features"].get("train_sparsity_tiers")
    seed = int(models_cfg.get("seed", val.get("seed", 42)))
    alphas = list(models_cfg["ridge_alpha"])

    pool_a = filter_mature_fast(ja, as_of)
    pool_b = filter_mature_fast(jb, as_of)
    if tiers:
        pool_a = pool_a[pool_a["sparsity_tier"].isin(tiers)]
        pool_b = pool_b[pool_b["sparsity_tier"].isin(tiers)]
    min_weeks = int(val["minimum_inner_training_weeks"])
    if pool_a["as_of"].nunique() < min_weeks // 2:
        raise ValueError(
            f"insufficient_training_data: {pool_a['as_of'].nunique()} mature weeks before {as_of.date()}"
        )

    alpha_a, _ = select_alpha_inner(
        pool_a, feature_lists()[0], alphas, seed=seed,
        n_blocks=int(val["inner_validation_blocks"]),
        block_weeks=int(val["inner_validation_weeks"]),
        min_train_weeks=min_weeks,
    )
    alpha_b, _ = select_alpha_inner(
        pool_b, feature_lists()[1], alphas, seed=seed,
        n_blocks=int(val["inner_validation_blocks"]),
        block_weeks=int(val["inner_validation_weeks"]),
        min_train_weeks=min_weeks,
    )
    fit_a = fit_ridge(ja, "A", alpha_a, as_of, seed=seed, sparsity_tiers=tiers)
    fit_b = fit_ridge(jb, "B", alpha_b, as_of, seed=seed, sparsity_tiers=tiers)
    return fit_a, fit_b, alpha_a, alpha_b


def append_predictions_immutable(rows: list[dict], cfg: dict[str, Any]) -> int:
    if not rows:
        return 0
    new_df = pd.DataFrame(rows)
    path = predictions_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        dup = new_df.merge(old[PREDICTION_KEYS], on=PREDICTION_KEYS, how="inner")
        if not dup.empty:
            raise ValueError(
                f"immutable prediction ledger: {len(dup)} duplicate keys for {PREDICTION_KEYS}"
            )
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_parquet(path, index=False)
    return len(new_df)


def record_skip(as_of: pd.Timestamp, reason: str, cfg: dict[str, Any], details: str = "") -> None:
    path = skips_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "as_of": pd.Timestamp(as_of),
        "reason": reason,
        "details": details,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": cfg["forecast"]["version"],
    }
    if path.exists():
        prev = pd.read_csv(path, parse_dates=["as_of"])
        if (prev["as_of"] == pd.Timestamp(as_of)).any():
            return
        df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)


def build_prediction_rows(
    as_of: pd.Timestamp,
    slice_a: pd.DataFrame,
    slice_b: pd.DataFrame,
    fit_a: Any,
    fit_b: Any,
    const_mu: float,
    cfg: dict[str, Any],
    quality: dict[str, Any],
) -> list[dict]:
    pa = predict_frame(fit_a, slice_a)
    pb = predict_frame(fit_b, slice_b)
    generated_at = datetime.now(timezone.utc).isoformat()
    model_version = cfg["forecast"]["version"]
    run_id = cfg["forecast"]["version"]
    baseline = _prospective_cfg(cfg).get("baseline_model", "A")
    rows: list[dict] = []
    for i, row in slice_a.iterrows():
        pred_a = float(pa.loc[i])
        pred_b = float(pb.loc[i])
        baseline_pred = pred_a if baseline == "A" else pred_b
        rows.append({
            "run_id": run_id,
            "model_version": model_version,
            "horizon": "1w",
            "generated_at": generated_at,
            "as_of": pd.Timestamp(as_of),
            "ticker": row["ticker"],
            "week": row["week"],
            "sparsity_tier": row.get("sparsity_tier"),
            "pred_m0": const_mu,
            "pred_m1": pred_a,
            "pred_m2": pred_b,
            "pred_baseline": baseline_pred,
            "baseline_model": baseline,
            "model_a": fit_a.model_name,
            "model_b": fit_b.model_name,
            "alpha_a": fit_a.alpha,
            "alpha_b": fit_b.alpha,
            "fit_cutoff": pd.Timestamp(as_of),
            "n_train_rows_a": fit_a.n_train_rows,
            "n_train_rows_b": fit_b.n_train_rows,
            "data_quality_ok": quality["sufficient_universe"],
            "availability_mode": cfg["data"]["availability_mode"],
            "prediction_kind": "prospective",
        })
    return rows


def write_watchlist(as_of: pd.Timestamp, rows: list[dict], cfg: dict[str, Any]) -> Path:
    out_dir = watchlist_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    baseline = _prospective_cfg(cfg).get("baseline_model", "A")
    pred_col = "pred_m1" if baseline == "A" else "pred_m2"
    tiers = cfg["features"].get("train_sparsity_tiers")
    if tiers:
        df = df[df["sparsity_tier"].isin(tiers)]
    ranked = df.sort_values(pred_col, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    tag = _model_tag(as_of)
    path = out_dir / f"watchlist_{tag}.csv"
    ranked[["rank", "ticker", "sparsity_tier", pred_col, "pred_m1", "pred_m2", "as_of"]].to_csv(
        path, index=False,
    )
    return path


def predict_for_as_of(
    as_of: pd.Timestamp,
    cfg: dict[str, Any] | None = None,
    *,
    rebuild_features: bool = False,
    force_refit: bool = False,
) -> dict[str, Any]:
    cfg = cfg or load_forecast_config()
    assert_data_paths(cfg)
    as_of = pd.Timestamp(as_of)

    skip_reason = None
    if predictions_path(cfg).exists():
        existing = pd.read_parquet(predictions_path(cfg), columns=["as_of"])
        if _ledger_has_as_of(existing, as_of):
            return {"as_of": str(as_of.date()), "status": "skipped", "reason": "duplicate_as_of"}

    features = load_or_build_features(cfg, rebuild=rebuild_features)
    if as_of not in set(pd.to_datetime(features["as_of"])):
        record_skip(as_of, "as_of_not_in_features", cfg)
        return {"as_of": str(as_of.date()), "status": "skipped", "reason": "as_of_not_in_features"}

    slice_feat = inference_slice(features, as_of)
    quality = validate_inference_slice(slice_feat, cfg)
    skip_reason = should_skip(as_of, quality, cfg)
    if skip_reason:
        record_skip(as_of, skip_reason, cfg, details=json.dumps(quality))
        return {"as_of": str(as_of.date()), "status": "skipped", "reason": skip_reason, "quality": quality}

    labels_path = resolve_path(cfg, "output_dir") / "labels.parquet"
    if not labels_path.exists() or rebuild_features:
        build_forecast_tables(cfg)
    ja = join_xy(features, pd.read_parquet(labels_path), "A")
    jb = join_xy(features, pd.read_parquet(labels_path), "B")

    all_as_ofs = as_of_list(ja)
    slice_a = ja[ja["as_of"] == as_of].copy()
    slice_b = jb[jb["as_of"] == as_of].copy()
    if slice_a.empty:
        record_skip(as_of, "empty_universe", cfg)
        return {"as_of": str(as_of.date()), "status": "skipped", "reason": "empty_universe"}

    tag = _model_tag(as_of)
    manifest_path = model_store(cfg) / f"manifest_{tag}.json"
    if force_refit or not manifest_path.exists():
        fit_a, fit_b, alpha_a, alpha_b = fit_production_models(ja, jb, as_of, cfg)
        save_model_bundle(as_of, fit_a, fit_b, alpha_a, alpha_b, cfg)
    else:
        fit_a, fit_b, _ = load_model_bundle(as_of, cfg)

    const_mu = constant_predictor(ja, as_of)
    rows = build_prediction_rows(as_of, slice_a, slice_b, fit_a, fit_b, const_mu, cfg, quality)
    n = append_predictions_immutable(rows, cfg)
    wl_path = write_watchlist(as_of, rows, cfg)
    return {
        "as_of": str(as_of.date()),
        "status": "predicted",
        "n_predictions": n,
        "quality": quality,
        "watchlist": str(wl_path),
        "baseline_model": _prospective_cfg(cfg).get("baseline_model", "A"),
    }


def score_mature_predictions(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Join immutable predictions with matured labels; never rewrite predictions."""
    cfg = cfg or load_forecast_config()
    pred_path = predictions_path(cfg)
    if not pred_path.exists():
        return pd.DataFrame()

    preds = pd.read_parquet(pred_path)
    labels_path = resolve_path(cfg, "output_dir") / "labels.parquet"
    if not labels_path.exists():
        build_forecast_tables(cfg)
    labels = pd.read_parquet(labels_path)
    labels = labels[labels["horizon"] == "1w"][
        ["ticker", "week", "as_of", "y_excess_1w", "y_outperform_1w",
         "label_status", "label_end_at", "label_available_at"]
    ]

    merged = preds.merge(labels, on=["ticker", "week", "as_of"], how="left", suffixes=("", "_lab"))
    now = pd.Timestamp.utcnow().tz_localize(None)
    mature = merged[
        merged["label_end_at"].notna()
        & merged["label_available_at"].notna()
        & (merged["label_available_at"] <= now)
        & (merged["label_status"] == "ok")
    ].copy()
    mature["scored_at"] = datetime.now(timezone.utc).isoformat()

    out_path = scores_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        keys = ["as_of", "ticker", "horizon", "model_version"]
        mature = mature.merge(prev[keys], on=keys, how="left", indicator=True)
        mature = mature[mature["_merge"] == "left_only"].drop(columns=["_merge"])
        if mature.empty:
            return prev
        out = pd.concat([prev, mature], ignore_index=True)
    else:
        out = mature
    out.to_parquet(out_path, index=False)

    # Weekly IC on newly scorable prospective weeks
    tiers = cfg["features"].get("train_sparsity_tiers")
    scored = out.dropna(subset=["y_excess_1w"])
    if tiers:
        scored = scored[scored["sparsity_tier"].isin(tiers)]
    min_names = int(cfg["validation"].get("min_names_per_ic_week", 30))
    ic = weekly_rank_ic(scored.rename(columns={"pred_baseline": "pred"}), min_names=min_names)
    report_dir = resolve_path(cfg, "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    if not ic.empty:
        ic.to_csv(report_dir / "prospective_weekly_ic.csv", index=False)
    return out


def operational_report(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize last N forecast weeks: predicted vs skipped (P3 acceptance)."""
    cfg = cfg or load_forecast_config()
    n_weeks = int(_prospective_cfg(cfg).get("min_operational_weeks", 12))
    features = load_or_build_features(cfg)
    all_as_ofs = as_of_list(features)
    window = all_as_ofs[-n_weeks:] if len(all_as_ofs) >= n_weeks else all_as_ofs

    pred_as_ofs: set[pd.Timestamp] = set()
    if predictions_path(cfg).exists():
        raw = pd.to_datetime(pd.read_parquet(
            predictions_path(cfg), columns=["as_of"],
        )["as_of"].unique())
        pred_as_ofs = {_normalize_as_of(x) for x in raw}

    skips = pd.DataFrame()
    if skips_path(cfg).exists():
        skips = pd.read_csv(skips_path(cfg), parse_dates=["as_of"])

    rows = []
    for as_of in window:
        norm = _normalize_as_of(as_of)
        if norm in pred_as_ofs:
            status = "predicted"
            reason = ""
        elif not skips.empty and (skips["as_of"] == as_of).any():
            status = "skipped"
            reason = str(skips.loc[skips["as_of"] == as_of, "reason"].iloc[0])
        else:
            status = "missing"
            reason = "no_record"
        rows.append({"as_of": as_of, "status": status, "reason": reason})

    df = pd.DataFrame(rows)
    n_pred = int((df["status"] == "predicted").sum())
    n_skip = int((df["status"] == "skipped").sum())
    n_missing = int((df["status"] == "missing").sum())
    complete = n_missing == 0

    summary = {
        "window_weeks": n_weeks,
        "as_ofs_in_window": len(window),
        "predicted": n_pred,
        "skipped": n_skip,
        "missing": n_missing,
        "all_accounted": complete,
        "operational_ready": complete and len(window) >= n_weeks,
    }

    report_dir = resolve_path(cfg, "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "prospective_operational.csv", index=False)
    lines = [
        "# P3 Prospective Operational Status",
        "",
        f"- window: last {n_weeks} feature as_ofs",
        f"- predicted: {n_pred}",
        f"- skipped (traceable): {n_skip}",
        f"- missing: {n_missing}",
        f"- all_accounted: **{complete}**",
        f"- operational_ready (≥{n_weeks} weeks): **{summary['operational_ready']}**",
        "",
        "Skipped weeks must have a row in `data/forecast/prospective/skips.csv`.",
        "Predictions are append-only in `data/forecast/prospective/predictions.parquet`.",
        "",
    ]
    (report_dir / "PROSPECTIVE_STATUS.md").write_text("\n".join(lines), encoding="utf-8")

    readiness_path = resolve_path(cfg, "output_dir") / "forecast_readiness.json"
    notes = [
        "P3 prospective ledger active",
        f"operational window {n_pred}+{n_skip}/{len(window)} accounted",
    ]
    blockers = []
    if n_missing:
        blockers.append(f"prospective: {n_missing} weeks without prediction or skip record")
    write_readiness(readiness_path, default_readiness(
        run_id=cfg["forecast"]["version"],
        data_ready=True,
        model_ready=True,
        eval_ready=True,
        trading_ready=False,
        blockers=blockers,
        notes=notes,
    ))
    return summary


def run_backfill(weeks: int, cfg: dict[str, Any] | None = None, **kwargs: Any) -> list[dict]:
    cfg = cfg or load_forecast_config()
    features = load_or_build_features(cfg, rebuild=kwargs.get("rebuild_features", False))
    all_as_ofs = as_of_list(features)[-weeks:]
    results = []
    for as_of in all_as_ofs:
        results.append(predict_for_as_of(as_of, cfg, **kwargs))
    score_mature_predictions(cfg)
    operational_report(cfg)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="P3 prospective forecast entry")
    parser.add_argument("--as-of", type=str, help="Monday as_of (YYYY-MM-DD)")
    parser.add_argument("--backfill-weeks", type=int, help="Replay last N as_ofs (operational test)")
    parser.add_argument("--score-mature", action="store_true", help="Score matured labels only")
    parser.add_argument("--operational-report", action="store_true", help="Write 12-week ops report")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_forecast_config()
    t0 = time.time()

    if args.score_mature:
        out = score_mature_predictions(cfg)
        print(json.dumps({"n_scored_rows": len(out)}, indent=2))
        return

    if args.operational_report:
        print(json.dumps(operational_report(cfg), indent=2, default=str))
        return

    if args.backfill_weeks:
        results = run_backfill(
            args.backfill_weeks, cfg,
            rebuild_features=args.rebuild_features,
            force_refit=args.force_refit,
        )
        print(json.dumps({"backfill": results, "elapsed_sec": round(time.time() - t0, 1)},
                         indent=2, default=str))
        return

    features = load_or_build_features(cfg, rebuild=args.rebuild_features)
    as_ofs = as_of_list(features)
    if not as_ofs:
        sys.exit("no as_of in features")
    as_of = pd.Timestamp(args.as_of) if args.as_of else as_ofs[-1]
    result = predict_for_as_of(
        as_of, cfg,
        rebuild_features=args.rebuild_features,
        force_refit=args.force_refit,
    )
    if result.get("status") == "predicted":
        score_mature_predictions(cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
