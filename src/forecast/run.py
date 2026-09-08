"""P0/P1 forecast entrypoint: build tables, walk-forward Ridge A/B, write eval."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.forecast.config import load_forecast_config, resolve_path
from src.forecast.dataset import build_forecast_tables, join_xy
from src.forecast.evaluate import (
    block_bootstrap_mean,
    coverage_metrics,
    paired_delta_ic,
    registered_gain_gate,
    regression_error_metrics,
    stability_breakdown,
    summarize_ics,
    top_decile_metrics,
    weekly_rank_ic,
    year_breakdown,
)
from src.forecast.features import feature_lists
from src.forecast.readiness import default_readiness, write_readiness
from src.forecast.splits import (
    as_of_list,
    filter_mature_fast,
    outer_year_plans,
    prediction_as_ofs_for_year,
    refit_schedule,
)
from src.forecast.train import constant_predictor, fit_ridge, predict_frame, select_alpha_inner
from src.forecast.validate_config import assert_data_paths, validate_forecast_config


def run_p1_experiment(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    cfg = cfg or load_forecast_config()
    assert_data_paths(cfg)
    cfg_warnings = validate_forecast_config(cfg)
    out_dir = resolve_path(cfg, "output_dir")
    report_dir = resolve_path(cfg, "report_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    tables = build_forecast_tables(cfg)
    features, labels = tables["features"], tables["labels"]
    ja = join_xy(features, labels, "A")
    jb = join_xy(features, labels, "B")
    ja.to_parquet(out_dir / "xy_A.parquet", index=False)
    jb.to_parquet(out_dir / "xy_B.parquet", index=False)

    val = cfg["validation"]
    models_cfg = cfg["models"]
    tiers = cfg["features"].get("train_sparsity_tiers")
    seed = int(models_cfg.get("seed", val.get("seed", 42)))
    alphas = list(models_cfg["ridge_alpha"])
    years = list(val["outer_test_years"])
    min_names = int(val["min_names_per_ic_week"])
    every = int(val["refit_every_prediction_weeks"])
    a_cols, b_cols, _ = feature_lists()
    generated_at = datetime.now(timezone.utc).isoformat()
    model_version = cfg["forecast"]["version"]

    all_as_ofs = as_of_list(ja)
    pred_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    alpha_a_by_year: dict[int, float] = {}
    alpha_b_by_year: dict[int, float] = {}

    for plan in outer_year_plans(years):
        year = plan.year
        pool_a = filter_mature_fast(ja, plan.train_cutoff)
        pool_b = filter_mature_fast(jb, plan.train_cutoff)
        if tiers:
            pool_a = pool_a[pool_a["sparsity_tier"].isin(tiers)]
            pool_b = pool_b[pool_b["sparsity_tier"].isin(tiers)]

        alpha_a, rec_a = select_alpha_inner(
            pool_a, a_cols, alphas, seed=seed,
            n_blocks=int(val["inner_validation_blocks"]),
            block_weeks=int(val["inner_validation_weeks"]),
            min_train_weeks=int(val["minimum_inner_training_weeks"]),
        )
        alpha_b, rec_b = select_alpha_inner(
            pool_b, b_cols, alphas, seed=seed,
            n_blocks=int(val["inner_validation_blocks"]),
            block_weeks=int(val["inner_validation_weeks"]),
            min_train_weeks=int(val["minimum_inner_training_weeks"]),
        )
        alpha_a_by_year[year] = alpha_a
        alpha_b_by_year[year] = alpha_b
        selection_rows.extend([
            rec_a.assign(year=year, feature_set="A"),
            rec_b.assign(year=year, feature_set="B"),
        ])

        year_as_ofs = prediction_as_ofs_for_year(all_as_ofs, year)
        refits = refit_schedule(year_as_ofs, every=every)
        fit_a = fit_b = None
        const_mu = 0.0

        for as_of in year_as_ofs:
            if fit_a is None or as_of in refits:
                fit_a = fit_ridge(ja, "A", alpha_a, as_of, seed=seed, sparsity_tiers=tiers)
                fit_b = fit_ridge(jb, "B", alpha_b, as_of, seed=seed, sparsity_tiers=tiers)
                const_mu = constant_predictor(ja, as_of)

            slice_a = ja[ja["as_of"] == as_of].copy()
            slice_b = jb[jb["as_of"] == as_of].copy()
            if slice_a.empty:
                continue

            pa = predict_frame(fit_a, slice_a)
            pb = predict_frame(fit_b, slice_b)
            for i, row in slice_a.iterrows():
                pred_rows.append({
                    "run_id": cfg["forecast"]["version"],
                    "model_version": model_version,
                    "horizon": "1w",
                    "generated_at": generated_at,
                    "as_of": as_of,
                    "ticker": row["ticker"],
                    "week": row["week"],
                    "year": year,
                    "y_excess_1w": row.get("y_excess_1w"),
                    "label_status": row.get("label_status"),
                    "sparsity_tier": row.get("sparsity_tier"),
                    "pred_m0": const_mu,
                    "pred_m1": float(pa.loc[i]),
                    "pred_m2": float(pb.loc[i]),
                    "model_a": fit_a.model_name,
                    "model_b": fit_b.model_name,
                    "alpha_a": alpha_a,
                    "alpha_b": alpha_b,
                    "fit_cutoff": as_of,
                    "n_train_rows_a": fit_a.n_train_rows,
                    "n_train_rows_b": fit_b.n_train_rows,
                })

        print(
            f"  year {year}: alpha_A={alpha_a} alpha_B={alpha_b} "
            f"as_ofs={len(year_as_ofs)}",
            flush=True,
        )

    preds = pd.DataFrame(pred_rows)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    if selection_rows:
        pd.concat(selection_rows, ignore_index=True).to_csv(
            out_dir / "hyperparam_selection.csv", index=False,
        )

    scored = preds.dropna(subset=["y_excess_1w"]).copy()
    scored = scored[scored["label_status"] == "ok"]
    if tiers:
        scored = scored[scored["sparsity_tier"].isin(tiers)]

    ic0 = weekly_rank_ic(scored.rename(columns={"pred_m0": "pred"}), min_names=min_names)
    ic1 = weekly_rank_ic(scored.rename(columns={"pred_m1": "pred"}), min_names=min_names)
    ic2 = weekly_rank_ic(scored.rename(columns={"pred_m2": "pred"}), min_names=min_names)
    delta = paired_delta_ic(ic1, ic2, min_names=min_names)

    bootstrap_sensitivity = {}
    for bw in val.get("bootstrap_sensitivity_block_weeks", []):
        bootstrap_sensitivity[str(bw)] = block_bootstrap_mean(
            delta.loc[delta["sufficient"], "delta_ic"],
            block_weeks=int(bw),
            reps=int(val["bootstrap_repetitions"]),
            seed=seed,
        )

    weekly = delta[["as_of", "ic_a", "ic_b", "delta_ic", "n", "sufficient"]].copy()
    weekly["ic_m0"] = ic0.set_index("as_of").reindex(weekly["as_of"])["ic"].values
    weekly.to_csv(report_dir / "evaluation_weekly.csv", index=False)

    summary = {
        "m0": summarize_ics(ic0),
        "m1_A": summarize_ics(ic1),
        "m2_B": summarize_ics(ic2),
        "delta_B_minus_A": summarize_ics(
            delta.rename(columns={"delta_ic": "ic"}), "ic",
        ),
        "delta_bootstrap": block_bootstrap_mean(
            delta.loc[delta["sufficient"], "delta_ic"],
            block_weeks=int(val["bootstrap_block_weeks"]),
            reps=int(val["bootstrap_repetitions"]),
            seed=seed,
        ),
        "delta_bootstrap_sensitivity": bootstrap_sensitivity,
        "regression_m1": regression_error_metrics(scored, "pred_m1"),
        "regression_m2": regression_error_metrics(scored, "pred_m2"),
        "top_decile_m1": top_decile_metrics(scored, "pred_m1"),
        "top_decile_m2": top_decile_metrics(scored, "pred_m2"),
        "coverage": coverage_metrics(preds, scored, pred_col="pred_m1"),
        "config_warnings": cfg_warnings,
        "alpha_a_by_year": {str(k): v for k, v in alpha_a_by_year.items()},
        "alpha_b_by_year": {str(k): v for k, v in alpha_b_by_year.items()},
        "n_predictions": int(len(preds)),
        "n_scored": int(len(scored)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    by_year = year_breakdown(delta, "delta_ic")
    by_year.to_csv(report_dir / "evaluation_by_year.csv", index=False)
    stability_breakdown(scored, "pred_m1").to_csv(
        report_dir / "stability_by_sparsity_m1.csv", index=False)
    stability_breakdown(scored, "pred_m2").to_csv(
        report_dir / "stability_by_sparsity_m2.csv", index=False)

    dboot = summary["delta_bootstrap"]
    year_pos = int((by_year["mean"] > 0).sum()) if not by_year.empty else 0
    gate = registered_gain_gate(
        summary["m2_B"],
        summary["delta_B_minus_A"],
        dboot,
        year_pos,
        len(by_year),
    )
    gain = gate["gain"]
    summary["pre_registered_gain"] = gain
    summary["year_positive_delta_count"] = year_pos
    summary["registered_gain_gate"] = gate

    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# P1 Forecast Evaluation (Ridge A vs B)",
        "",
        f"- run: `{cfg['forecast']['version']}`",
        f"- availability_mode: `{cfg['data']['availability_mode']}`",
        f"- label_definition: `{cfg['forecast'].get('label_definition')}`",
        f"- universe: `{cfg['universe']['development_mode']}`",
        "",
        "## Mean weekly Rank IC",
        "- M0 constant: Rank IC undefined (constant cross-section ranks); kept as level baseline only",
        f"- M1 Ridge A (price): {summary['m1_A']['mean']:.4f} (n={summary['m1_A']['n_weeks']})",
        f"- M2 Ridge B (price+PTT): {summary['m2_B']['mean']:.4f} (n={summary['m2_B']['n_weeks']})",
        f"- ΔIC (B−A): {summary['delta_B_minus_A']['mean']:.4f} "
        f"bootstrap95=({dboot['ci_low']:.4f}, {dboot['ci_high']:.4f})",
        f"- years with ΔIC>0: {year_pos} / {len(by_year)}",
        f"- pre-registered historical gain: **{gain}**",
        "",
        "If gain is false, keep A as the prediction baseline; PTT has no sufficient "
        "incremental evidence under this registered rule.",
        "",
    ]
    (report_dir / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")

    write_readiness(out_dir / "forecast_readiness.json", default_readiness(
        run_id=cfg["forecast"]["version"],
        data_ready=True,
        model_ready=True,
        eval_ready=True,
        notes=[
            "P0/P1 complete under retrospective_proxy + adjusted_price_proxy",
            "Trading not enabled (P2)",
        ],
    ))

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def main() -> None:
    run_p1_experiment()


if __name__ == "__main__":
    main()
