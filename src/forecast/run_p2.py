"""P2: probability models, 4w labels, HGB, and trading backtest."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import (
    execution_diagnostics,
    simulate_benchmark_buy_hold,
    simulate_benchmark_weekly,
    simulate_weekly_rank_strategy,
)
from src.backtest.metrics import compare_strategies
from src.forecast.calibrate import (
    apply_calibrator,
    fit_calibrator_from_probs,
    paired_brier_delta,
    reliability_bins,
)
from src.forecast.config import load_forecast_config, resolve_path
from src.forecast.dataset import build_forecast_tables, join_xy
from src.forecast.evaluate import (
    block_bootstrap_mean,
    classification_metrics,
    non_overlapping_subsample,
    paired_delta_ic,
    summarize_ics,
    weekly_rank_ic,
)
from src.forecast.features import feature_lists
from src.forecast.readiness import assess_trading_readiness, default_readiness, write_readiness
from src.forecast.splits import (
    as_of_list,
    filter_mature_fast,
    inner_forward_blocks,
    outer_year_plans,
    prediction_as_ofs_for_year,
    refit_schedule,
)
from src.forecast.train import (
    _logistic_pipeline,
    _xy_class,
    fit_hgb,
    fit_logistic,
    predict_hgb_frame,
    predict_proba_frame,
    select_c_inner,
)
from src.forecast.validate_config import assert_data_paths


def _load_predictions(cfg: dict, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(
        "predictions.parquet missing — run `python -m src.forecast.run` (P1) first"
    )


def _fit_year_calibrator(
    pool: pd.DataFrame,
    cols: list[str],
    c: float,
    seed: int,
    val_cfg: dict,
    tiers: list[str] | None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Collect inner-fold OOF probabilities for Platt scaling."""
    blocks = inner_forward_blocks(
        pool,
        n_blocks=int(val_cfg["inner_validation_blocks"]),
        block_weeks=int(val_cfg["inner_validation_weeks"]),
        min_train_weeks=int(val_cfg["minimum_inner_training_weeks"]),
    )
    oof_prob, oof_y = [], []
    for _, valid_start, valid_end in blocks:
        tr = filter_mature_fast(pool, fit_cutoff=valid_start)
        if tiers:
            tr = tr[tr["sparsity_tier"].isin(tiers)]
        va = pool[
            (pool["as_of"] >= valid_start)
            & (pool["as_of"] < valid_end)
            & (pool["y_outperform_1w"].notna())
            & (pool["label_status"] == "ok")
        ]
        if len(tr) < 100 or len(va) < 50:
            continue
        Xtr, ytr, _ = _xy_class(tr, cols)
        pipe = _logistic_pipeline(len(cols), c, seed)
        pipe.fit(Xtr, ytr)
        Xva, yva, _ = _xy_class(va, cols)
        oof_prob.extend(pipe.predict_proba(Xva)[:, 1])
        oof_y.extend(yva)
    cal = fit_calibrator_from_probs(np.array(oof_y), np.array(oof_prob))
    return cal, np.array(oof_y), np.array(oof_prob)


def run_p2_experiment(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    cfg = cfg or load_forecast_config()
    assert_data_paths(cfg)
    root = Path(cfg["_root"])
    out_dir = resolve_path(cfg, "output_dir")
    report_dir = resolve_path(cfg, "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    tables = build_forecast_tables(cfg)
    features, labels = tables["features"], tables["labels"]
    stock_daily = tables["stock_daily"]
    bench_daily = tables.get("bench_daily", stock_daily)

    ja = join_xy(features, labels, "A", horizon="1w")
    jb = join_xy(features, labels, "B", horizon="1w")
    ja4 = join_xy(features, labels, "A", horizon="4w")
    jb4 = join_xy(features, labels, "B", horizon="4w")

    val = cfg["validation"]
    models_cfg = cfg["models"]
    bt_cfg = cfg.get("backtest", {})
    tiers = cfg["features"].get("train_sparsity_tiers")
    seed = int(models_cfg.get("seed", val.get("seed", 42)))
    c_values = list(models_cfg.get("logistic_c", [0.01, 0.1, 1, 10]))
    years = list(val["outer_test_years"])
    min_names = int(val["min_names_per_ic_week"])
    every = int(val["refit_every_prediction_weeks"])
    a_cols, b_cols, _ = feature_lists()
    generated_at = datetime.now(timezone.utc).isoformat()

    # --- M3/M4 logistic walk-forward with optional calibration ---
    all_as_ofs = as_of_list(ja)
    prob_rows: list[dict] = []
    c_a_by_year: dict[int, float] = {}
    c_b_by_year: dict[int, float] = {}
    cal_a_by_year: dict[int, Any] = {}
    cal_b_by_year: dict[int, Any] = {}

    for plan in outer_year_plans(years):
        year = plan.year
        pool_a = filter_mature_fast(ja, plan.train_cutoff)
        pool_b = filter_mature_fast(jb, plan.train_cutoff)
        if tiers:
            pool_a = pool_a[pool_a["sparsity_tier"].isin(tiers)]
            pool_b = pool_b[pool_b["sparsity_tier"].isin(tiers)]

        c_a, _ = select_c_inner(pool_a, a_cols, c_values, seed=seed,
                                n_blocks=int(val["inner_validation_blocks"]),
                                block_weeks=int(val["inner_validation_weeks"]),
                                min_train_weeks=int(val["minimum_inner_training_weeks"]))
        c_b, _ = select_c_inner(pool_b, b_cols, c_values, seed=seed,
                                n_blocks=int(val["inner_validation_blocks"]),
                                block_weeks=int(val["inner_validation_weeks"]),
                                min_train_weeks=int(val["minimum_inner_training_weeks"]))
        c_a_by_year[year] = c_a
        c_b_by_year[year] = c_b
        cal_a_by_year[year], _, _ = _fit_year_calibrator(
            pool_a, a_cols, c_a, seed, val, tiers)
        cal_b_by_year[year], _, _ = _fit_year_calibrator(
            pool_b, b_cols, c_b, seed, val, tiers)

        year_as_ofs = prediction_as_ofs_for_year(all_as_ofs, year)
        refits = refit_schedule(year_as_ofs, every=every)
        fit_a = fit_b = None

        for as_of in year_as_ofs:
            if fit_a is None or as_of in refits:
                fit_a = fit_logistic(ja, "A", c_a, as_of, seed=seed, sparsity_tiers=tiers)
                fit_b = fit_logistic(jb, "B", c_b, as_of, seed=seed, sparsity_tiers=tiers)
            sa = ja[ja["as_of"] == as_of]
            sb = jb[jb["as_of"] == as_of]
            if sa.empty:
                continue
            pa_raw = predict_proba_frame(fit_a, sa)
            pb_raw = predict_proba_frame(fit_b, sb)
            pa = pa_raw.copy()
            pb = pb_raw.copy()
            cal_a = cal_a_by_year.get(year)
            cal_b = cal_b_by_year.get(year)
            if cal_a is not None:
                pa = pd.Series(apply_calibrator(cal_a, pa.to_numpy()), index=pa.index)
            if cal_b is not None:
                pb = pd.Series(apply_calibrator(cal_b, pb.to_numpy()), index=pb.index)
            for i, row in sa.iterrows():
                prob_rows.append({
                    "as_of": as_of, "ticker": row["ticker"], "week": row["week"],
                    "year": year, "generated_at": generated_at,
                    "y_outperform_1w": row.get("y_outperform_1w"),
                    "label_status": row.get("label_status"),
                    "prob_m3_raw": float(pa_raw.loc[i]),
                    "prob_m4_raw": float(pb_raw.loc[i]),
                    "prob_m3": float(pa.loc[i]),
                    "prob_m4": float(pb.loc[i]),
                    "calibrated": cal_a is not None and cal_b is not None,
                })

    probs = pd.DataFrame(prob_rows)
    probs.to_parquet(out_dir / "probability_predictions.parquet", index=False)

    scored_prob = probs.dropna(subset=["y_outperform_1w"])
    scored_prob = scored_prob[scored_prob["label_status"] == "ok"]
    y = scored_prob["y_outperform_1w"].to_numpy()
    brier = paired_brier_delta(
        y, scored_prob["prob_m3"].to_numpy(), scored_prob["prob_m4"].to_numpy())
    cls_m3 = classification_metrics(y, scored_prob["prob_m3"].to_numpy())
    cls_m4 = classification_metrics(y, scored_prob["prob_m4"].to_numpy())
    rel_a = reliability_bins(y, scored_prob["prob_m3"].to_numpy())
    rel_b = reliability_bins(y, scored_prob["prob_m4"].to_numpy())
    rel_a.to_csv(report_dir / "calibration_m3.csv", index=False)
    rel_b.to_csv(report_dir / "calibration_m4.csv", index=False)

    # --- 4w HGB walk-forward (refit every 4 weeks like M1/M2) ---
    hgb_params = dict(models_cfg.get("hgb", {}))
    if "early_stopping" not in hgb_params:
        hgb_params["early_stopping"] = False
    hgb_rows: list[dict] = []
    for plan in outer_year_plans(years):
        year = plan.year
        year_as_ofs = prediction_as_ofs_for_year(as_of_list(ja4), year)
        refits = refit_schedule(year_as_ofs, every=every)
        fit_a4 = fit_b4 = None
        for as_of in year_as_ofs:
            if fit_a4 is None or as_of in refits:
                fit_a4 = fit_hgb(ja4, "A", hgb_params, as_of, seed=seed,
                                 sparsity_tiers=tiers, y_col="y_excess_4w")
                fit_b4 = fit_hgb(jb4, "B", hgb_params, as_of, seed=seed,
                                 sparsity_tiers=tiers, y_col="y_excess_4w")
            test_a = ja4[(ja4["as_of"] == as_of)]
            test_b = jb4[(jb4["as_of"] == as_of)]
            if tiers:
                test_a = test_a[test_a["sparsity_tier"].isin(tiers)]
                test_b = test_b[test_b["sparsity_tier"].isin(tiers)]
            if test_a.empty:
                continue
            ta = test_a[["as_of", "ticker", "y_excess_4w", "label_status"]].copy()
            ta["pred_hgb_a"] = predict_hgb_frame(fit_a4, test_a).values
            tb = test_b[["as_of", "ticker"]].copy()
            tb["pred_hgb_b"] = predict_hgb_frame(fit_b4, test_b).values
            merged = ta.merge(tb, on=["as_of", "ticker"], how="inner")
            merged["year"] = year
            hgb_rows.extend(merged.to_dict("records"))

    hgb_pred = pd.DataFrame(hgb_rows)
    hgb_pred.to_parquet(out_dir / "hgb_4w_predictions.parquet", index=False)
    hgb_scored = hgb_pred.dropna(subset=["y_excess_4w"])
    hgb_scored = hgb_scored[hgb_scored["label_status"] == "ok"]
    ic_hgb_a = weekly_rank_ic(
        hgb_scored.rename(columns={"pred_hgb_a": "pred", "y_excess_4w": "y_excess_1w"}),
        min_names=min_names,
    )
    ic_hgb_b = weekly_rank_ic(
        hgb_scored.rename(columns={"pred_hgb_b": "pred", "y_excess_4w": "y_excess_1w"}),
        min_names=min_names,
    )
    delta_4w = paired_delta_ic(ic_hgb_a, ic_hgb_b, min_names=min_names)
    delta_4w_nonoverlap = non_overlapping_subsample(
        delta_4w.rename(columns={"delta_ic": "ic"}), value_col="ic", step=4)
    hgb_4w_bootstrap = block_bootstrap_mean(
        delta_4w.loc[delta_4w["sufficient"], "delta_ic"],
        block_weeks=4,
        reps=int(val["bootstrap_repetitions"]),
        seed=seed,
    )

    # --- Trading backtest ---
    preds = _load_predictions(cfg, out_dir)
    lab1w = labels[labels["horizon"] == "1w"]
    cost_path = root / bt_cfg.get("cost_schedule", "config/trading_costs.csv")
    top_k = int(bt_cfg.get("top_k", 20))
    bt_kwargs = dict(
        top_k=top_k,
        max_weight=float(bt_cfg.get("max_weight_per_name", 0.05)),
        min_adv20=float(bt_cfg.get("min_adv20_twd", 5_000_000)),
        max_order_frac_adv=float(bt_cfg.get("max_order_fraction_adv20", 0.01)),
        sparsity_tiers=tiers,
    )

    def _run_bt(name: str, pred_col: str, cash: float, stress: float = 1.0) -> dict:
        return simulate_weekly_rank_strategy(
            preds, stock_daily, lab1w, cost_path,
            pred_col=pred_col, strategy_name=name,
            initial_cash=cash, slippage_stress=stress, **bt_kwargs,
        )

    initial = float(bt_cfg.get("initial_cash_twd", 1_000_000))
    sens_cash = float(bt_cfg.get("sensitivity_initial_cash_twd", initial * 10))
    bt_a = _run_bt("ridge_A", "pred_m1", initial)
    bt_b = _run_bt("ridge_B", "pred_m2", initial)
    stress_a = _run_bt("ridge_A_stress2x", "pred_m1", initial,
                       float(bt_cfg.get("slippage_stress_multiplier", 2)))
    bt_a_sens = _run_bt("ridge_A_sensitivity_10M", "pred_m1", sens_cash)

    bt_0050_w = simulate_benchmark_weekly(
        bench_daily, lab1w, cost_path, initial_cash=initial)
    bt_0050_bh = simulate_benchmark_buy_hold(
        bench_daily, lab1w, cost_path, initial_cash=initial)

    for tag, res in [
        ("ridge_A", bt_a), ("ridge_B", bt_b), ("ridge_A_stress2x", stress_a),
        ("ridge_A_sensitivity_10M", bt_a_sens),
        ("benchmark_0050_weekly", bt_0050_w), ("benchmark_0050_buy_hold", bt_0050_bh),
    ]:
        if not res["nav"].empty:
            res["nav"].to_csv(report_dir / f"nav_{tag}.csv", index=False)
        if not res["orders"].empty:
            res["orders"].to_csv(report_dir / f"orders_{tag}.csv", index=False)

    exec_diag = execution_diagnostics(bt_a["orders"], bt_a["nav"], top_k=top_k)
    exec_diag.to_csv(report_dir / "execution_diagnostics.csv", index=False)

    bt_summary = compare_strategies({
        "ridge_A": bt_a["metrics"],
        "ridge_B": bt_b["metrics"],
        "ridge_A_stress2x": stress_a["metrics"],
        "ridge_A_sensitivity_10M": bt_a_sens["metrics"],
        "benchmark_0050_weekly": bt_0050_w["metrics"],
        "benchmark_0050_buy_hold": bt_0050_bh["metrics"],
    })
    bt_summary.to_csv(report_dir / "backtest_summary.csv", index=False)

    trading_ready, trading_blockers = assess_trading_readiness(
        {
            "ridge_A": bt_a["metrics"],
            "ridge_B": bt_b["metrics"],
            "ridge_A_stress2x": stress_a["metrics"],
        },
        max_drawdown_limit=bt_cfg.get("max_drawdown_limit"),
    )

    summary = {
        "probability": {
            "brier_m3": brier["brier_a"],
            "brier_m4": brier["brier_b"],
            "delta_brier_b_minus_a": brier["delta_b_minus_a"],
            "m4_better_than_m3": brier["delta_b_minus_a"] > 0,
            "classification_m3": cls_m3,
            "classification_m4": cls_m4,
            "calibrated": bool(scored_prob["calibrated"].all()) if "calibrated" in scored_prob else False,
            "c_a_by_year": {str(k): v for k, v in c_a_by_year.items()},
            "c_b_by_year": {str(k): v for k, v in c_b_by_year.items()},
        },
        "hgb_4w_secondary": {
            "ic_a": summarize_ics(ic_hgb_a),
            "ic_b": summarize_ics(ic_hgb_b),
            "delta_ic": summarize_ics(delta_4w.rename(columns={"delta_ic": "ic"}), "ic"),
            "delta_ic_nonoverlap_every_4w": summarize_ics(delta_4w_nonoverlap, "ic"),
            "delta_ic_block_bootstrap_4w": hgb_4w_bootstrap,
        },
        "backtest": bt_summary.to_dict(orient="records"),
        "execution_diagnostics": {
            "mean_fill_rate": float(exec_diag["fill_rate"].mean()) if not exec_diag.empty else np.nan,
            "mean_cash_fraction": float(exec_diag["cash_fraction"].mean()) if not exec_diag.empty else np.nan,
        },
        "trading_ready": trading_ready,
        "trading_blockers": trading_blockers,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (report_dir / "p2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# P2 Evaluation",
        "",
        "## Probability (M3/M4, 1w outperform vs 0050)",
        f"- Brier M3 (price): {brier['brier_a']:.4f}",
        f"- Brier M4 (price+PTT): {brier['brier_b']:.4f}",
        f"- ΔBrier (M3−M4, positive => M4 better): {brier['delta_b_minus_a']:.4f}",
        f"- Calibrated: {summary['probability']['calibrated']}",
        "",
        "## 4w HGB (secondary, non-overlapping subsample every 4w)",
        f"- Mean IC A (all weeks): {summary['hgb_4w_secondary']['ic_a']['mean']:.4f}",
        f"- Mean IC B (all weeks): {summary['hgb_4w_secondary']['ic_b']['mean']:.4f}",
        f"- ΔIC non-overlap: {summary['hgb_4w_secondary']['delta_ic_nonoverlap_every_4w']['mean']:.4f}",
        "",
        "## Trading backtest (1w, ridge rankings, daily_bar_proxy)",
    ]
    for row in bt_summary.itertuples():
        lines.append(
            f"- {row.strategy}: return={row.total_return:.2%}, maxDD={row.max_drawdown:.2%}, "
            f"fees={row.total_fees:,.0f}"
        )
    if not exec_diag.empty:
        lines += [
            "",
            f"Execution: mean fill rate {exec_diag['fill_rate'].mean():.1%}, "
            f"mean cash fraction {exec_diag['cash_fraction'].mean():.1%}.",
        ]
    lines += [
        "",
        f"Trading ready (spec gate): **{trading_ready}**",
        "",
    ]
    (report_dir / "VERDICT_P2.md").write_text("\n".join(lines), encoding="utf-8")

    write_readiness(out_dir / "forecast_readiness.json", default_readiness(
        run_id=cfg["forecast"]["version"],
        data_ready=True,
        model_ready=True,
        eval_ready=True,
        trading_ready=trading_ready,
        blockers=trading_blockers,
        notes=["P2 complete: probability, 4w HGB secondary, 1w backtest with 0050 benchmarks"],
    ))

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def main() -> None:
    run_p2_experiment()


if __name__ == "__main__":
    main()
