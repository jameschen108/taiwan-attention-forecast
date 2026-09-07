"""P2: probability models, 4w labels, HGB, and trading backtest."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import simulate_weekly_rank_strategy
from src.backtest.metrics import compare_strategies
from src.forecast.calibrate import paired_brier_delta, reliability_bins
from src.forecast.config import load_forecast_config, resolve_path
from src.forecast.dataset import build_forecast_tables, join_xy
from src.forecast.evaluate import paired_delta_ic, summarize_ics, weekly_rank_ic
from src.forecast.features import feature_lists
from src.forecast.readiness import default_readiness, write_readiness
from src.forecast.splits import (
    as_of_list,
    filter_mature_fast,
    outer_year_plans,
    prediction_as_ofs_for_year,
    refit_schedule,
)
from src.forecast.train import (
    fit_hgb,
    fit_logistic,
    predict_hgb_frame,
    predict_proba_frame,
    select_c_inner,
)


def _load_predictions(cfg: dict, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(
        "predictions.parquet missing — run `python -m src.forecast.run` (P1) first"
    )


def run_p2_experiment(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    cfg = cfg or load_forecast_config()
    root = Path(cfg["_root"])
    out_dir = resolve_path(cfg, "output_dir")
    report_dir = resolve_path(cfg, "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    tables = build_forecast_tables(cfg)
    features, labels = tables["features"], tables["labels"]
    stock_daily = tables["stock_daily"]

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

    # --- M3/M4 logistic walk-forward (1w outperform) ---
    all_as_ofs = as_of_list(ja)
    prob_rows: list[dict] = []
    c_a_by_year: dict[int, float] = {}
    c_b_by_year: dict[int, float] = {}

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
            pa = predict_proba_frame(fit_a, sa)
            pb = predict_proba_frame(fit_b, sb)
            for i, row in sa.iterrows():
                prob_rows.append({
                    "as_of": as_of, "ticker": row["ticker"], "week": row["week"],
                    "year": year,
                    "y_outperform_1w": row.get("y_outperform_1w"),
                    "label_status": row.get("label_status"),
                    "prob_m3": float(pa.loc[i]),
                    "prob_m4": float(pb.loc[i]),
                })

    probs = pd.DataFrame(prob_rows)
    probs.to_parquet(out_dir / "probability_predictions.parquet", index=False)

    scored_prob = probs.dropna(subset=["y_outperform_1w"])
    scored_prob = scored_prob[scored_prob["label_status"] == "ok"]
    y = scored_prob["y_outperform_1w"].to_numpy()
    brier = paired_brier_delta(y, scored_prob["prob_m3"].to_numpy(), scored_prob["prob_m4"].to_numpy())
    rel_a = reliability_bins(y, scored_prob["prob_m3"].to_numpy())
    rel_b = reliability_bins(y, scored_prob["prob_m4"].to_numpy())
    rel_a.to_csv(report_dir / "calibration_m3.csv", index=False)
    rel_b.to_csv(report_dir / "calibration_m4.csv", index=False)

    # --- 4w HGB secondary eval (single refit per outer year) ---
    hgb_params = models_cfg.get("hgb", {
        "learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15,
        "min_samples_leaf": 100, "l2_regularization": 10,
    })
    hgb_rows: list[dict] = []
    for plan in outer_year_plans(years):
        year = plan.year
        fit_a4 = fit_hgb(ja4, "A", hgb_params, plan.train_cutoff, seed=seed,
                         sparsity_tiers=tiers, y_col="y_excess_4w")
        fit_b4 = fit_hgb(jb4, "B", hgb_params, plan.train_cutoff, seed=seed,
                         sparsity_tiers=tiers, y_col="y_excess_4w")
        test_a = filter_mature_fast(ja4, pd.Timestamp(f"{year + 1}-01-01"), y_col="y_excess_4w")
        test_a = test_a[test_a["as_of"] >= plan.year_start]
        test_a = test_a[test_a["as_of"] < pd.Timestamp(f"{year + 1}-01-01")]
        test_b = filter_mature_fast(jb4, pd.Timestamp(f"{year + 1}-01-01"), y_col="y_excess_4w")
        test_b = test_b[test_b["as_of"] >= plan.year_start]
        test_b = test_b[test_b["as_of"] < pd.Timestamp(f"{year + 1}-01-01")]
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

    # --- Trading backtest from P1 ridge predictions ---
    preds = _load_predictions(cfg, out_dir)
    lab1w = labels[labels["horizon"] == "1w"]
    cost_path = root / cfg.get("backtest", {}).get("cost_schedule", "config/trading_costs.csv")

    bt_a = simulate_weekly_rank_strategy(
        preds, stock_daily, lab1w, cost_path,
        pred_col="pred_m1", strategy_name="ridge_A",
        top_k=int(bt_cfg.get("top_k", 20)),
        max_weight=float(bt_cfg.get("max_weight_per_name", 0.05)),
        min_adv20=float(bt_cfg.get("min_adv20_twd", 5_000_000)),
        max_order_frac_adv=float(bt_cfg.get("max_order_fraction_adv20", 0.01)),
        initial_cash=float(bt_cfg.get("initial_cash_twd", 1_000_000)),
        slippage_stress=1.0,
    )
    bt_b = simulate_weekly_rank_strategy(
        preds, stock_daily, lab1w, cost_path,
        pred_col="pred_m2", strategy_name="ridge_B",
        top_k=int(bt_cfg.get("top_k", 20)),
        max_weight=float(bt_cfg.get("max_weight_per_name", 0.05)),
        min_adv20=float(bt_cfg.get("min_adv20_twd", 5_000_000)),
        max_order_frac_adv=float(bt_cfg.get("max_order_fraction_adv20", 0.01)),
        initial_cash=float(bt_cfg.get("initial_cash_twd", 1_000_000)),
        slippage_stress=1.0,
    )
    stress_a = simulate_weekly_rank_strategy(
        preds, stock_daily, lab1w, cost_path,
        pred_col="pred_m1", strategy_name="ridge_A_stress2x",
        top_k=int(bt_cfg.get("top_k", 20)),
        max_weight=float(bt_cfg.get("max_weight_per_name", 0.05)),
        min_adv20=float(bt_cfg.get("min_adv20_twd", 5_000_000)),
        max_order_frac_adv=float(bt_cfg.get("max_order_fraction_adv20", 0.01)),
        initial_cash=float(bt_cfg.get("initial_cash_twd", 1_000_000)),
        slippage_stress=float(bt_cfg.get("slippage_stress_multiplier", 2)),
    )

    bt_a["nav"].to_csv(report_dir / "nav_ridge_A.csv", index=False)
    bt_b["nav"].to_csv(report_dir / "nav_ridge_B.csv", index=False)
    bt_a["orders"].to_csv(report_dir / "orders_ridge_A.csv", index=False)
    bt_b["orders"].to_csv(report_dir / "orders_ridge_B.csv", index=False)
    stress_a["nav"].to_csv(report_dir / "nav_ridge_A_stress2x.csv", index=False)

    bt_summary = compare_strategies({
        "ridge_A": bt_a["metrics"],
        "ridge_B": bt_b["metrics"],
        "ridge_A_stress2x": stress_a["metrics"],
    })
    bt_summary.to_csv(report_dir / "backtest_summary.csv", index=False)

    summary = {
        "probability": {
            "brier_m3": brier["brier_a"],
            "brier_m4": brier["brier_b"],
            "delta_brier_b_minus_a": brier["delta_b_minus_a"],
            "m4_better_than_m3": brier["delta_b_minus_a"] > 0,
            "c_a_by_year": {str(k): v for k, v in c_a_by_year.items()},
            "c_b_by_year": {str(k): v for k, v in c_b_by_year.items()},
        },
        "hgb_4w_secondary": {
            "ic_a": summarize_ics(ic_hgb_a),
            "ic_b": summarize_ics(ic_hgb_b),
            "delta_ic": summarize_ics(delta_4w.rename(columns={"delta_ic": "ic"}), "ic"),
        },
        "backtest": bt_summary.to_dict(orient="records"),
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
        "",
        "## 4w HGB (secondary, not primary conclusion)",
        f"- Mean IC A: {summary['hgb_4w_secondary']['ic_a']['mean']:.4f}",
        f"- Mean IC B: {summary['hgb_4w_secondary']['ic_b']['mean']:.4f}",
        f"- ΔIC: {summary['hgb_4w_secondary']['delta_ic']['mean']:.4f}",
        "",
        "## Trading backtest (1w, ridge rankings, daily_bar_proxy)",
    ]
    for row in bt_summary.itertuples():
        lines.append(
            f"- {row.strategy}: return={row.total_return:.2%}, maxDD={row.max_drawdown:.2%}, "
            f"fees={row.total_fees:,.0f}"
        )
    lines += [
        "",
        "Execution: whole lots, ADV20 filter, costs from trading_costs.csv.",
        "Stress run uses 2x slippage on ridge_A.",
        "",
    ]
    (report_dir / "VERDICT_P2.md").write_text("\n".join(lines), encoding="utf-8")

    write_readiness(out_dir / "forecast_readiness.json", default_readiness(
        run_id=cfg["forecast"]["version"],
        data_ready=True,
        model_ready=True,
        eval_ready=True,
        notes=["P2 complete: probability, 4w HGB secondary, 1w backtest"],
    ))

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def main() -> None:
    run_p2_experiment()


if __name__ == "__main__":
    main()
