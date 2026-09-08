"""P4 incremental source experiments: Ridge A vs A+news vs A+revenue vs B vs E."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from pathlib import Path

from src.forecast.config import load_forecast_config, resolve_path
from src.forecast.dataset import build_forecast_tables, join_xy
from src.forecast.evaluate import (
    block_bootstrap_mean,
    paired_delta_ic,
    registered_gain_gate,
    summarize_ics,
    weekly_rank_ic,
    year_breakdown,
)
from src.forecast.features import columns_for_set
from src.forecast.folds import build_folds
from src.forecast.sources.build import build_supplemental_features
from src.forecast.splits import (
    as_of_list,
    filter_mature_fast,
    outer_year_plans,
    prediction_as_ofs_for_year,
    refit_schedule,
)
from src.forecast.train import fit_ridge, predict_frame, select_alpha_inner
from src.forecast.universe import build_universe_snapshots
from src.forecast.validate_config import assert_data_paths

P4_COMPARISONS: list[tuple[str, str, bool]] = [
    ("C", "A", True),
    ("D", "A", True),
    ("B", "A", True),
    ("E", "D", False),
]


def _walk_forward(
    xy_by_set: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    *,
    sets: list[str],
) -> tuple[pd.DataFrame, dict]:
    val = cfg["validation"]
    models_cfg = cfg["models"]
    tiers = cfg["features"].get("train_sparsity_tiers")
    seed = int(models_cfg.get("seed", val.get("seed", 42)))
    alphas = list(models_cfg["ridge_alpha"])
    years = list(val["outer_test_years"])
    every = int(val["refit_every_prediction_weeks"])

    ja = xy_by_set["A"]
    all_as_ofs = as_of_list(ja)
    pred_rows: list[dict] = []
    alpha_by_year: dict[str, dict[int, float]] = {s: {} for s in sets}

    for plan in outer_year_plans(years):
        year = plan.year
        fits: dict[str, Any] = {}
        alphas_y: dict[str, float] = {}

        for fs in sets:
            pool = filter_mature_fast(xy_by_set[fs], plan.train_cutoff)
            if tiers:
                pool = pool[pool["sparsity_tier"].isin(tiers)]
            alpha, _ = select_alpha_inner(
                pool, columns_for_set(fs), alphas, seed=seed,
                n_blocks=int(val["inner_validation_blocks"]),
                block_weeks=int(val["inner_validation_weeks"]),
                min_train_weeks=int(val["minimum_inner_training_weeks"]),
            )
            alphas_y[fs] = alpha
            alpha_by_year[fs][year] = alpha

        year_as_ofs = prediction_as_ofs_for_year(all_as_ofs, year)
        refits = refit_schedule(year_as_ofs, every=every)

        for as_of in year_as_ofs:
            for fs in sets:
                if fs not in fits or as_of in refits:
                    fits[fs] = fit_ridge(
                        xy_by_set[fs], fs, alphas_y[fs], as_of,
                        seed=seed, sparsity_tiers=tiers,
                    )
            if ja[ja["as_of"] == as_of].empty:
                continue
            for fs in sets:
                sl = xy_by_set[fs][xy_by_set[fs]["as_of"] == as_of]
                pred = predict_frame(fits[fs], sl)
                for i, row in sl.iterrows():
                    pred_rows.append({
                        "as_of": as_of,
                        "ticker": row["ticker"],
                        "feature_set": fs,
                        "pred": float(pred.loc[i]),
                        "y_excess_1w": row.get("y_excess_1w"),
                        "label_status": row.get("label_status"),
                        "sparsity_tier": row.get("sparsity_tier"),
                        "year": year,
                        "alpha": alphas_y[fs],
                        "model": fits[fs].model_name,
                    })

    return pd.DataFrame(pred_rows), {"alpha_by_year": alpha_by_year}


def _ic_for_set(preds: pd.DataFrame, feature_set: str, min_names: int) -> pd.DataFrame:
    sub = preds[preds["feature_set"] == feature_set].rename(columns={"pred": "pred"})
    return weekly_rank_ic(sub, min_names=min_names)


def _compare_pair(
    preds: pd.DataFrame,
    candidate: str,
    baseline: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    val = cfg["validation"]
    min_names = int(val["min_names_per_ic_week"])
    seed = int(cfg["models"].get("seed", val.get("seed", 42)))

    ic_base = _ic_for_set(preds, baseline, min_names)
    ic_cand = _ic_for_set(preds, candidate, min_names)
    delta = paired_delta_ic(ic_base, ic_cand, min_names=min_names)
    by_year = year_breakdown(delta, "delta_ic")
    year_pos = int((by_year["mean"] > 0).sum()) if not by_year.empty else 0

    boot = block_bootstrap_mean(
        delta.loc[delta["sufficient"], "delta_ic"],
        block_weeks=int(val["bootstrap_block_weeks"]),
        reps=int(val["bootstrap_repetitions"]),
        seed=seed,
    )
    gate = registered_gain_gate(
        summarize_ics(ic_cand),
        summarize_ics(delta.rename(columns={"delta_ic": "ic"}), "ic"),
        boot,
        year_pos,
        len(by_year),
    )

    return {
        "candidate": candidate,
        "baseline": baseline,
        "comparison": f"{candidate}_vs_{baseline}",
        "candidate_ic": summarize_ics(ic_cand),
        "baseline_ic": summarize_ics(ic_base),
        "delta": summarize_ics(delta.rename(columns={"delta_ic": "ic"}), "ic"),
        "bootstrap": boot,
        "by_year": by_year,
        "gate": gate,
        "delta_weekly": delta,
    }


def _news_coverage_stats(features_p4: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    if "log1p_news_count_1w" not in features_p4.columns:
        return {"news_week_share": np.nan, "news_ticker2330_share": np.nan}
    nz = int((features_p4["log1p_news_count_1w"] > 0).sum())
    total = int(len(features_p4))
    share = nz / total if total else np.nan
    nm_path = Path(cfg["_root"]) / "data" / "forecast" / "supplemental" / "news_matched.parquet"
    t2330_share = np.nan
    if nm_path.exists():
        nm = pd.read_parquet(nm_path, columns=["ticker"])
        n_rows = len(nm)
        t2330_share = float((nm["ticker"] == "2330").sum() / n_rows) if n_rows else np.nan
    return {
        "news_week_share": share,
        "news_weeks_with_any": nz,
        "news_panel_rows": total,
        "news_ticker2330_share": t2330_share,
    }


def _write_verdict(
    report_dir: Path,
    comparisons: list[dict[str, Any]],
    source_meta: list[dict],
    news_stats: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    def _fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None and pd.notna(v) else "n/a"

    lines = [
        "# P4 Incremental Source Evaluation",
        "",
        "Same-sample Ridge comparisons. Registered gate: FORECAST_SPEC §17.2.",
        "",
        f"- A (price): mean IC {_fmt(summary['A']['mean'])}",
        "",
        "## Comparisons",
        "",
    ]

    for comp in comparisons:
        cand, base = comp["candidate"], comp["baseline"]
        pre = comp["pre_registered"]
        d = comp["delta"]["mean"]
        boot = comp["bootstrap"]
        gate = comp["gate"]
        tag = "pre-registered" if pre else "exploratory"
        lines.append(
            f"### {cand} vs {base} ({tag})"
        )
        lines.append(
            f"- mean ΔIC: {_fmt(d)} bootstrap95=({boot['ci_low']:.4f}, {boot['ci_high']:.4f})"
        )
        lines.append(
            f"- candidate IC: {_fmt(comp['candidate_ic']['mean'])} "
            f"(n={comp['candidate_ic']['n_weeks']})"
        )
        lines.append(
            f"- years with ΔIC>0: {gate['years_positive']} / {gate['n_years']}"
        )
        lines.append(
            f"- gate: candidate_ic={gate['gate_candidate_ic']} "
            f"mean_delta={gate['gate_mean_delta']} "
            f"boot_low={gate['gate_boot_low']} "
            f"years={gate['gate_years']} "
            f"→ **gain={gate['gain']}**"
        )
        if pre and not gate["gain"]:
            if cand == "D":
                lines.append(
                    "- **Verdict:** keep A as prediction baseline; month revenue has no "
                    "sufficient incremental evidence under the registered rule."
                )
            elif cand == "B":
                lines.append(
                    "- **Verdict:** keep A as prediction baseline; PTT has no sufficient "
                    "incremental evidence under the registered rule."
                )
            elif cand == "C":
                lines.append(
                    "- **Verdict:** keep A as prediction baseline; news has no sufficient "
                    "incremental evidence under the registered rule."
                )
        elif pre and gate["gain"]:
            lines.append(
                f"- **Verdict:** {cand} passes the registered historical gain gate vs A."
            )
        else:
            lines.append(
                "- **Verdict:** exploratory comparison only; does not change the registered baseline."
            )
        lines.append("")

    lines.extend([
        "## Source availability",
    ])
    for meta in source_meta:
        lines.append(
            f"- {meta.get('source')}: available={meta.get('available')} "
            f"n_rows={meta.get('n_rows', 0)}"
        )

    if pd.notna(news_stats.get("news_week_share")):
        pct = news_stats["news_week_share"] * 100
        t2330 = news_stats.get("news_ticker2330_share", np.nan)
        lines.extend([
            "",
            "## News coverage caveat",
            f"- Only **{pct:.1f}%** of ticker-weeks have any 1w headline news "
            f"({news_stats.get('news_weeks_with_any', 'n/a')} / "
            f"{news_stats.get('news_panel_rows', 'n/a')}).",
        ])
        if pd.notna(t2330):
            lines.append(
                f"- Ticker 2330 accounts for **{t2330 * 100:.1f}%** of matched news rows; "
                "C vs A null result may reflect sparse/biased coverage, not a clean null on news."
            )
    lines.extend([
        "",
        "Missing sources still run with `missing_*` flags; interpret C/D gains only when coverage is adequate.",
        "",
    ])
    (report_dir / "VERDICT_P4.md").write_text("\n".join(lines), encoding="utf-8")


def run_p4_experiment(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.time()
    cfg = cfg or load_forecast_config()
    assert_data_paths(cfg)
    out_dir = resolve_path(cfg, "output_dir")
    report_dir = resolve_path(cfg, "report_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    tables = build_forecast_tables(cfg)
    features, labels = tables["features"], tables["labels"]
    panel = pd.read_parquet(resolve_path(cfg, "panel_path"))
    panel["week"] = pd.to_datetime(panel["week"])
    panel, source_meta = build_supplemental_features(panel, cfg)

    supp_cols = list(dict.fromkeys(columns_for_set("C") + columns_for_set("D")))
    supp_cols = [c for c in supp_cols if c in panel.columns and c not in features.columns]
    if supp_cols:
        features = features.merge(panel[["ticker", "week"] + supp_cols], on=["ticker", "week"], how="left")
    features.to_parquet(out_dir / "features_p4.parquet", index=False)
    news_stats = _news_coverage_stats(features, cfg)

    build_universe_snapshots(as_of_list(features), cfg)
    feature_sets = sorted({fs for fs, _, _ in P4_COMPARISONS} | {"A"})
    xy_map = {fs: join_xy(features, labels, fs) for fs in feature_sets}
    build_folds(xy_map["A"], cfg)

    preds, aux = _walk_forward(xy_map, cfg, sets=feature_sets)
    preds.to_parquet(out_dir / "p4_predictions.parquet", index=False)

    ic_a = _ic_for_set(preds, "A", int(cfg["validation"]["min_names_per_ic_week"]))
    summary: dict[str, Any] = {
        "sources": source_meta,
        "alpha_by_year": aux["alpha_by_year"],
        "A": summarize_ics(ic_a),
        "news_coverage": news_stats,
    }

    comparison_results: list[dict[str, Any]] = []
    registry_rows: list[dict] = []
    by_year_rows: list[dict] = []
    generated_at = datetime.now(timezone.utc).isoformat()

    for candidate, baseline, pre_registered in P4_COMPARISONS:
        comp = _compare_pair(preds, candidate, baseline, cfg)
        comp["pre_registered"] = pre_registered
        comparison_results.append(comp)

        comp["delta_weekly"].to_csv(
            report_dir / f"p4_delta_{candidate}_vs_{baseline}.csv", index=False,
        )
        summary[comp["comparison"]] = {
            "candidate_ic": comp["candidate_ic"],
            "baseline_ic": comp["baseline_ic"],
            "delta": comp["delta"],
            "bootstrap": comp["bootstrap"],
            "gate": comp["gate"],
            "pre_registered": pre_registered,
        }

        by = comp["by_year"].copy()
        by["comparison"] = comp["comparison"]
        by["candidate"] = candidate
        by["baseline"] = baseline
        by["pre_registered"] = pre_registered
        by_year_rows.extend(by.to_dict("records"))

        gate = comp["gate"]
        boot = comp["bootstrap"]
        registry_rows.append({
            "comparison": comp["comparison"],
            "baseline": baseline,
            "candidate": candidate,
            "pre_registered": pre_registered,
            "candidate_mean_ic": comp["candidate_ic"]["mean"],
            "mean_delta_ic": comp["delta"]["mean"],
            "boot_ci_low": boot["ci_low"],
            "boot_ci_high": boot["ci_high"],
            "years_positive": gate["years_positive"],
            "n_years": gate["n_years"],
            "gate_candidate_ic": gate["gate_candidate_ic"],
            "gate_mean_delta": gate["gate_mean_delta"],
            "gate_boot_low": gate["gate_boot_low"],
            "gate_years": gate["gate_years"],
            "gain": gate["gain"],
            "generated_at": generated_at,
        })

    pd.DataFrame(by_year_rows).to_csv(report_dir / "p4_delta_by_year.csv", index=False)
    pd.DataFrame(registry_rows).to_csv(report_dir / "experiment_registry.csv", index=False)

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["generated_at"] = generated_at

    (report_dir / "p4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    _write_verdict(report_dir, comparison_results, source_meta, news_stats, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def main() -> None:
    run_p4_experiment()


if __name__ == "__main__":
    main()
