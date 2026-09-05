"""論文圖 F1–F5（PRD §7.3）。

F1 關注度時間序列與覆蓋熱圖（267 × 週）
F2 主係數與信賴區間的規格曲線圖
F3 β₂ 隨覆蓋度分位數變化的曲線
F4 起始事件的 CAR 路徑圖
F5 投資組合累積報酬（成本前／成本後兩條線）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字型：macOS 常駐字型優先，缺字時退回英文標籤不致崩潰
for family in ("Heiti TC", "PingFang TC", "Arial Unicode MS", "Songti SC"):
    if family in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = family
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def f1_coverage_heatmap(panel: pd.DataFrame, out: Path) -> None:
    """覆蓋熱圖：個股（依總關注度排序）× 週，顏色為 log1p(文章數)。"""
    piv = panel.pivot_table(index="ticker", columns="week", values="att_all",
                            aggfunc="sum")
    order = piv.sum(axis=1).sort_values(ascending=False).index
    piv = piv.loc[order]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    im = axes[0].imshow(np.log1p(piv.fillna(0).values), aspect="auto",
                        cmap="magma", interpolation="nearest")
    axes[0].set_title("F1a 關注度覆蓋熱圖（個股依總關注度降冪；顏色為 log1p 文章數）")
    axes[0].set_ylabel("個股（共 %d 檔）" % len(piv))
    axes[0].set_xlabel("")
    ticks = np.linspace(0, piv.shape[1] - 1, 10).astype(int)
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels([str(piv.columns[i].date()) for i in ticks],
                            rotation=45, ha="right", fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01)

    weekly = panel.groupby("week")["att_all"].sum()
    nonzero = panel.groupby("week")["att_all"].apply(lambda s: (s > 0).sum())
    axes[1].plot(weekly.index, weekly.values, lw=0.8, label="每週總文章數")
    ax2 = axes[1].twinx()
    ax2.plot(nonzero.index, nonzero.values, lw=0.8, color="tab:orange",
             label="每週有討論的個股數")
    axes[1].set_title("F1b 關注度時間序列")
    axes[1].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def f2_specification_curve(tables_dir: Path, out: Path) -> None:
    """規格曲線：所有規格的週末係數與 95% 信賴區間。"""
    frames = []
    for name in ("T3_main_regressions", "T12_robustness"):
        path = tables_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "term" in df.columns:
            df = df[df["term"] == "abn_attention_weekend"]
            df = df.rename(columns={"coef": "beta_focus", "se": "se_focus",
                                    "model": "check"})
        else:
            df = df[df["status"] == "OK"].copy()
            df["se_focus"] = df["beta_focus"] / df["t_focus"].replace(0, np.nan)
        frames.append(df[["check", "beta_focus", "se_focus"]])
    if not frames:
        return
    d = pd.concat(frames, ignore_index=True).dropna(subset=["beta_focus"])
    d = d.sort_values("beta_focus").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.25 * len(d))))
    y = np.arange(len(d))
    ax.errorbar(d["beta_focus"], y, xerr=1.96 * d["se_focus"].abs(),
                fmt="o", ms=3, lw=0.8, capsize=2)
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["check"], fontsize=6)
    ax.set_xlabel("β（週末異常關注度）與 95% CI")
    ax.set_title("F2 規格曲線（全部為 diagnostic 結果）")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def f3_beta_by_coverage(panel: pd.DataFrame, out: Path, n_bins: int = 5) -> None:
    """β₂ 隨常態關注度水準分位數變化的曲線（H6 的圖示）。"""
    from src.analysis.regressions import BASE_CONTROLS, fit_twoway, standardize_within

    d = panel.dropna(subset=["log_att_mean_level_52"]).copy()
    if d.empty:
        return
    try:
        d["bin"] = pd.qcut(d["log_att_mean_level_52"].rank(method="first"),
                           n_bins, labels=range(1, n_bins + 1))
    except ValueError:
        return

    rows = []
    for b, g in d.groupby("bin", observed=True):
        std = standardize_within(g, ["abn_attention_weekend",
                                     *[c for c in BASE_CONTROLS
                                       if c != "att_zero_base"]])
        r = fit_twoway(std, "ret_next", ["abn_attention_weekend"],
                       BASE_CONTROLS, f"bin{b}")
        if r.status == "OK":
            rows.append({
                "bin": int(b),
                "beta": r.params["abn_attention_weekend"],
                "se": r.stderr["abn_attention_weekend"],
                "mean_level": float(g["att_mean_level_52"].mean()),
                "n_obs": r.n_obs,
            })
    if not rows:
        return
    res = pd.DataFrame(rows)
    res.to_csv(out.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(res["bin"], res["beta"], yerr=1.96 * res["se"],
                fmt="o-", capsize=3, lw=1)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(res["bin"])
    ax.set_xticklabels([f"Q{b}\n均{l:.1f}篇" for b, l in
                        zip(res["bin"], res["mean_level"])], fontsize=8)
    ax.set_xlabel("常態關注度水準分位（低 → 高）")
    ax.set_ylabel("β（週末異常關注度 → 次週報酬）")
    ax.set_title("F3 效果強度隨覆蓋度變化（H6；diagnostic）")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def f4_car_path(car: pd.DataFrame, out: Path) -> None:
    """起始事件的 CAR 路徑，含事前四週的平行趨勢檢查。"""
    if car.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(car["tau"], car["car"], "o-", lw=1.2, label="CAR")
    lo = car["car"] - 1.96 * car["se"].fillna(0).cumsum() ** 0.5 * 0
    ax.errorbar(car["tau"], car["mean_ar"], yerr=1.96 * car["se"],
                fmt="s--", ms=3, lw=0.7, alpha=0.6, label="每週 AR 與 95% CI")
    ax.axvline(0, color="crimson", lw=1, ls=":", label="起始事件週")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.axvspan(-4.4, -0.6, color="0.9", zorder=0)
    ax.text(-2.5, ax.get_ylim()[1] * 0.92, "平行趨勢檢查區",
            ha="center", fontsize=8, color="0.35")
    ax.set_xlabel("相對事件週 τ")
    ax.set_ylabel("累積異常報酬")
    ax.set_title(f"F4 關注度起始事件 CAR 路徑（{int(car['n_events'].max())} 事件；diagnostic）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def f5_portfolio_cumulative(weekly: pd.DataFrame, summary: pd.DataFrame,
                            out: Path) -> None:
    """投資組合累積報酬：成本前／成本後兩條線。"""
    if weekly.empty or "long_short" not in weekly.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for spec, g in weekly.groupby("spec"):
        row = summary[summary["portfolio"] == spec]
        if row.empty or row.iloc[0]["status"] != "OK":
            continue
        cost = float(row.iloc[0]["weekly_cost_assumed"])
        g = g.sort_values("week")
        weeks = pd.to_datetime(g["week"])
        ax.plot(weeks, g["long_short"].fillna(0).cumsum(), lw=1,
                label=f"{spec}（成本前）")
        ax.plot(weeks, (g["long_short"].fillna(0) - cost).cumsum(), lw=1,
                ls="--", label=f"{spec}（成本後）")
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.set_ylabel("累積多空價差（未年化）")
    ax.set_title("F5 五等分多空累積價差\n"
                 "ex-post universe；不可解讀為可投資績效（PRD §11）", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run(panel: pd.DataFrame, tables_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    f1_coverage_heatmap(panel, out_dir / "F1_coverage.png")
    made.append("F1_coverage.png")
    f2_specification_curve(tables_dir, out_dir / "F2_specification_curve.png")
    made.append("F2_specification_curve.png")
    f3_beta_by_coverage(panel[panel["sparsity_tier"].isin(["dense", "sparse"])],
                        out_dir / "F3_beta_by_coverage.png")
    made.append("F3_beta_by_coverage.png")

    car_path = tables_dir / "T7_car_path.csv"
    if car_path.exists():
        f4_car_path(pd.read_csv(car_path), out_dir / "F4_car_path.png")
        made.append("F4_car_path.png")

    wk, sm = tables_dir / "T9_portfolio_weekly.csv", tables_dir / "T9_portfolios.csv"
    if wk.exists() and sm.exists():
        f5_portfolio_cumulative(pd.read_csv(wk), pd.read_csv(sm),
                                out_dir / "F5_portfolio.png")
        made.append("F5_portfolio.png")
    return made
