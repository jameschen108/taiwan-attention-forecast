"""測度效度檢定（PRD §3.10）。

三道篩檢：
  1. 代號 vs. 簡稱一致性：兩種比對得到的週序列應同步跳動，corr < 0.5 標記可疑並
     降級為代號比對。
  2. 交易活動共動：Attention ~ Turnover + |Return|，係數不顯著者標記可疑。
     **這一關在長尾個股上會刷掉不少標的，刷掉本身就是結果，須完整報告而非靜默剔除。**
  3. 人工抽驗：分層抽樣，樣本與判定存檔可查。

輸出 audit/keyword_screening_report.csv 與每檔的 match_quality_tier。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def code_vs_name_consistency(matches: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """檢定 1：同一檔以代號與以簡稱得到的週序列相關性。"""
    rows = []
    for ticker, grp in matches.groupby("ticker"):
        code = (grp[grp["match_mode"] == "code"].groupby("week").size()
                .reindex(weeks, fill_value=0))
        name = (grp[grp["match_mode"] != "code"].groupby("week").size()
                .reindex(weeks, fill_value=0))
        n_code, n_name = int(code.sum()), int(name.sum())
        if n_code >= 5 and n_name >= 5 and code.std() > 0 and name.std() > 0:
            corr = float(np.corrcoef(np.log1p(code), np.log1p(name))[0, 1])
        else:
            corr = np.nan
        rows.append({"ticker": ticker, "n_code_matches": n_code,
                     "n_name_matches": n_name, "code_name_corr": corr})
    return pd.DataFrame(rows)


def trading_activity_comovement(panel: pd.DataFrame) -> pd.DataFrame:
    """檢定 2：Attention ~ Turnover + |Return|（最具說服力的效度證據）。"""
    rows = []
    for ticker, grp in panel.groupby("ticker"):
        d = grp[["att_all", "turnover", "ret"]].dropna()
        d = d[np.isfinite(d).all(axis=1)]
        if len(d) < 30 or d["att_all"].std() == 0:
            rows.append({"ticker": ticker, "n_obs": len(d),
                         "beta_turnover": np.nan, "t_turnover": np.nan,
                         "beta_absret": np.nan, "t_absret": np.nan,
                         "comovement_pass": False,
                         "comovement_note": "觀測數不足或關注度無變異"})
            continue
        y = np.log1p(d["att_all"])
        X = sm.add_constant(pd.DataFrame({
            "turnover": np.log1p(d["turnover"] * 1e4),
            "absret": d["ret"].abs(),
        }))
        try:
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        except Exception as exc:  # noqa: BLE001
            rows.append({"ticker": ticker, "n_obs": len(d),
                         "beta_turnover": np.nan, "t_turnover": np.nan,
                         "beta_absret": np.nan, "t_absret": np.nan,
                         "comovement_pass": False, "comovement_note": str(exc)[:60]})
            continue
        t_to = float(fit.tvalues["turnover"])
        t_ar = float(fit.tvalues["absret"])
        rows.append({
            "ticker": ticker, "n_obs": len(d),
            "beta_turnover": float(fit.params["turnover"]), "t_turnover": t_to,
            "beta_absret": float(fit.params["absret"]), "t_absret": t_ar,
            "comovement_pass": bool(t_to > 1.96 or t_ar > 1.96),
            "comovement_note": "",
        })
    return pd.DataFrame(rows)


def manual_review_sample(matches: pd.DataFrame, panel: pd.DataFrame,
                         collision_members: list[str], rng_seed: int = 20260905,
                         per_tier: int = 50, per_member: int = 20) -> pd.DataFrame:
    """檢定 3：分層抽樣供人工判定歸屬正確與否。樣本存檔可查。"""
    rng = np.random.default_rng(rng_seed)
    tier_by_ticker = (panel.groupby("ticker")["sparsity_tier"]
                      .agg(lambda s: s.dropna().mode().iat[0]
                           if not s.dropna().empty else "unknown"))
    m = matches.merge(tier_by_ticker.rename("sparsity_tier"),
                      left_on="ticker", right_index=True, how="left")

    parts = []
    for tier, grp in m.groupby("sparsity_tier"):
        n = min(per_tier, len(grp))
        parts.append(grp.sample(n, random_state=int(rng.integers(1 << 31)))
                     .assign(stratum=f"tier:{tier}"))
    for ticker in collision_members:
        grp = m[m["ticker"] == ticker]
        if grp.empty:
            continue
        n = min(per_member, len(grp))
        parts.append(grp.sample(n, random_state=int(rng.integers(1 << 31)))
                     .assign(stratum=f"collision:{ticker}"))

    sample = pd.concat(parts, ignore_index=True)
    sample["human_verdict"] = ""       # correct / wrong / ambiguous — 待人工填寫
    sample["human_note"] = ""
    return sample[["stratum", "ticker", "article_id", "timestamp", "category",
                   "match_mode", "effort", "sparsity_tier",
                   "human_verdict", "human_note"]]


def run(matches_path: Path, panel_path: Path, universe_cfg: Path,
        audit_dir: Path, min_corr: float = 0.5) -> pd.DataFrame:
    import yaml

    matches = pd.read_parquet(matches_path)
    matches["week"] = pd.to_datetime(matches["week"])
    panel = pd.read_parquet(panel_path)
    cfg = yaml.safe_load(universe_cfg.read_text(encoding="utf-8"))
    weeks = pd.DatetimeIndex(sorted(panel["week"].unique()))

    c1 = code_vs_name_consistency(matches, weeks)
    c2 = trading_activity_comovement(panel)
    rep = c1.merge(c2, on="ticker", how="outer")

    uni = pd.read_csv("data/external/universe.csv", dtype={"ticker": str})
    rep = rep.merge(uni[["ticker", "name_short", "sector"]], on="ticker", how="left")

    code_only = set(cfg.get("code_only_tickers", []))
    rep["already_code_only"] = rep["ticker"].isin(code_only)
    rep["corr_fail"] = (rep["code_name_corr"] < min_corr) & rep["code_name_corr"].notna()
    rep["recommend_demote_to_code_only"] = rep["corr_fail"] & ~rep["already_code_only"]

    def tier(r):
        if not r["comovement_pass"]:
            return "C_suspect_no_comovement"
        if r["corr_fail"] or r["already_code_only"]:
            return "B_code_only"
        return "A_clean"

    rep["match_quality_tier"] = rep.apply(tier, axis=1)

    audit_dir.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "name_short", "sector", "n_code_matches", "n_name_matches",
            "code_name_corr", "corr_fail", "already_code_only",
            "recommend_demote_to_code_only", "n_obs", "beta_turnover", "t_turnover",
            "beta_absret", "t_absret", "comovement_pass", "comovement_note",
            "match_quality_tier"]
    rep[cols].sort_values("ticker").to_csv(
        audit_dir / "keyword_screening_report.csv", index=False)

    members = sorted({t for g in cfg["collision_groups"].values() for t in g})
    sample = manual_review_sample(matches, panel, members)
    sample.to_csv(audit_dir / "ptt_manual_review_sample.csv", index=False)

    print(f"效度篩檢：A_clean {int((rep['match_quality_tier'] == 'A_clean').sum())}、"
          f"B_code_only {int((rep['match_quality_tier'] == 'B_code_only').sum())}、"
          f"C_suspect {int((rep['match_quality_tier'] == 'C_suspect_no_comovement').sum())}")
    print(f"建議新增降級 {int(rep['recommend_demote_to_code_only'].sum())} 檔；"
          f"人工抽驗樣本 {len(sample)} 篇")
    return rep


if __name__ == "__main__":
    run(Path("data/interim/ptt_matches.parquet"),
        Path("data/processed/panel.parquet"),
        Path("config/universe.yaml"), Path("audit"))
