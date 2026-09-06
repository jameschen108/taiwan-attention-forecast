"""關注度起始事件的反向因果處理（H7 的識別修補）。

**問題**：未匹配的事件研究顯示 τ = −2 的異常報酬已顯著為正（t = 2.34）。也就是
**價格先動、討論才出現**，而不是討論預測價格。平行趨勢不成立時，事件後的 CAR 型態
無法區分「關注度造成報酬」與「兩者都由更早的價格衝擊驅動」。

**做法**：同週、同稀疏度分層的匹配對照事件研究（matched-control DiD event study）。

對每個起始事件 (i, t)，在**同一週 t** 的未起始個股中挑選事前特徵最相近者作為對照：
- 匹配在同一週內進行，市場層級的共同衝擊自動被差分掉（等同於週固定效果）。
- 匹配變數：事前 4 週累積異常報酬（**直接針對反向因果**）、規模、周轉率。
- 估計量為 DiD：`CAR_treated(τ) − CAR_control(τ)`。

**判讀**：
- 匹配後事前 CAR 差異若仍顯著 → 匹配失敗，H7 無法識別。
- 事前差異被消除、事後仍有顯著差異 → 關注度起始帶有超出價格動能的資訊。
- 事前與事後差異都被消除 → 原本的 CAR 型態**完全是動能的重述**，H7 不成立。

另提供兩個對照：
1. `clean` 子樣本——事前累積異常報酬落在中間分位的事件（未被價格驅動）。
2. 傾向分數模型——把「事前報酬是否預測起始」量化，作為反向因果強度的直接證據。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PRE, POST = 4, 8

# 匹配變數：**事前每一週的異常報酬分開進入**，而非只用 4 週累積。
# 只匹配累積值會讓組成不同——關注度起始多半由「上一週剛發生的價格變動」觸發，
# 累積量相同但集中在 τ=−1 的事件仍會殘留事前趨勢（實測 τ=−1 的 t 值達 4.75）。
MATCH_VARS = ["ar_m1", "ar_m2", "pre_car_34", "log_mktcap", "log_turnover"]


# ---------------------------------------------------------------------------
# 事件窗擷取
# ---------------------------------------------------------------------------

def _prepare(panel: pd.DataFrame, tiers: tuple[str, ...]) -> pd.DataFrame:
    d = panel[panel["sparsity_tier"].isin(tiers)].copy()
    d = d.sort_values(["ticker", "week"])
    # 異常報酬 = 個股週報酬 − 同週橫斷面均值（週效果的事件研究對應）
    d["ar"] = d["ret"] - d.groupby("week")["ret"].transform("mean")
    d["log_mktcap"] = np.log(d["market_cap"].where(d["market_cap"] > 0))
    d["log_turnover"] = np.log1p(d["turnover"] * 1e4)
    return d


def _windows(d: pd.DataFrame, event_col: str) -> pd.DataFrame:
    """對每檔取出所有「有完整 τ∈[−PRE, +POST] 連續週」的位置。

    回傳每個候選位置一列，含事前累積 AR、匹配變數，以及整條 AR 路徑。
    """
    recs = []
    for ticker, g in d.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        weeks = g["week"].values
        ar = g["ar"].values
        is_ev = g[event_col].fillna(False).astype(bool).values
        n = len(g)
        for i in range(PRE, n - POST):
            # 事件窗必須是連續週，否則 τ 對齊會錯
            span = weeks[i - PRE:i + POST + 1]
            if np.any(np.diff(span).astype("timedelta64[D]").astype(int) != 7):
                continue
            path = ar[i - PRE:i + POST + 1]
            if np.isnan(path).any():
                continue
            recs.append({
                "ticker": ticker,
                "week": g["week"].iat[i],
                "treated": bool(is_ev[i]),
                "sparsity_tier": g["sparsity_tier"].iat[i],
                "pre_car": float(path[:PRE].sum()),
                "ar_m1": float(path[PRE - 1]),          # τ = −1
                "ar_m2": float(path[PRE - 2]),          # τ = −2
                "pre_car_34": float(path[:PRE - 2].sum()),  # τ = −4, −3 累積
                "log_mktcap": g["log_mktcap"].iat[i],
                "log_turnover": g["log_turnover"].iat[i],
                **{f"ar_{tau}": float(path[k])
                   for k, tau in enumerate(range(-PRE, POST + 1))},
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------

# 事前報酬的匹配容忍度（絕對值，週報酬）。超過即視為不可接受的對照。
# 只用最近鄰不夠：SMD 雖小（< 0.1），但 2,845 個事件使極小的均值差（0.3%）仍
# 高度顯著，事前趨勢因此殘留。改用 caliper，以樣本數換取平衡。
RETURN_CALIPER = 0.010
RET_MATCH_VARS = ["ar_m1", "ar_m2", "pre_car_34"]


def match(cands: pd.DataFrame, k: int = 1, seed: int = 20260906,
          caliper: float | None = RETURN_CALIPER) -> pd.DataFrame:
    """同週、同分層的 k-近鄰匹配，可加事前報酬 caliper。

    距離為匹配變數標準化後的歐氏距離。同週匹配使市場共同衝擊自動差分掉。
    `caliper` 給定時，對照在 τ=−1、−2 與 τ=−4..−3 累積上都必須落在容忍度內；
    否則該事件**無可用對照而被剔除**——寧可損失事件數，也不要帶著事前趨勢做推論。
    """
    usable = cands.dropna(subset=MATCH_VARS)
    pairs = []

    for (week, tier), g in usable.groupby(["week", "sparsity_tier"], sort=False):
        treated = g[g["treated"]]
        controls = g[~g["treated"]]
        if treated.empty or len(controls) < k:
            continue
        # 在該週該分層內標準化，避免跨週尺度差異主導距離
        mu = g[MATCH_VARS].mean()
        sd = g[MATCH_VARS].std().replace(0, np.nan)
        T = ((treated[MATCH_VARS] - mu) / sd).fillna(0.0).to_numpy()
        C = ((controls[MATCH_VARS] - mu) / sd).fillna(0.0).to_numpy()
        # (n_treated, n_control) 距離矩陣
        dist = np.sqrt(((T[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))

        if caliper is not None:
            # 逐一報酬維度施加絕對值 caliper；違反者距離設為無限大
            for v in RET_MATCH_VARS:
                tv = treated[v].to_numpy()[:, None]
                cv = controls[v].to_numpy()[None, :]
                dist = np.where(np.abs(tv - cv) > caliper, np.inf, dist)

        order = np.argsort(dist, axis=1)[:, :k]
        for ti in range(len(treated)):
            trow = treated.iloc[ti]
            for ci in order[ti]:
                if not np.isfinite(dist[ti, ci]):
                    continue          # caliper 內無可用對照 → 該事件剔除
                crow = controls.iloc[ci]
                pairs.append({
                    "week": week, "sparsity_tier": tier,
                    "treated_ticker": trow["ticker"],
                    "control_ticker": crow["ticker"],
                    "distance": float(dist[ti, ci]),
                    **{f"t_ar_{tau}": trow[f"ar_{tau}"]
                       for tau in range(-PRE, POST + 1)},
                    **{f"c_ar_{tau}": crow[f"ar_{tau}"]
                       for tau in range(-PRE, POST + 1)},
                    **{f"t_{v}": trow[v] for v in MATCH_VARS},
                    **{f"c_{v}": crow[v] for v in MATCH_VARS},
                })
    return pd.DataFrame(pairs)


def balance_table(cands: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """匹配前後的標準化差異（|SMD| < 0.1 通常視為平衡）。"""
    rows = []
    t_all = cands[cands["treated"]]
    c_all = cands[~cands["treated"]]
    for v in MATCH_VARS:
        pre_smd = ((t_all[v].mean() - c_all[v].mean())
                   / np.sqrt((t_all[v].var() + c_all[v].var()) / 2))
        if pairs.empty:
            post_smd = np.nan
        else:
            tv, cv = pairs[f"t_{v}"], pairs[f"c_{v}"]
            post_smd = ((tv.mean() - cv.mean())
                        / np.sqrt((tv.var() + cv.var()) / 2))
        rows.append({
            "variable": v,
            "treated_mean": float(t_all[v].mean()),
            "control_mean_unmatched": float(c_all[v].mean()),
            "smd_unmatched": float(pre_smd),
            "control_mean_matched": (float(pairs[f"c_{v}"].mean())
                                     if not pairs.empty else np.nan),
            "smd_matched": float(post_smd) if pairs is not None else np.nan,
            "balanced": bool(abs(post_smd) < 0.1) if not np.isnan(post_smd) else False,
        })
    return pd.DataFrame(rows)


def did_car(pairs: pd.DataFrame) -> pd.DataFrame:
    """DiD 事件路徑：每個 τ 的 (treated − control) 平均差與 t 值。

    標準誤在**事件週**層級群集——同一週的多個事件共享市場狀態，視為獨立會低估標準誤。
    """
    if pairs.empty:
        return pd.DataFrame()
    # 先把同一 treated 事件的多個對照平均掉，回到「每事件一列」
    keys = ["week", "treated_ticker"]
    agg = {f"t_ar_{tau}": "first" for tau in range(-PRE, POST + 1)}
    agg.update({f"c_ar_{tau}": "mean" for tau in range(-PRE, POST + 1)})
    ev = pairs.groupby(keys, as_index=False).agg(agg)

    rows = []
    for tau in range(-PRE, POST + 1):
        diff = ev[f"t_ar_{tau}"] - ev[f"c_ar_{tau}"]
        # 群集於週
        tmp = pd.DataFrame({"d": diff, "week": ev["week"]}).dropna()
        if tmp.empty:
            continue
        fit = sm.OLS(tmp["d"], np.ones(len(tmp))).fit(
            cov_type="cluster", cov_kwds={"groups": tmp["week"]})
        rows.append({
            "tau": tau,
            "mean_diff": float(fit.params.iloc[0]),
            "se": float(fit.bse.iloc[0]),
            "t": float(fit.tvalues.iloc[0]),
            "p": float(fit.pvalues.iloc[0]),
            "n_events": int(len(tmp)),
            "n_week_clusters": int(tmp["week"].nunique()),
            "treated_mean_ar": float(ev[f"t_ar_{tau}"].mean()),
            "control_mean_ar": float(ev[f"c_ar_{tau}"].mean()),
        })
    out = pd.DataFrame(rows)
    out["car_diff"] = out["mean_diff"].cumsum()
    return out


# ---------------------------------------------------------------------------
# 傾向分數：把反向因果的強度量化
# ---------------------------------------------------------------------------

def initiation_propensity(cands: pd.DataFrame) -> pd.DataFrame:
    """事前報酬是否預測「關注度起始」？這是反向因果的直接檢定。"""
    d = cands.dropna(subset=MATCH_VARS).copy()
    if d.empty or d["treated"].nunique() < 2:
        return pd.DataFrame()
    X = sm.add_constant(d[MATCH_VARS])
    try:
        fit = sm.Logit(d["treated"].astype(int), X).fit(disp=0)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.values,
        "z": fit.tvalues.values,
        "p": fit.pvalues.values,
        "odds_ratio": np.exp(fit.params.values),
        "n_obs": len(d),
        "pseudo_r2": fit.prsquared,
    })


# ---------------------------------------------------------------------------
# 執行
# ---------------------------------------------------------------------------

@dataclass
class Result:
    unmatched: pd.DataFrame
    matched: pd.DataFrame
    balance: pd.DataFrame
    propensity: pd.DataFrame
    clean: pd.DataFrame
    verdict: str


def run(panel: pd.DataFrame, event_col: str = "is_initiation",
        tiers: tuple[str, ...] = ("sparse", "silent"),
        k: int = 3) -> Result:
    d = _prepare(panel, tiers)
    cands = _windows(d, event_col)
    if cands.empty or not cands["treated"].any():
        return Result(*[pd.DataFrame()] * 5, "無可用事件窗")

    # 未匹配基準（供對照）
    unmatched_rows = []
    for tau in range(-PRE, POST + 1):
        t = cands.loc[cands["treated"], f"ar_{tau}"]
        c = cands.loc[~cands["treated"], f"ar_{tau}"]
        unmatched_rows.append({
            "tau": tau, "treated_mean_ar": float(t.mean()),
            "control_mean_ar": float(c.mean()),
            "mean_diff": float(t.mean() - c.mean()),
            "n_events": int(len(t)),
        })
    unmatched = pd.DataFrame(unmatched_rows)
    unmatched["car_diff"] = unmatched["mean_diff"].cumsum()

    pairs = match(cands, k=k)
    matched = did_car(pairs)
    bal = balance_table(cands, pairs)
    prop = initiation_propensity(cands)

    # 「乾淨」子樣本：事前累積 AR 落在中間三分位（未明顯被價格驅動）
    tr = cands[cands["treated"]]
    lo, hi = tr["pre_car"].quantile([1 / 3, 2 / 3])  # noqa: E501 — 見下方 clean 定義
    clean_ids = tr[(tr["pre_car"] >= lo) & (tr["pre_car"] <= hi)]
    clean_pairs = pairs.merge(
        clean_ids[["ticker", "week"]].rename(columns={"ticker": "treated_ticker"}),
        on=["treated_ticker", "week"], how="inner")
    clean = did_car(clean_pairs)

    verdict = _verdict(matched)
    return Result(unmatched, matched, bal, prop, clean, verdict)


def _verdict(matched: pd.DataFrame) -> str:
    if matched.empty:
        return "匹配後無可用事件，無法判讀"
    pre = matched[matched["tau"] < 0]
    post = matched[matched["tau"] >= 1]
    pre_ok = bool((pre["t"].abs() < 1.96).all())
    post_sig = bool((post["t"].abs() > 1.96).any())
    if not pre_ok:
        return ("匹配後事前差異仍顯著 → 反向因果未被消除，H7 在本設計下無法識別")
    if post_sig:
        return ("事前差異已消除、事後仍有顯著差異 → 關注度起始帶有超出價格動能的資訊，"
                "支持 H7")
    return ("事前差異已消除、事後亦無顯著差異 → 原始 CAR 型態可由價格動能完全解釋，"
            "H7 不成立")


def main() -> None:
    from src.analysis.regressions import add_derived

    panel = add_derived(pd.read_parquet("data/processed/panel.parquet"))
    out_dir = Path("output/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    for col, tag in (("is_initiation", "all"),
                     ("is_initiation_weekend", "weekend")):
        res = run(panel, col)
        if res.matched.empty:
            print(f"[{tag}] {res.verdict}")
            continue
        res.unmatched.to_csv(out_dir / f"T7M_unmatched_{tag}.csv", index=False)
        res.matched.to_csv(out_dir / f"T7M_matched_{tag}.csv", index=False)
        res.balance.to_csv(out_dir / f"T7M_balance_{tag}.csv", index=False)
        res.clean.to_csv(out_dir / f"T7M_clean_{tag}.csv", index=False)
        if not res.propensity.empty:
            res.propensity.to_csv(out_dir / f"T7M_propensity_{tag}.csv", index=False)
        n_ev = int(res.matched["n_events"].max())
        print(f"[{tag}] 事件 {n_ev}、週群集 {int(res.matched['n_week_clusters'].max())}；"
              f"平衡 {int(res.balance['balanced'].sum())}/{len(res.balance)} 項")
        print(f"        {res.verdict}")


if __name__ == "__main__":
    main()
