"""Weekly long-only ranking backtest with costs (daily_bar_proxy)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import CostSchedule, load_cost_schedule
from src.backtest.metrics import nav_metrics


@dataclass
class Position:
    ticker: str
    shares: int
    cost_basis: float


@dataclass
class BacktestState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    receivables: float = 0.0


ADV20_MIN_OBS = 20


def _adv20(daily: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Past 20 trading days mean daily value strictly before as_of.

    Returns (adv_mean, n_obs) per ticker. Tickers with fewer than ADV20_MIN_OBS
    sessions get NaN adv and are ineligible for new positions (spec §11.2).
    """
    cutoff = pd.Timestamp(as_of).normalize()
    hist = daily[daily["date"] < cutoff].copy()
    adv_parts: dict[str, float] = {}
    nobs_parts: dict[str, float] = {}
    for ticker, grp in hist.groupby("ticker", sort=False):
        tail = grp.sort_values("date").tail(ADV20_MIN_OBS)
        n_obs = len(tail)
        nobs_parts[ticker] = float(n_obs)
        if n_obs < ADV20_MIN_OBS:
            adv_parts[ticker] = np.nan
        else:
            adv_parts[ticker] = float(tail["value"].mean())
    return pd.Series(adv_parts), pd.Series(nobs_parts)


def _week_prices(
    daily: pd.DataFrame,
    ticker: str,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
) -> tuple[float, float] | tuple[None, None]:
    sub = daily[
        (daily["ticker"] == ticker)
        & (daily["date"] >= pd.Timestamp(entry_at).normalize())
        & (daily["date"] <= pd.Timestamp(exit_at).normalize())
    ].sort_values("date")
    if sub.empty:
        return None, None
    entry_px = sub.iloc[0]["adj_open"]
    exit_px = sub.iloc[-1]["adj_close"]
    if pd.isna(entry_px) or pd.isna(exit_px) or entry_px <= 0:
        return None, None
    return float(entry_px), float(exit_px)


def _mark_positions(
    daily: pd.DataFrame,
    positions: dict[str, Position],
    mark_date: pd.Timestamp,
) -> float:
    total = 0.0
    for ticker, pos in positions.items():
        sub = daily[
            (daily["ticker"] == ticker)
            & (daily["date"] <= pd.Timestamp(mark_date).normalize())
        ].sort_values("date")
        if sub.empty:
            continue
        px = sub.iloc[-1]["adj_close"]
        if pd.notna(px):
            total += pos.shares * float(px)
    return total


def _liquidate_carried(
    state: BacktestState,
    daily: pd.DataFrame,
    as_of: pd.Timestamp,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
    costs: CostSchedule,
    strategy_name: str,
    slippage_stress: float,
    orders: list[dict],
    total_fees: float,
    total_turnover: float,
) -> tuple[float, float]:
    """Attempt to sell positions carried from prior failed exits."""
    if not state.positions:
        return total_fees, total_turnover
    proceeds = 0.0
    still_carry: dict[str, Position] = {}
    for ticker, pos in list(state.positions.items()):
        _, px_exit = _week_prices(daily, ticker, entry_at, exit_at)
        if px_exit is None:
            # Try last available close on or before exit_at
            sub = daily[
                (daily["ticker"] == ticker)
                & (daily["date"] <= pd.Timestamp(exit_at).normalize())
            ].sort_values("date")
            px_exit = float(sub.iloc[-1]["adj_close"]) if not sub.empty else None
        if px_exit is None:
            orders.append({
                "as_of": as_of, "ticker": ticker, "side": "sell",
                "status": "failed_exit_carry", "shares": pos.shares,
                "strategy": strategy_name,
            })
            still_carry[ticker] = pos
            continue
        exec_sell = px_exit * costs.sell_slippage_multiplier(slippage_stress)
        gross = pos.shares * exec_sell
        fee = costs.sell_fee(gross)
        proceeds += gross - fee
        total_fees += fee
        total_turnover += gross
        orders.append({
            "as_of": as_of, "ticker": ticker, "side": "sell",
            "status": "filled_carry_liquidation", "shares": pos.shares,
            "price": exec_sell, "gross": gross, "fee": fee,
            "strategy": strategy_name, "entry_at": entry_at, "exit_at": exit_at,
        })
    state.positions = still_carry
    state.cash += proceeds
    return total_fees, total_turnover


def simulate_weekly_rank_strategy(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    cost_path,
    *,
    pred_col: str = "pred_m1",
    strategy_name: str = "ridge_A",
    top_k: int = 20,
    max_weight: float = 0.05,
    min_adv20: float = 5_000_000,
    max_order_frac_adv: float = 0.01,
    initial_cash: float = 1_000_000,
    slippage_stress: float = 1.0,
    sparsity_tiers: list[str] | None = None,
) -> dict:
    """Simulate non-overlapping weekly round-trips from saved predictions."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ticker"] = daily["ticker"].astype(str)

    lab = labels[labels["horizon"] == "1w"][
        ["as_of", "ticker", "entry_at", "label_end_at", "label_status"]
    ].drop_duplicates(["as_of", "ticker"])

    pred_cols = [
        c for c in predictions.columns
        if c not in {"label_status", "entry_at", "label_end_at"}
    ]
    pred = predictions[pred_cols].merge(lab, on=["as_of", "ticker"], how="left")
    if pred.empty:
        return _empty_result(initial_cash, strategy_name, slippage_stress)

    if sparsity_tiers and "sparsity_tier" in pred.columns:
        pred = pred[pred["sparsity_tier"].isin(sparsity_tiers)].copy()

    pred = pred[pred["label_status"] == "ok"].copy()
    pred = pred.dropna(subset=[pred_col, "entry_at", "label_end_at"])

    state = BacktestState(cash=initial_cash)
    orders: list[dict] = []
    nav_rows: list[dict] = [{
        "as_of": pd.NaT,
        "nav": initial_cash,
        "cash": initial_cash,
        "n_positions_carried": 0,
        "strategy": strategy_name,
        "is_initial": True,
    }]
    total_turnover = 0.0
    total_fees = 0.0

    for as_of, grp in pred.groupby("as_of", sort=True):
        costs = load_cost_schedule(cost_path, as_of)
        adv, n_obs = _adv20(daily, as_of)
        ranked = grp.sort_values(pred_col, ascending=False)
        targets = ranked.head(top_k)
        if targets.empty:
            continue

        entry_at = targets.iloc[0]["entry_at"]
        exit_at = targets.iloc[0]["label_end_at"]

        total_fees, total_turnover = _liquidate_carried(
            state, daily, as_of, entry_at, exit_at, costs,
            strategy_name, slippage_stress, orders, total_fees, total_turnover,
        )

        nav_before = state.cash + _mark_positions(daily, state.positions, entry_at)
        target_value = nav_before * max_weight
        spent = 0.0
        week_positions: dict[str, Position] = {}

        for _, row in targets.iterrows():
            ticker = row["ticker"]
            if ticker not in adv.index or pd.isna(adv[ticker]):
                status = "insufficient_adv_history" if (
                    ticker in n_obs.index and n_obs[ticker] < ADV20_MIN_OBS
                ) else "rejected_adv"
                orders.append({
                    "as_of": as_of, "ticker": ticker, "side": "buy",
                    "status": status, "strategy": strategy_name,
                })
                continue
            if adv[ticker] < min_adv20:
                orders.append({
                    "as_of": as_of, "ticker": ticker, "side": "buy",
                    "status": "rejected_adv", "strategy": strategy_name,
                })
                continue
            px_entry, px_exit = _week_prices(daily, ticker, entry_at, exit_at)
            if px_entry is None:
                orders.append({
                    "as_of": as_of, "ticker": ticker, "side": "buy",
                    "status": "no_price", "strategy": strategy_name,
                })
                continue
            exec_buy = px_entry * costs.buy_slippage_multiplier(slippage_stress)
            budget = min(target_value, state.cash - spent)
            if budget <= 0:
                break
            max_notional = adv[ticker] * max_order_frac_adv
            budget = min(budget, max_notional)
            shares = int(budget / exec_buy / costs.lot_size) * costs.lot_size
            if shares <= 0:
                orders.append({
                    "as_of": as_of, "ticker": ticker, "side": "buy",
                    "status": "lot_too_small", "strategy": strategy_name,
                })
                continue
            gross = shares * exec_buy
            fee = costs.buy_fee(gross)
            total_cost = gross + fee
            if total_cost > state.cash - spent:
                continue
            spent += total_cost
            total_fees += fee
            total_turnover += gross
            week_positions[ticker] = Position(ticker, shares, exec_buy)
            orders.append({
                "as_of": as_of, "ticker": ticker, "side": "buy",
                "status": "filled", "shares": shares, "price": exec_buy,
                "gross": gross, "fee": fee, "strategy": strategy_name,
                "entry_at": entry_at, "exit_at": exit_at,
            })

        state.cash -= spent
        week_proceeds = 0.0
        for ticker, pos in week_positions.items():
            _, px_exit = _week_prices(daily, ticker, entry_at, exit_at)
            if px_exit is None:
                orders.append({
                    "as_of": as_of, "ticker": ticker, "side": "sell",
                    "status": "failed_exit_carry", "shares": pos.shares,
                    "strategy": strategy_name,
                })
                state.positions[ticker] = pos
                continue
            exec_sell = px_exit * costs.sell_slippage_multiplier(slippage_stress)
            gross = pos.shares * exec_sell
            fee = costs.sell_fee(gross)
            week_proceeds += gross - fee
            total_fees += fee
            total_turnover += gross
            orders.append({
                "as_of": as_of, "ticker": ticker, "side": "sell",
                "status": "filled", "shares": pos.shares, "price": exec_sell,
                "gross": gross, "fee": fee, "strategy": strategy_name,
            })

        state.cash += week_proceeds
        nav = state.cash + _mark_positions(daily, state.positions, exit_at)
        nav_rows.append({
            "as_of": as_of, "nav": nav, "cash": state.cash,
            "n_positions_carried": len(state.positions),
            "strategy": strategy_name,
            "is_initial": False,
        })

    nav_df = pd.DataFrame(nav_rows)
    metrics = nav_metrics(nav_df, initial_cash=initial_cash) if len(nav_df) > 1 else {}
    metrics["total_fees"] = total_fees
    metrics["total_turnover"] = total_turnover
    metrics["strategy"] = strategy_name
    metrics["slippage_stress"] = slippage_stress
    metrics["initial_cash"] = initial_cash
    return {
        "metrics": metrics,
        "nav": nav_df,
        "orders": pd.DataFrame(orders),
        "final_state": state,
    }


def simulate_benchmark_weekly(
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    cost_path,
    *,
    bench_ticker: str = "0050",
    strategy_name: str = "benchmark_0050_weekly",
    initial_cash: float = 1_000_000,
    slippage_stress: float = 1.0,
) -> dict:
    """Hold benchmark ETF for each 1w window (same entry/exit as stock strategies)."""
    lab = labels[labels["horizon"] == "1w"][
        ["as_of", "entry_at", "label_end_at", "label_status"]
    ].drop_duplicates(subset=["as_of"], keep="first")
    lab = lab[lab["label_status"] == "ok"].sort_values("as_of")

    state = BacktestState(cash=initial_cash)
    orders: list[dict] = []
    nav_rows: list[dict] = [{
        "as_of": pd.NaT, "nav": initial_cash, "cash": initial_cash,
        "n_positions_carried": 0, "strategy": strategy_name, "is_initial": True,
    }]
    total_fees = 0.0
    total_turnover = 0.0

    for _, row in lab.iterrows():
        as_of = row["as_of"]
        entry_at = row["entry_at"]
        exit_at = row["label_end_at"]
        costs = load_cost_schedule(cost_path, as_of)
        px_entry, px_exit = _week_prices(daily, bench_ticker, entry_at, exit_at)
        if px_entry is None or px_exit is None:
            nav_rows.append({
                "as_of": as_of, "nav": state.cash, "cash": state.cash,
                "n_positions_carried": 0, "strategy": strategy_name, "is_initial": False,
            })
            continue
        exec_buy = px_entry * costs.buy_slippage_multiplier(slippage_stress)
        shares = int(state.cash / exec_buy / costs.lot_size) * costs.lot_size
        if shares <= 0:
            nav_rows.append({
                "as_of": as_of, "nav": state.cash, "cash": state.cash,
                "n_positions_carried": 0, "strategy": strategy_name, "is_initial": False,
            })
            continue
        gross_buy = shares * exec_buy
        fee_buy = costs.buy_fee(gross_buy)
        state.cash -= gross_buy + fee_buy
        total_fees += fee_buy
        total_turnover += gross_buy
        exec_sell = px_exit * costs.sell_slippage_multiplier(slippage_stress)
        gross_sell = shares * exec_sell
        fee_sell = costs.sell_fee(gross_sell)
        state.cash += gross_sell - fee_sell
        total_fees += fee_sell
        total_turnover += gross_sell
        orders.extend([
            {"as_of": as_of, "ticker": bench_ticker, "side": "buy", "status": "filled",
             "shares": shares, "price": exec_buy, "gross": gross_buy, "fee": fee_buy,
             "strategy": strategy_name, "entry_at": entry_at, "exit_at": exit_at},
            {"as_of": as_of, "ticker": bench_ticker, "side": "sell", "status": "filled",
             "shares": shares, "price": exec_sell, "gross": gross_sell, "fee": fee_sell,
             "strategy": strategy_name},
        ])
        nav_rows.append({
            "as_of": as_of, "nav": state.cash, "cash": state.cash,
            "n_positions_carried": 0, "strategy": strategy_name, "is_initial": False,
        })

    nav_df = pd.DataFrame(nav_rows)
    metrics = nav_metrics(nav_df, initial_cash=initial_cash) if len(nav_df) > 1 else {}
    metrics.update({
        "total_fees": total_fees, "total_turnover": total_turnover,
        "strategy": strategy_name, "slippage_stress": slippage_stress,
        "initial_cash": initial_cash,
    })
    return {"metrics": metrics, "nav": nav_df, "orders": pd.DataFrame(orders),
            "final_state": state}


def simulate_benchmark_buy_hold(
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    cost_path,
    *,
    bench_ticker: str = "0050",
    strategy_name: str = "benchmark_0050_buy_hold",
    initial_cash: float = 1_000_000,
    slippage_stress: float = 1.0,
) -> dict:
    """Buy benchmark once at first entry week; mark-to-market through last exit."""
    lab = labels[labels["horizon"] == "1w"][
        ["as_of", "entry_at", "label_end_at", "label_status"]
    ].drop_duplicates(subset=["as_of"], keep="first")
    lab = lab[lab["label_status"] == "ok"].sort_values("as_of")
    if lab.empty:
        return _empty_result(initial_cash, strategy_name, slippage_stress)

    first = lab.iloc[0]
    last = lab.iloc[-1]
    costs = load_cost_schedule(cost_path, first["as_of"])
    px_entry, _ = _week_prices(daily, bench_ticker, first["entry_at"], first["label_end_at"])
    if px_entry is None:
        return _empty_result(initial_cash, strategy_name, slippage_stress)

    exec_buy = px_entry * costs.buy_slippage_multiplier(slippage_stress)
    shares = int(initial_cash / exec_buy / costs.lot_size) * costs.lot_size
    gross_buy = shares * exec_buy
    fee_buy = costs.buy_fee(gross_buy)
    cash = initial_cash - gross_buy - fee_buy
    total_fees = fee_buy
    total_turnover = gross_buy
    pos = Position(bench_ticker, shares, exec_buy) if shares > 0 else None

    nav_rows: list[dict] = [{
        "as_of": pd.NaT, "nav": initial_cash, "cash": initial_cash,
        "n_positions_carried": 0, "strategy": strategy_name, "is_initial": True,
    }]
    orders: list[dict] = [{
        "as_of": first["as_of"], "ticker": bench_ticker, "side": "buy", "status": "filled",
        "shares": shares, "price": exec_buy, "gross": gross_buy, "fee": fee_buy,
        "strategy": strategy_name,
    }]

    for _, row in lab.iterrows():
        as_of = row["as_of"]
        exit_at = row["label_end_at"]
        nav = cash + (pos.shares * (_week_prices(daily, bench_ticker, row["entry_at"], exit_at)[1] or 0)
                      if pos else 0)
        nav_rows.append({
            "as_of": as_of, "nav": nav, "cash": cash,
            "n_positions_carried": 1 if pos else 0,
            "strategy": strategy_name, "is_initial": False,
        })

    # Final liquidation at last window close
    if pos:
        _, px_exit = _week_prices(daily, bench_ticker, last["entry_at"], last["label_end_at"])
        if px_exit:
            exec_sell = px_exit * costs.sell_slippage_multiplier(slippage_stress)
            gross_sell = pos.shares * exec_sell
            fee_sell = costs.sell_fee(gross_sell)
            cash += gross_sell - fee_sell
            total_fees += fee_sell
            total_turnover += gross_sell
            orders.append({
                "as_of": last["as_of"], "ticker": bench_ticker, "side": "sell",
                "status": "filled", "shares": pos.shares, "price": exec_sell,
                "gross": gross_sell, "fee": fee_sell, "strategy": strategy_name,
            })

    nav_df = pd.DataFrame(nav_rows)
    metrics = nav_metrics(nav_df, initial_cash=initial_cash) if len(nav_df) > 1 else {}
    metrics.update({
        "total_fees": total_fees, "total_turnover": total_turnover,
        "strategy": strategy_name, "slippage_stress": slippage_stress,
        "initial_cash": initial_cash,
    })
    return {"metrics": metrics, "nav": nav_df, "orders": pd.DataFrame(orders),
            "final_state": BacktestState(cash=cash)}


def execution_diagnostics(orders: pd.DataFrame, nav: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """Weekly fill rate, cash drag, and rejection reasons."""
    if orders.empty:
        return pd.DataFrame()
    buys = orders[orders["side"] == "buy"].copy()
    rows = []
    for as_of, grp in buys.groupby("as_of", sort=True):
        status_counts = grp["status"].value_counts().to_dict()
        filled = int(status_counts.get("filled", 0))
        nav_row = nav[nav["as_of"] == as_of]
        nav_val = float(nav_row["nav"].iloc[0]) if not nav_row.empty else np.nan
        cash_val = float(nav_row["cash"].iloc[0]) if not nav_row.empty else np.nan
        rows.append({
            "as_of": as_of,
            "intended_buys": top_k,
            "attempted_buys": len(grp),
            "filled_buys": filled,
            "fill_rate": filled / top_k if top_k else np.nan,
            "rejected_adv": status_counts.get("rejected_adv", 0),
            "insufficient_adv_history": status_counts.get("insufficient_adv_history", 0),
            "lot_too_small": status_counts.get("lot_too_small", 0),
            "no_price": status_counts.get("no_price", 0),
            "nav": nav_val,
            "cash": cash_val,
            "cash_fraction": cash_val / nav_val if nav_val and nav_val > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _empty_result(initial_cash: float, strategy_name: str, slippage_stress: float) -> dict:
    return {
        "metrics": {
            "total_return": 0.0, "max_drawdown": 0.0, "final_nav": initial_cash,
            "initial_nav": initial_cash, "n_periods": 0, "strategy": strategy_name,
            "total_fees": 0.0, "total_turnover": 0.0, "slippage_stress": slippage_stress,
            "initial_cash": initial_cash,
        },
        "nav": pd.DataFrame(),
        "orders": pd.DataFrame(),
        "final_state": BacktestState(cash=initial_cash),
    }
