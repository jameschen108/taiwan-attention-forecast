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


def _adv20(daily: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Past 20 trading days mean daily value strictly before as_of."""
    cutoff = pd.Timestamp(as_of).normalize()
    hist = daily[daily["date"] < cutoff].copy()
    adv = (
        hist.groupby("ticker")["value"]
        .apply(lambda s: s.tail(20).mean())
    )
    return adv


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
) -> dict:
    """Simulate non-overlapping weekly round-trips from saved predictions.

    Each as_of week: sell prior (none — flat start), buy top_k at entry open,
    sell at label_end close. Cash between weeks.
    """
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
        return {
            "metrics": {"total_return": 0.0, "max_drawdown": 0.0, "final_nav": initial_cash,
                        "initial_nav": initial_cash, "n_periods": 0, "strategy": strategy_name,
                        "total_fees": 0.0, "total_turnover": 0.0, "slippage_stress": slippage_stress},
            "nav": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "final_state": BacktestState(cash=initial_cash),
        }
    pred = pred[pred["label_status"] == "ok"].copy()
    pred = pred.dropna(subset=[pred_col, "entry_at", "label_end_at"])

    state = BacktestState(cash=initial_cash)
    orders: list[dict] = []
    nav_rows: list[dict] = []
    total_turnover = 0.0
    total_fees = 0.0

    for as_of, grp in pred.groupby("as_of", sort=True):
        costs = load_cost_schedule(cost_path, as_of)
        adv = _adv20(daily, as_of)
        ranked = grp.sort_values(pred_col, ascending=False)
        targets = ranked.head(top_k)
        if targets.empty:
            continue

        entry_at = targets.iloc[0]["entry_at"]
        exit_at = targets.iloc[0]["label_end_at"]
        target_value = state.cash * max_weight
        spent = 0.0
        week_positions: dict[str, Position] = {}

        for _, row in targets.iterrows():
            ticker = row["ticker"]
            if ticker not in adv.index or pd.isna(adv[ticker]) or adv[ticker] < min_adv20:
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
            total = gross + fee
            if total > state.cash - spent:
                continue
            spent += total
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
            net = gross - fee
            week_proceeds += net
            total_fees += fee
            total_turnover += gross
            orders.append({
                "as_of": as_of, "ticker": ticker, "side": "sell",
                "status": "filled", "shares": pos.shares, "price": exec_sell,
                "gross": gross, "fee": fee, "strategy": strategy_name,
            })

        state.cash += week_proceeds
        nav = state.cash + sum(
            p.shares * (_week_prices(daily, p.ticker, entry_at, exit_at)[1] or 0)
            for p in state.positions.values()
        )
        nav_rows.append({
            "as_of": as_of, "nav": nav, "cash": state.cash,
            "n_positions_carried": len(state.positions),
            "strategy": strategy_name,
        })

    nav_df = pd.DataFrame(nav_rows)
    metrics = nav_metrics(nav_df) if not nav_df.empty else {}
    metrics["total_fees"] = total_fees
    metrics["total_turnover"] = total_turnover
    metrics["strategy"] = strategy_name
    metrics["slippage_stress"] = slippage_stress
    return {
        "metrics": metrics,
        "nav": nav_df,
        "orders": pd.DataFrame(orders),
        "final_state": state,
    }
