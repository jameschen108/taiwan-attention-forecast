# Forecast data dictionary (P0)

## Keys

| Table | Grain | Key |
|---|---|---|
| `data/forecast/features.parquet` | ticker × as_of | `ticker`, `week`, `as_of` |
| `data/forecast/labels.parquet` | ticker × as_of × horizon | `ticker`, `week`, `as_of`, `horizon` |
| `data/forecast/predictions.parquet` | ticker × as_of × model run | `run_id`, `as_of`, `ticker` |

## Time fields

| Field | Meaning |
|---|---|
| `week` | Feature week end (W-SUN) |
| `as_of` | Monday 00:00 after `week` — information cutoff |
| `entry_at` | First trading day of return week |
| `label_end_at` | Last trading day of return week |
| `label_available_at` | Day after `label_end_at` (proxy) |

## Labels

| Field | Definition |
|---|---|
| `y_stock_1w` | Stock Mon-open→Fri-close adj return of return week (`ret_next`) |
| `y_bench_1w` | 0050 same-window adj return |
| `y_excess_1w` | `y_stock_1w - y_bench_1w` |
| `label_definition` | `adjusted_price_proxy` until cash ledger total return |

## Feature sets

See `config/forecast_feature_whitelist.yaml`. Forbidden columns include all `ret_next` / `ret_fwd*` / `y_*`.
