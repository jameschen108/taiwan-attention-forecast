# Forecast source availability assumptions (P0)

Version: attention_forecast_v1  
Mode: `retrospective_proxy` (not `historical_pit` / `prospective`)

| Source | `event_at` | `published_at` | `available_at` assumption (proxy) | Notes |
|---|---|---|---|---|
| PTT posts | post timestamp | post timestamp | post timestamp | Edits after publish not versioned; body treated as publish-time snapshot with limitation |
| TWSE/FinMind daily OHLCV | trade date | trade date EOD | next calendar day 00:00 (proxy: same as label_available) | No historical ingest clock; retrospective |
| Shareholding (MI_QFIIS) | sample date | sample date | sample date (ffill only; **no bfill** in forecast rebuilds) | First published value must not be pushed earlier |
| Ex-rights / reductions | effective date | announcement/table date if known | effective date (proxy) | 0050 uses `ex_rights.csv` factors when ticker matches |
| Benchmark 0050 | trade date | trade date | same as stock prices | FinMind `TaiwanStockPrice` JSON |
| Universe membership | — | — | frozen ex-post list | `existing_ex_post_universe`; not historically tradable claim |

Formal prospective mode requires real `ingested_at` snapshots and must flip `availability_mode` only when those exist.
