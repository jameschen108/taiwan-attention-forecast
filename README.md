# 台股長尾關注度與報酬可預測性

把 Li, Liu, Ye, Zhao & Zhao, *"It Depends on When You Search"*（MIS Quarterly）的
「非交易時段關注度預測次週報酬」研究設計移植到台灣股市，並利用一個橫跨 28 產業、
267 檔、關注度極度不均的長尾樣本，把「哪一種股票的關注度才有預測力」這個原論文
無法回答的問題，變成識別**資訊處理**與**價格壓力**兩條機制管道的主要工具。

> **先讀 [`LIMITATIONS.md`](LIMITATIONS.md)。** 它是交付門檻的一部分——任何引用本專案
> 結果的論文、簡報或摘要都必須同時引用。第一條是宇宙選樣偏誤：**本清單不是指數、
> 不具市場代表性，推論母體只能寫成「本樣本涵蓋之 267 檔個股」。**

**目前所有結果均標記為 `diagnostic`**（授權控制變數未取得，見 LIMITATIONS §5）。

---

## 安裝

```bash
pip install -r requirements.txt
```

需要 Python ≥ 3.10。無需 API 金鑰：所有公開資料來自 TWSE 開放端點與 FinMind 免費層。

## 執行

一條指令重建全部輸出（實測 **339 秒**）：

```bash
python3 -m src.run_all
```

各階段可獨立重跑（除錯用）：

```bash
python3 -m src.run_all --only panel analysis report
```

階段名稱：`universe` → `audit` → `market` → `ptt` → `panel` → `screen` →
`analysis` → `report`。

### 資料收集（首次執行前）

原始資料收集與分析管線分離，且**永遠不會**由建置指令自動觸發（PRD §4.3 第 8 點）：

```bash
python3 -m src.market.collect_finmind --kinds price   # 267 檔日成交
python3 -m src.market.collect_exrights                # 除權息參考價（全市場）
python3 -m src.market.collect_reduction               # 減資恢復買賣參考價
python3 -m src.market.collect_shareholding            # 發行股數與外資持股
```

各來源的選用理由與被否決的選項見 [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)。

三者皆支援斷點續傳；已抓取的檔案會自動跳過。三大法人資料使用既有的
`data/twse/t86/` 封存，不需重抓。

## 測試

```bash
python3 -m pytest tests/ -q
```

涵蓋 PRD §6.2 F4 列出的每一項核心邏輯：關注度指標、稀疏度分層、起始事件、名稱碰撞
消解、時段判定、訂單失衡、日曆對齊，以及兩個獨立來源的**單位交叉驗證**
（T86 vs. FinMind 逐日比對，防範 PRD R9 的量綱錯誤）。

---

## 資料契約

| 檔案 | 必要欄位 | 單位與鍵值 |
|---|---|---|
| `data/external/universe.csv` | `ticker, name_short, sector, listing_date, delisting_date, venue, is_ky` | 267 列；`ticker` 唯一 |
| `data/external/market_venue.csv` | `ticker, venue, effective_from, effective_to` | `venue ∈ {TWSE, TPEx}`；閉區間、不重疊 |
| `data/interim/ptt_matches.parquet` | `ticker, timestamp, week, category, is_reply, match_mode, effort, window, session` | 同文多檔輸出多列 |
| `data/interim/market_daily.parquet` | `ticker, date, venue, close, adj_close, volume, inst_buy, inst_sell` | 三個量皆為**股數** |
| `data/interim/trading_days.csv` | 實際開市日清單、`is_makeup_saturday` | 由實際成交日推導 |
| `data/interim/ex_rights.csv` | `ticker, date, before_price, after_price, factor` | TWSE 除權除息計算結果表；factor < 1 |
| `data/interim/capital_reductions.csv` | 同上 ＋ `reason` | TWSE 減資恢復買賣參考價格；**factor > 1** |
| `data/processed/panel.parquet` | ticker×week 非平衡面板 | **主結果用** |
| `data/processed/panel_dense.parquet` | `dense` 子樣本 | H2 用 |

格式錯誤的輸入會中止建置；**缺值一律維持缺值**，不補零也不補均值（PRD §3.8）。

## 產出

```
audit/      稽核與品質報告（覆蓋率、名稱碰撞、效度篩檢、health_checks、權值還原）
output/tables/   論文表格 T1–T12
data/processed/  panel.parquet、panel_dense.parquet、analysis_readiness.csv
```

`audit/health_checks.csv` 每列寫出「要求／觀測值／是否通過」，對應 PRD §8 的驗收標準。

---

## 可重現性

- `data/raw/` 唯讀：分析步驟只寫入 `data/interim/` 與 `data/processed/`。
- 每次抓取保存原始回應、抓取日期與 SHA-256。
- 所有門檻（稀疏度 tier、成交量下限、交易成本假設）寫死於 `config/settings.yaml`，
  不得在分析中臨時調整。
- 隨機種子固定於程式碼中（人工抽驗抽樣、wild cluster bootstrap）。

### 版控範圍

`data/pttweb/`（1.4G）與 `data/twse/`（347M）因體積不進版控；其餘產物（宇宙、
稽核報告、表格）皆進版控。PTT 封存為學術使用，**原始封存不得再散布**。

## 專案結構

```
config/     universe.yaml, settings.yaml
src/
  universe/ build.py（宇宙與市場別）, name_matching.py（碰撞消解）, screen.py（效度）
  ptt/      parse.py（分類與分層）, transform.py（歸屬與窗口）
  market/   collect_finmind.py, collect_twse.py（T86）, collect_exrights.py,
            collect_reduction.py（減資）, collect_shareholding.py,
            normalize.py（權值還原）, validate_prices.py（Yahoo 第三方驗證）
  features/ sessions.py（時段與週對齊）, attention.py（AbnAtt 與稀疏度）,
            imbalance.py（訂單失衡）, build.py（面板）
  analysis/ regressions.py, heterogeneity.py, events.py, sector.py,
            portfolios.py, makeup_days.py, robustness.py, report.py
  run_all.py, audit_data.py
tests/
```

## 相關文件

| 文件 | 用途 |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | 結果摘要（全部為 diagnostic） |
| [`LIMITATIONS.md`](LIMITATIONS.md) | **交付門檻**：研究限制與命名紀律 |
| [`PROJECT.md`](PROJECT.md) | 研究設計與變數定義的單一真相來源 |
| [`REPLICATION.md`](REPLICATION.md) | 複製套件：一條指令、環境、可重現性保證 |
| [`docs/UNIVERSE_PROVENANCE.md`](docs/UNIVERSE_PROVENANCE.md) | 267 檔的來源、選樣規則、已確認缺口 |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | 每項資料採用哪個來源、為何、被否決的選項 |
| [`docs/TEJ_DATA_REQUEST.md`](docs/TEJ_DATA_REQUEST.md) | 授權資料需求與匯入契約 |
| [`audit/adjudication/README.md`](audit/adjudication/README.md) | 歸屬正確率抽驗：六類誤配與修正 |
| `PRD.md` | 產品需求文件（不進版控） |
