# taiwan-attention-forecast

由 [`taiwan-attention-long-tail`](https://github.com/jameschen108/taiwan-attention-long-tail)
延伸而來的**預測管線**專案：保留既有研究資產，另建可時間外推的未來相對強弱預測。

- 改造規格：[`docs/FORECAST_SPEC.md`](docs/FORECAST_SPEC.md)

## 預測管線（P0–P2）

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 預測專用時點資料（研究面板不動）
.venv/bin/python -m src.forecast.build_pit_data

.venv/bin/python -m src.forecast.run          # P1：Ridge A/B + 評估
.venv/bin/python -m src.forecast.run_p2         # P2：機率、HGB、回測（需先跑 P1）
.venv/bin/python -m src.forecast.predict        # P3：最新週前瞻預測（append-only ledger）
.venv/bin/python -m src.forecast.predict --backfill-weeks 12  # 回放最近 12 週驗證運作
.venv/bin/python -m src.forecast.run_p4          # P4：C/D/B vs A（已登記）+ E vs D（探索性）
.venv/bin/python -m src.forecast.collect_month_revenue  # 可選：抓取 FinMind 月營收
.venv/bin/python -m pytest tests/ -q            # 145 passed, 1 skipped（Python 3.12 venv）
```

產出：

| 路徑 | 內容 |
|---|---|
| `data/forecast/panel_pit.parquet` | 預測專用時點面板（無持股 bfill、上市後才進滾動窗） |
| `data/forecast/features.parquet` | 時點特徵（無未來標籤欄） |
| `data/forecast/labels.parquet` | 1w 超額報酬 vs 0050 |
| `data/forecast/predictions.parquet` | 可追溯樣本外預測（含 `generated_at` / `horizon` / `model_version`） |
| `output/forecast/VERDICT.md` | A/B 主比較結論 |
| `output/forecast/VERDICT_P2.md` | P2 回測、0050 基準、成交率診斷 |
| `output/forecast/PROSPECTIVE_STATUS.md` | P3 最近 12 週預測／跳過可追溯性 |
| `data/forecast/prospective/predictions.parquet` | 不可改寫的前瞻預測 ledger |
| `output/forecast/evaluation_weekly.csv` | 逐週 Rank IC / ΔIC |
| `output/forecast/VERDICT_P4.md` | P4 增量來源比較（C/D/B vs A；E vs D 探索性） |
| `output/forecast/experiment_registry.csv` | §17.2 登記門檻逐比較判定 |

P1 預先登記成功條件見規格 §17.2：B 相對 A 的 ΔIC 需穩定為正。  
目前評估結果寫在 `output/forecast/VERDICT.md`（實驗完成 ≠ 宣稱可獲利）。

規格全文：[`docs/FORECAST_SPEC.md`](docs/FORECAST_SPEC.md)。  
下方文件仍描述**研究繼承狀態**；預測實作以 `FORECAST_SPEC` 為準，並與研究管線分離。

---

# 台股長尾關注度與報酬可預測性（研究繼承）

把 Li, Liu, Ye, Zhao & Zhao, *"It Depends on When You Search"*（MIS Quarterly）的
「非交易時段關注度預測次週報酬」研究設計移植到台灣股市，並利用一個橫跨 28 產業、
267 檔、關注度極度不均的長尾樣本，把「哪一種股票的關注度才有預測力」這個原論文
無法回答的問題，變成識別**資訊處理**與**價格壓力**兩條機制管道的工具。

> **所有數值結果目前均標記為 `diagnostic`。** 授權控制變數未取得，
> `analysis_readiness.csv` 的 `formal_main_return` 為 False。依 研究規格 §11，
> 不得稱為主結果或初步結論。
>
> **引用本專案任何結果時必須同時引用 [`LIMITATIONS.md`](LIMITATIONS.md)。**
> 第一條為宇宙選樣偏誤：推論母體只能寫成「本樣本涵蓋之 267 檔個股」，
> **不得寫成「台股」**。

---

## 1. 現況一覽

| 項目 | 狀態 |
|---|---|
| 管線 | 8 階段，一條指令從 raw 重建全部輸出，**實測 339 秒** |
| 測試 | **145 passed**, 1 skipped |
| 面板 | 123,828 列 × **260 檔** × 506 週（2015-05-03 ~ 2025-01-05） |
| 驗收門檻 | **18/20 通過**（未通過兩項同一根因：授權資料未取得） |
| 產出 | 表格 T1–T13 ＋ R15 ＋ T7M、圖 F1–F6、17 份稽核報告 |

未通過的兩項：

| 門檻 | 要求 | 實得 | 根因 |
|---|---|---|---|
| 正式主表可產出 | `formal_controls_available` = True | False | 缺 `news_count`、分析師、四因子 |
| `dense` 檔數 | ≥ 40 檔 | 36 | PTT 對長尾覆蓋不足 |

兩項都由**新聞管道**解決（研究規格 §4.2／§8.2 明訂未達門檻即觸發此備案）。
可行方案已驗證並記於 [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)。

---

## 2. 資料

| 資料 | 來源 | 涵蓋 |
|---|---|---|
| PTT 股板文章 | 既有封存 | **251,858 篇**，2015-04 ~ 2025-01，118 個月零缺口 |
| 個股日成交 | FinMind | 267 檔，2014-01 ~ 2025-03 |
| 三大法人（股數） | 既有 TWSE `T86` 封存 | 2,609 個交易日，2015-01 ~ 2024-12 |
| 除權息參考價 | TWSE `TWT49U` | 9,815 事件 |
| **減資參考價** | TWSE `TWTAUU` | 394 事件（**原規格未涵蓋，見 §5.1**） |
| 發行股數、外資持股 | TWSE `MI_QFIIS` | 每 5 交易日抽樣 |
| 價格交叉驗證 | 既有 Yahoo 快取 | 2011-09 ~ 2026-09 |

抓取腳本全部在版控內、支援斷點續傳、不需 API 金鑰。
建置指令**永遠不會**自動觸發抓取。來源選擇與被否決的選項見
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)。

---

## 3. 執行

```bash
pip install -r requirements.txt        # python >= 3.10
python3 -m src.run_all                 # 從 raw 重建全部輸出（339 秒）
python3 -m pytest tests/ -q            # 139 passed, 1 skipped
```

各階段可獨立重跑（除錯用）：

```bash
python3 -m src.run_all --only panel analysis report
```

階段：`universe` → `audit` → `market` → `ptt` → `panel` → `screen` → `analysis` → `report`

首次執行前的資料收集（與分析管線分離）：

```bash
python3 -m src.market.collect_finmind --kinds price   # 267 檔日成交
python3 -m src.market.collect_exrights                # 除權息參考價
python3 -m src.market.collect_reduction               # 減資參考價
python3 -m src.market.collect_shareholding            # 發行股數與外資持股
python3 -m src.audit_integrity                        # 封存 checksum（讀 1.4 GB）
```

---

## 4. 結果摘要

完整版見 [`FINDINGS.md`](FINDINGS.md)。**以下全部為 `diagnostic`。**

### 4.1 最強的一組證據：測度效度

| 檢定 | 本研究 | 原論文 |
|---|---|---|
| `corr(整週, 週間)` | **0.939** | 0.90 |
| `corr(整週, 週末)` | **0.403** | 0.40 |

測度換成 PTT 發文數（不是 Google 搜尋量）、市場換成台灣、樣本換成長尾，
**兩個窗口與整週關注度的相關性型態幾乎逐位數重現**。這支持「週末關注度是一個
與週間關注度不同的建構」——原論文設計的前提在台灣成立。

### 4.2 H1 週末效果：方向對，強度不足

雙向固定效果，個股與週雙重 cluster：

| 變數 | 係數 | t | p |
|---|---|---|---|
| `abn_attention_weekend` | **0.000460** | **1.80** | **0.073** |
| `abn_attention_weekday` | 0.000230 | 1.08 | 0.278 |

週末係數為週間的 2 倍且為正，方向與 H1 一致，**但未達 5% 顯著**。
交易時段切法（intraday / non_trading）同樣不顯著。

另一個旁證：關注度與**同期**報酬的相關性 0.0705、與**次週**報酬 −0.0030，
相差 23 倍（`audit/lead_lag_verification.csv`）。關注度確實跟當週價格一起動，
但不預測下週。

### 4.3 H3 機制：唯一乾淨的一組

| 應變數 | 週間關注度 | 週末關注度 |
|---|---|---|
| 非三大法人訂單失衡 | **−0.00286（t = −4.12）** | 0.00027（t = 0.46） |
| 異常周轉率 | **0.00520（t = 8.14）** | **0.00254（t = 5.16）** |

依 研究規格 §5.3 的判讀表，「週末 → Turnover 顯著、週末 → ROI 不顯著」落在
**資訊處理管道**那一格。但由於 H1 本身未達顯著，**這個機制證據懸空**——
沒有需要被解釋的報酬效果。

### 4.4 H6 異質性、H7 起始事件、H8 產業外溢

- **H6**：五個調節變數中，`amihud` 交互項顯著為負（t = −3.05，符合 H6 預測），
  但 `turnover` 顯著為**正**（相反方向），其餘三個不顯著。**方向不一致。**
- **H7**：未匹配的 CAR 路徑呈現漂亮的「衝高後回吐」，但平行趨勢**未通過**
  （價格先動、討論才出現；傾向分數顯示前一週異常報酬使起始勝算比達 5.55）。
  以同週匹配 ＋ 事前報酬 caliper 處理後，事前差異消除、**事後每一週皆不顯著**
  ——原型態可由價格動能完全解釋，**H7 不成立**。
- **H8**：產業外溢與產業內相對關注度皆不顯著。

### 4.5 H4 補班日、投資組合

- **補班日**：全期間補班星期六僅 9 天，主樣本內 **8 個事件週**，且 2018-12 後歸零。
  wild cluster bootstrap（群集於週）p = 0.039，但**有效群集數 = 8**，
  依 研究規格 §5.10 備案 (c) 只能定位為**探索性訊號**。
- **投資組合**：等權未篩選成本前週價差 +0.098%（t = 1.99），
  但按**實際換手率**（33–43%）計的週成本為 0.57–0.69%，**成本後全部深度為負**
  （t ≈ −5 至 −10）。依 研究規格 §5.7，不含成本的多空價差不得作為主要結論。

### 4.6 與原論文的逐項對照（T13）

完整表格見 [`output/T13_paper_comparison.md`](output/T13_paper_comparison.md)。
13 個對照項中：**型態一致 8 項、部分不同 2 項、明確不同 2 項**。

**在同一尺度上（論文的應變數也標準化），本研究的週末係數比論文大 1.58 倍：**

| | β | SE | t |
|---|---|---|---|
| 論文 ASVI67（Table 3a 規格 4） | +0.0068 | 0.0021 | +3.24\*\*\* |
| 本研究 週末（雙重 cluster） | **+0.0108** | 0.0059 | +1.82 |
| 本研究 週末（**論文的 cluster 方式**） | +0.0108 | 0.0046 | **+2.36\*\*** |

**差別不在效果大小，在推論標準。** 雙重 cluster 的標準誤是個股 cluster 的
1.30 倍，5% 門檻因此翻轉。本專案採較保守的做法（研究規格 §5.1 C9，理由是 267 檔
同時暴露於相同的週別市場衝擊）——這是方法論選擇，不是資料失敗。

三項與論文明確不同或無對應：

1. **週間關注度也強烈預測周轉率**（本研究 t=8.20 且強於週末；論文不顯著）
   ——符合台灣散戶當沖比重高的結構。
2. **投資組合成本後全毀**（−25.2%/年）。論文**未報告成本後報酬**；
   S&P 500 的交易成本低一個數量級，這個問題不會浮現。
3. **訂單失衡符號相反**，但論文用 Boehmer et al. 的散戶訂單失衡、
   本研究用非三大法人殘差（含大戶），**測度不同不可直接比**。

橫斷面型態則一致：論文的效果在 S&P 500 內的小型半邊（3.36%\*\*）而非大型半邊
（0.53 ns）；本研究 sparse t=+2.01、dense t=−0.90。

### 4.7 對多重檢定的誠實處理

`sparse` 子樣本的 H1 係數 t = 2.01、`amihud` 交互項 t = −3.05、補班日 p = 0.039
——這三個都擦到 5%。但在跑了 **81 個模型**（`audit/model_status.csv`）之後，
少數幾個落在 5% 內是預期中的事，且方向彼此不一致。

**本專案的結論是：證據不足以區分兩條管道。** 這正是 研究規格 §2.3 明確允許的結果——
「成功標準不是複製出顯著結果，而是用足以區分兩個管道的證據回答問題」。

---

## 5. 過程中修正的重大錯誤

這些不是研究規格預期到的問題，而是實作與驗證過程中發現的。每一項都有回歸測試鎖住。

### 5.1 減資未還原（報酬序列的實質錯誤）

TWSE 除權除息計算結果表**不含減資**。長榮 2603 於 2022-09-19 減資，只用除權息表
還原時該日「上漲 109%」——台股漲跌幅上限為 10%，這在定義上不可能。

補上減資表後，主樣本內 `|日報酬| > 11%` 由 **127 筆／59 檔降至 79 筆／20 檔**，
殘留者 82% 落在 TWSE 上市日之前（興櫃無漲跌幅限制），不進入面板。

**這個錯誤是由既有 Yahoo 價格快取的交叉比對揭露的。** 兩個來源的還原性質互補
（Yahoo 還原股本變動但不還原股利），因此構成 研究規格 §3.6 要求的第三方驗證：
上市後 **255/259 檔通過**，`price_adjustment_verified` 因此為 True。

### 5.2 大量清單型貼文（測度汙染）

覆蓋熱圖露出 2018–19 一條「幾乎所有個股同時有討論」的假垂直帶。追查後是
「加權股價指數成分股暨市值比重」「非擔任主管職務之全時員工薪資」「本週小小程式
選股」這類貼文，一篇列出數十至 **242 檔**代號。

**2,323 篇（佔有配對文章 3.7%）貢獻了 42.7% 的配對列。** 標記為
`is_bulk_listing` 並自主規格排除，門檻寫入設定檔並做敏感度測試（R15）。

### 5.3 歸屬正確率 77.6% → 95%

分層抽驗 550 筆發現六類系統性誤配：

| 機制 | 例 |
|---|---|
| 簡稱擴成集團旗下另一家公司 | 統一 20 筆中 18 筆是統一證券／投信／獅；三商 17 筆全是三商壽／銀／餐飲 |
| 簡稱被切在更長的詞中間 | 「中**華電**信」→ 華電；「平**台聚**集」→ 台聚 |
| 簡稱是普通名詞 | 「新興市場」「銷售冠軍」「冠軍教練」 |
| **PTT 發文樣板** | `[標的]` 格式範例「(例 2330 台積電)」誤配台積電 |
| **排行表數字欄** | 「大成鋼　1410　29　宇環」中的 1410 是買賣超張數 |
| 量詞漏列 | 「累計確診 2433 例」「持有 1762 噸黃金」「1517 張設質」 |

修正後分析用配對列 105,199 → 97,986（−6.8%）。**修正幅度與規模相關**
（台積電 −7% vs. 長尾個股 −60~95%），會影響 H6 的橫斷面比較，
`is_code_only_matched` 已進入面板供對照。詳見
[`audit/adjudication/README.md`](audit/adjudication/README.md)。

### 5.4 其他

- `listing_age_years` 與雙向固定效果**共線**（上市年資是週與上市日的線性組合），
  已自 `BASE_CONTROLS` 移除。
- `amihud` 原本跨個股邊界做 `pct_change`。
- 交易時段窗口誤用日曆欄位（`intraday`／`non_trading` 全為空）。
- 效度篩檢的代號-簡稱相關性檢定缺**檢定力守門**：長尾個股簡稱只命中個位數次時
  相關係數趨近 0，會把長尾股系統性誤判為可疑並全數降級——而降級又與規模相關，
  正好製造出 H6 要檢定的那種偏誤。

---

## 6. 與研究規格的重大偏離

完整清單見 [`LIMITATIONS.md`](LIMITATIONS.md)。

| # | 原規格假設 | 實際 |
|---|---|---|
| 1 | 宇宙含上市與上櫃 | **267 檔全為 TWSE 上市，TPEx 0 檔**。整條雙市場工作線不適用 |
| 2 | 樣本期 2011–2026 | PTT 封存為 2015-04 ~ 2025-01，主樣本 **2015-05 ~ 2024-12** |
| 3 | 267 檔全數有觀測 | **7 檔上市日晚於樣本結束**，有效 **260 檔** |
| 4 | 推文有作者與時戳 | **無**。1,103 萬則推文無法定位，`low_effort` 依研究規格備案改為「不含推文」 |
| 5 | 分類清單八類 | 實測缺「情報」（9.4%）等；研究規格列的「問卷」僅佔 0.02% |
| 6 | 補班日可識別 | 僅 **8 個事件週**且 2018-12 後歸零；群集數沒有隨個股數增加 |
| 7 | Yahoo 新聞可用 | **時戳 12,365 列全空**，且僅約 3 天視窗，無法還原歷史 |
| 8 | 除權息即足以還原 | **需另加減資表**（見 §5.1） |

---

## 7. 資料契約

| 檔案 | 必要欄位 | 單位與鍵值 |
|---|---|---|
| `data/external/universe.csv` | `ticker, name_short, sector, listing_date, delisting_date, venue, is_ky` | 267 列；`ticker` 唯一 |
| `data/external/market_venue.csv` | `ticker, venue, effective_from, effective_to` | 閉區間、不重疊 |
| `data/interim/ptt_matches.parquet` | `ticker, timestamp, week, category, is_reply, match_mode, effort, window, session, is_bulk_listing` | 同文多檔輸出多列 |
| `data/interim/market_daily.parquet` | `ticker, date, venue, close, adj_close, volume, inst_buy, inst_sell` | 三個量皆為**股數** |
| `data/interim/ex_rights.csv` | `ticker, date, before_price, after_price, factor` | factor **< 1** |
| `data/interim/capital_reductions.csv` | 同上 ＋ `reason` | factor **> 1** |
| `data/interim/trading_days.csv` | 實際開市日、`is_makeup_saturday` | 由實際成交日推導 |
| `data/processed/panel.parquet` | ticker×week 非平衡面板 | **主結果用** |

格式錯誤的輸入會中止建置；**缺值一律維持缺值**，不補零也不補均值（研究規格 §3.8）。
`audit/full_coverage.csv` 是這條規則的證據。

---

## 8. 產出

```
audit/            17 份稽核報告（覆蓋率、效度篩檢、抽驗、健檢、權值驗證、封存指紋）
output/tables/    T1–T12、R15 敏感度、T7M 匹配事件研究
output/figures/   F1 覆蓋熱圖、F2 規格曲線、F3 β×覆蓋度、F4 CAR、F5 投組、F6 匹配前後
data/processed/   panel.parquet、panel_dense.parquet、analysis_readiness.csv
```

`audit/health_checks.csv` 每列寫出「要求／觀測值／是否通過」，對應 研究規格 §8 的驗收標準。
`audit/model_status.csv` 列出全部 81 個模型與其狀態（75 OK、6 SKIPPED 並註明原因）。

---

## 9. 可重現性

- `data/raw/` 唯讀；分析只寫 `data/interim/` 與 `data/processed/`。
- 每次抓取保存原始回應、抓取日期與 SHA-256；PTT 封存另有總指紋。
- 所有門檻（稀疏度 tier、成交量下限、清單型貼文門檻、交易成本假設）
  寫死於 `config/settings.yaml`，**不得在分析中臨時調整**。
- 隨機種子固定於程式碼：抽驗抽樣、wild cluster bootstrap、事件匹配。
- 外部端點皆有快取，重建不需連網。

### 版控範圍

`data/pttweb/`（1.4 GB）與 `data/twse/`（347 MB）因體積不進版控；
其餘產物（宇宙、稽核報告、表格、圖）皆進版控。
PTT 封存為學術使用，**原始封存不得再散布**。

---

## 10. 專案結構

```
config/     universe.yaml（名稱變體、碰撞群組、降級與阻擋規則）、settings.yaml
src/
  universe/ build.py（宇宙與市場別）, name_matching.py（碰撞消解）,
            screen.py（效度篩檢）, review_evidence.py（抽驗證據）
  ptt/      parse.py（分類與分層）, transform.py（歸屬與窗口）
  market/   collect_finmind.py, collect_twse.py（T86）, collect_exrights.py,
            collect_reduction.py（減資）, collect_shareholding.py,
            normalize.py（權值還原）, validate_prices.py（第三方驗證）
  features/ sessions.py（時段與週對齊）, attention.py（AbnAtt 與稀疏度）,
            imbalance.py（訂單失衡）, build.py（面板）
  analysis/ regressions.py, heterogeneity.py, events.py, matched_events.py,
            sector.py, portfolios.py, makeup_days.py, robustness.py,
            paper_comparison.py（T13）, report.py, coverage_reports.py,
            figures.py
  run_all.py, audit_data.py, audit_integrity.py
scripts/    sensitivity_bulk_listing.py（R15）
tests/      113 項回歸測試
```

---

## 11. 相關文件

| 文件 | 用途 |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | 結果詳述（全部為 diagnostic） |
| [`LIMITATIONS.md`](LIMITATIONS.md) | **交付門檻**：研究限制與命名紀律 |
| [`PROJECT.md`](PROJECT.md) | 研究設計與變數定義的單一真相來源 |
| [`REPLICATION.md`](REPLICATION.md) | 複製套件：一條指令、可重現性保證、驗收狀態 |
| [`docs/UNIVERSE_PROVENANCE.md`](docs/UNIVERSE_PROVENANCE.md) | 267 檔的來源、選樣規則、已確認缺口 |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | 資料來源決策登記簿與被否決的選項 |
| [`docs/TEJ_DATA_REQUEST.md`](docs/TEJ_DATA_REQUEST.md) | 授權資料需求與匯入契約 |
| [`audit/adjudication/README.md`](audit/adjudication/README.md) | 歸屬正確率抽驗：六類誤配與修正 |

---

## 12. 下一步

### 研究管線

依邊際價值排序：

1. **新聞管道**（鉅亨網，方案已驗證）——一次解掉兩個未通過的驗收門檻：
   `news_count` 是 研究規格 §4.2 明訂的主表必要控制項，缺它 H1 的資訊解讀有未排除的
   替代假說；且新聞覆蓋率遠高於 PTT，可補足 `dense` 檔數。§5.9 的需求端 vs. 供給端
   比較**在 H1 邊緣時特別有價值**。
2. **人工複核 LLM 判讀**——本專案的抽驗由 LLM 執行，研究規格 §8.1 要求的是人工。
   證據已備妥（550 筆逐筆命中片段與上下文），建議優先複核 `ambiguous` 案例。
3. **取得 TEJ 授權資料**——解除全部 `diagnostic` 標記。
4. H7 若要復活，需要一個**外生**的關注度衝擊，改匹配規格沒有用。

### 預測管線（FORECAST_SPEC §16）

P0–P2 已實作並重跑時點面板；結論見 `output/forecast/VERDICT*.md`。

| 階段 | 內容 | 狀態 |
|---|---|---|
| P3 | 前瞻紀錄（`predict.py`、每週快照、不可改寫預測史） | **已實作**（12 週回放 `operational_ready=true`；真實前瞻待 PTT 更新） |
| P4 | 擴充來源（新聞、月營收、E=月營收+PTT；`experiment_registry.csv`） | **已實作**（D 登記未通過；E vs D 探索性） |

P3 需 PTT 封存更新至可產生「當週」預測；P4 與研究管線新聞/TEJ 工作可部分共用資料來源，但驗收規格獨立。

---

## 命名紀律（研究規格 §11 節錄）

| 正確用語 | 不可用 |
|---|---|
| 本樣本涵蓋之 267 檔個股（有效 260 檔） | 台股、台灣上市公司 |
| 非三大法人訂單失衡 | 散戶訂單失衡、retail order imbalance |
| 異常 PTT 關注度 | ASVI、SVI、搜尋量 |
| 銀行與保險業子樣本 | 金融業（本宇宙無任何金控） |
| 外資及陸資持股比率 proxy | 機構持股比例 |
| ex-post universe 之機械年化多空價差 | 年化績效、可投資報酬 |
| diagnostic 結果 | 主結果、初步結論 |
