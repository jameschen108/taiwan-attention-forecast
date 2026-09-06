# 資料來源決策登記簿

> PRD §7.4 的 `PHASE_D_DATA_SOURCES.md`。記錄每一項資料**最後採用了哪個來源、
> 為什麼、以及被否決的選項為何被否決**——這是複製者最需要、也最容易在專案結束後
> 遺失的資訊。

## 總表

| 資料 | 採用來源 | 為何不用 PRD 原訂來源 | 落地位置 |
|---|---|---|---|
| PTT 股板文章 | 既有封存 `data/pttweb/` | 同 PRD | 251,858 篇，2015-04~2025-01 |
| 個股日成交 | **FinMind** `TaiwanStockPrice` | PRD 訂 TWSE `STOCK_DAY`，需 267×120 = 32,040 次請求；FinMind 一檔一次共 267 次 | `data/raw/finmind/price/` |
| 三大法人 | **既有 TWSE `T86` 封存** | FinMind 同資料但受免費額度限制；T86 封存已涵蓋 2015-01~2024-12 全主樣本 | `data/twse/t86/` |
| 除權息參考價 | **TWSE `TWT49U`**（全市場區間） | PRD 未指定具體端點；逐檔抓需 267 次，此端點 45 次 | `data/interim/ex_rights.csv` |
| **減資參考價** | **TWSE `TWTAUU`** | **PRD 完全沒有這一項**，見下方「重大偏離」 | `data/interim/capital_reductions.csv` |
| 發行股數、外資持股 | **TWSE `MI_QFIIS`**（全市場單日） | FinMind 逐檔受額度限制；此端點每抽樣日一次 | `data/interim/shareholding.csv` |
| 交易日曆 | 由 FinMind 價格的實際成交日推導 | 同 PRD §4.1 | `data/interim/trading_days.csv` |
| 公司基本資料 | TWSE `t187ap03_L` ＋ TPEx `mopsfin_t187ap03_O` | 同 PRD | `data/raw/company_basics/` |
| 價格交叉驗證 | 既有 Yahoo 快取 `data/raw/yahoo/` | PRD 稱之為 `checkpoint.json`，實際結構為逐檔逐年 CSV | 見下方 |
| 新聞 | **未採用** | 既有 Yahoo 新聞快取不可用，見下方 | — |
| 分析師、四因子 | **未取得** | 需 TEJ 授權 | 見 `TEJ_DATA_REQUEST.md` |

## 重大偏離：減資（PRD 未涵蓋）

PRD §4.1 只列了「除權息參考價」。實作後發現**除權除息計算結果表不含減資**，
而減資會讓股價機械性跳升——長榮 2603 於 2022-09-19 減資，只用除權息表還原時
該日「上漲 109%」，而台股漲跌幅上限為 10%。

補上 TWSE 股票減資恢復買賣參考價格（`TWTAUU`，394 事件／261 檔）後，主樣本內
`|日報酬| > 11%` 的觀測由 **127 筆／59 檔降至 79 筆／20 檔**，且殘留者 82% 落在
TWSE 上市日之前（興櫃期間無漲跌幅限制），不進入分析面板。

**這個錯誤是由既有 Yahoo 價格快取的交叉比對揭露的。** 若沒有那批資料，錯誤會靜默
留在報酬序列裡。

## Yahoo 快取的兩份資料：一份可用、一份不可用

`data/raw/yahoo/<ticker>/` 下有 `price/` 與 `news/` 兩個目錄。

### price/ — 可用，且是關鍵的第三方驗證

2011-09 ~ 2026-09，267 檔齊全。**未還原股利但已還原股本變動**，與本專案的
`adj_close`（兩者都還原）互補，因此可作為獨立對照：

- 非除權息日的日報酬應**幾乎相等** → 驗證股本變動的還原
- 除權息日應**系統性分歧** → 驗證股利確實有還原

結果：TWSE 上市後 **255/259 檔通過**（`audit/price_validation_yahoo.csv`）。
`price_adjustment_verified` 因此為 True，且是由第三方證據而非自我宣告支持。

### news/ — 不可用

| 問題 | 實測 |
|---|---|
| **時戳全空** | `time` 欄位在 12,365 列中 **100% 為空**（`source` 亦然） |
| **不是歷史封存** | 267 檔合計僅 12,365 則（每檔中位數 38），多檔卡在 100 則上限 |

沒有時戳就無法指派到週，而本研究的一切建立在「週末 vs. 週間」的時間切分上。
連只算 `news_count` 都做不到。

**重抓也無法解決**：實測 Yahoo 個股新聞頁的無限捲動在「前天」就耗盡
（台積電 169 則、scrollHeight 停在 4,930px），背後 API 有 crumb 反爬（403）。
Yahoo 只提供約 3 天的視窗，**無法還原 2015–2024**。

### 若日後要啟用新聞管道

已驗證可行的替代來源為**鉅亨網**：

```
api.cnyes.com/media/api/v1/newslist/category/headline?startAt=&endAt=&limit=30&page=N
```

- `publishAt` 為**精確 unix 時戳**；另有 `title`、`summary`、`content`
- 日期區間查詢，2015 年起可得；`headline` 類別 2024 年 33,481 則 → 11 年約 33 萬則
- 不需認證；`limit` 實測鎖在 30，故約需 12,300 次請求
- `stock` 欄位為空，需沿用本專案既有的名稱比對器

**注意**：鉅亨網是**媒體供給端**，與 PTT 的**投資人需求端**性質不同。這正是
PRD §5.9 要比較的東西，但論文中不可稱它為「Yahoo 的替代品」。且大量清單型貼文
的問題（§8.2）在新聞語料中大概率同樣存在（「三大法人買賣超排行」這類稿）。

## 被否決的選項

| 選項 | 否決理由 |
|---|---|
| TWSE `STOCK_DAY` 逐檔逐月 | 267 × 120 = 32,040 次請求，約 26 小時 |
| FinMind `TaiwanStockInstitutionalInvestorsBuySell` | 免費額度限制；且既有 T86 封存已完整涵蓋主樣本 |
| FinMind `TaiwanStockDividendResult` | 逐檔 267 次；TWSE 全市場端點只需 45 次，且含減資對應表 |
| Google Trends | 本宇宙長尾個股多半低於回報門檻，全樣本不可行（PRD §3.10 已載明） |
| Yahoo 新聞 | 見上 |

## 速率限制與禮貌

- FinMind：免費層無 token，實測約 300 次／小時；程式以 402/429 指數退避處理。
- TWSE：無公開限制，程式以 2–4 秒隨機延遲；偶發 `IncompleteRead` 已加重試。
- 所有抓取皆**斷點續傳**，且原始回應落地（含 SHA-256），重跑不重抓。
- 建置指令**永遠不會**自動觸發抓取（PRD §4.3 第 8 點）。
