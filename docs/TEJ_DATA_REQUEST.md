# 授權資料需求（TEJ 或等價來源）

> PRD §7.4 要求的文件。目前 `analysis_readiness.csv` 的 `formal_main_return`
> 為 **False**，唯一原因就是本文列出的欄位未取得——**所有結果因此只能標記為
> `diagnostic`**，不得稱為主結果或初步結論（PRD §11）。

## 為什麼這是硬阻斷

PRD §3.8 的缺值政策不可妥協：未取得的控制變數**維持缺欄位，不得補零或補均值，
也不得以公開替代變數暗中頂替**。正式主表的估計函式在欄位不齊時**直接中止**，
而非降級輸出（PRD §6.2 F6）。

程式已實作此規則：`regressions.fit_twoway(is_diagnostic=False)` 在控制變數不齊時
回傳 `SKIPPED`。診斷模式則剔除涵蓋率不足的欄位並把被剔除者記入該列的 `note`。

## 需求清單

| # | 欄位 | TEJ 資料表 | 頻率 | 用途 | 缺席的後果 |
|---|---|---|---|---|---|
| 1 | `analyst_count`、`forecast_dispersion`、`forecast_revision` | `TWN/ABRKCST1`、`TWN/ABRKCST3` | 事件 | PRD §3.8 控制變數；§5.5 的調節變數之一 | H6 少一個調節維度 |
| 2 | 台灣四因子週報酬（mkt/smb/hml） | `TWN/AFACTORW` | 週 | §5.7 投資組合的風險調整 alpha | 只能報未調整價差 |
| 3 | 無風險利率 | 因子表**未見** rf 欄位 | 週 | 同上 | 需另取（台銀一年期定存利率） |

### 本宇宙的特殊考量

**分析師覆蓋在本宇宙近乎不存在。** PRD §3.8 已預期此事，要求改為：

- 僅用於 `dense` 子樣本（目前 36 檔），或
- 以 `has_analyst_coverage` **二元變數**形式進入全樣本

**不得對無覆蓋股填 0 家分析師後當作連續變數使用。**

**四因子的偏誤**：TEJ 因子由較大市值樣本建構，用於長尾個股的風險調整存在偏誤，
須揭露；另須並報未調整價差（PRD §4.5）。

## 非 TEJ 但同樣缺席的欄位

| 欄位 | 現況 | 可行處置 |
|---|---|---|
| `news_count` | **未取得**。PRD §4.2 明訂為主表**必要**控制項（非 robustness），用來排除「週末 PTT 關注度只是週末新聞的反射」 | 見 `DATA_SOURCES.md` 的鉅亨網方案 |
| `news_sentiment`、`news_impact` | 未建置 | 需先有新聞語料 |
| `selling_expense_to_sales` | MOPS 損益表僅抓到 22/267 檔（`audit/mops_selling_expense_coverage.csv`） | 需補抓 MOPS `ajax_t164sb04`；金融業與 -KY 無此科目，須記為未涵蓋而非零 |
| `sector_pit`（逐年當期產業分類） | 只有 2026 年現況分類 | 需 TWSE 歷年分類快照；§5.11 第 10 項因此標記 SKIPPED |

**命名紀律**：若最終改用推銷費用，欄位名須為 `selling_expense_to_sales`，
不得寫成「廣告密度」或 advertising-to-sales——推銷費用涵蓋銷售人員薪酬與通路費用
（PRD §11）。

## 匯入契約

取得資料後放入 `data/external/`（該目錄的原始匯出為 gitignored），須滿足
PRD §4.6 的資料契約：

| 檔案 | 必要欄位 | 規格 |
|---|---|---|
| `firm_controls.csv` | `ticker, period_end, available_date` ＋ 控制變數 | 比率用小數（12% = 0.12） |
| `factors_weekly.csv` | `week, mkt, smb, hml, rf` | 須同時記錄供應商原始因子名 |

### Point-in-time 防護（PRD §4.4）

- 一律以 **`available_date`**（公開／可用日）合併，**不是** `period_end`。
- 財報類若只有期末日而無發布日，以法定申報期限作為保守的 `available_date`
  （年報為次年 3 月 31 日，證券交易法 §36）。
- **`available_date == period_end` 視為 look-ahead 警訊，匯入檢查必須攔下。**

## 取得後的驗收

`data/processed/analysis_readiness.csv` 的 `formal_main_return` 轉為 True，
且 `audit/health_checks.csv` 的「正式主表可產出」通過。屆時：

1. 以 `is_diagnostic=False` 重跑主表，所有 `diagnostic` 標記解除。
2. `LIMITATIONS.md` §5 須同步改寫。
3. 現有的 diagnostic 結果**不會**因此自動升格——須確認係數在完整控制下的方向與
   顯著性，並在論文中報告兩者的差異。

## 授權紀律（PRD R13）

- 原始匯出置於 gitignored 目錄，**不進版控**。
- 複製套件只散布**程式碼、資料契約與欄位規格**，不散布授權資料本身。
- PTT 封存為學術使用，原始封存同樣不得再散布。
