# 複製套件（Replication Package）

> PRD §7.5。散布**程式碼、公開資料重建腳本、宇宙清單與名稱比對規則、
> 授權資料的匯入契約**；授權資料本身與 PTT 原始封存**不再散布**。

## 一條指令

```bash
python3 -m src.run_all
```

實測從 `data/raw/` 重建全部輸出耗時 **339 秒**（8 階段，單機）。
結束時印出每階段的模型估計數與驗收門檻通過數。

各階段可獨立重跑：`--only universe audit market ptt panel screen analysis report`

## 環境

```bash
pip install -r requirements.txt   # python >= 3.10
python3 -m pytest tests/ -q       # 113 passed, 1 skipped
```

## 需要另行取得的輸入

| 輸入 | 體積 | 取得方式 | 是否散布 |
|---|---|---|---|
| `data/pttweb/` | 1.4 GB | 既有學術封存 | **否**（PRD §4.1 學術使用，不得再散布） |
| `data/twse/t86/` | 部分 | TWSE `T86` 逐日 | 否（體積） |
| `data/raw/finmind/price/` | — | `python3 -m src.market.collect_finmind --kinds price` | 否（可重抓） |
| `data/interim/ex_rights.csv` | 小 | `python3 -m src.market.collect_exrights` | 否（可重抓） |
| `data/interim/capital_reductions.csv` | 小 | `python3 -m src.market.collect_reduction` | 否（可重抓） |
| `data/interim/shareholding.csv` | 4 MB | `python3 -m src.market.collect_shareholding` | 否（可重抓） |
| `data/raw/yahoo/` | 83 MB | 既有快取，僅供權值還原的第三方驗證 | 否 |
| TEJ 授權資料 | — | 見 `docs/TEJ_DATA_REQUEST.md` | **否**（合約限制） |

**所有公開資料的抓取腳本都在版控內、都支援斷點續傳、都不需 API 金鑰。**
建置指令永遠不會自動觸發抓取（PRD §4.3 第 8 點）。

## 進版控的內容

```
config/          universe.yaml（含名稱變體、碰撞群組、降級與阻擋規則）、settings.yaml
src/             全部管線與分析程式
tests/           113 項回歸測試
scripts/         R15 敏感度測試
audit/           全部稽核與品質報告（不含原始資料）
output/tables/   論文表格 T1–T13、R15、T7M
output/figures/  論文圖 F1–F6
data/external/   universe.csv、market_venue.csv
docs/            UNIVERSE_PROVENANCE、DATA_SOURCES、TEJ_DATA_REQUEST
PROJECT.md LIMITATIONS.md FINDINGS.md README.md
```

`data/` 下的原始與中間產物皆 gitignored（見 `.gitignore`）。

## 可重現性保證

| 項目 | 做法 |
|---|---|
| 隨機性 | 種子固定於程式碼：人工抽驗抽樣 `20260905`、wild cluster bootstrap `20260905`、事件匹配 `20260906` |
| 門檻 | 全部寫死於 `config/settings.yaml`，不得在分析中臨時調整 |
| 原始資料 | `data/raw/` 唯讀；分析只寫 `data/interim/` 與 `data/processed/` |
| 抓取證據 | 每次抓取保存原始回應、抓取日期與 SHA-256 |
| 離線重建 | 公司基本資料等外部端點皆有快取，重建不需連網 |
| 缺值 | 一律維持缺值，不補零不補均值；`audit/full_coverage.csv` 為其證據 |

## 驗收狀態（PRD §8.4）

| 項目 | 狀態 |
|---|---|
| 一條指令可從 raw 重建全部輸出 | ✅ 339 秒 |
| 測試套件全綠 | ✅ 113 passed, 1 skipped |
| PTT 封存 checksum | ✅ 251,858 檔、0.99 GB、總指紋 `e156143357e09783…` |
| 標題解析正確率 > 95% | ✅ 100%（兩套獨立實作的一致率，非人工判讀） |
| 次週報酬嚴格領先 | ✅ 5/5 項；同期相關 0.0705 vs 次週 −0.0030 |
| `formal_main_return` 為 True | ❌ 授權資料未取得（見 `docs/TEJ_DATA_REQUEST.md`） |
| `LIMITATIONS.md` 與結果同步 | ✅ |

**因第三項未通過，本套件的所有數值結果均標記為 `diagnostic`。**

## 引用紀律

任何引用本專案結果的論文、簡報或摘要**必須同時引用 `LIMITATIONS.md`**
（PRD §7.4）。第一條為宇宙選樣偏誤：推論母體只能寫成「本樣本涵蓋之 267 檔個股」
（有效 260 檔），不得寫成「台股」。

`LIMITATIONS.md` §11 有完整的命名紀律對照表。

## 封存指紋

```
audit/ptt_archive_checksums.csv    逐批次 SHA-256 ＋ 總指紋
```

重跑 `python3 -m src.audit_integrity` 可驗證封存未被改動。
指紋與檔案順序無關、與內容完全綁定。
