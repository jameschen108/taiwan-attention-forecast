"""建立研究宇宙：267 檔 ＋ 市場別 ＋ 上市日 ＋ 名稱變體 ＋ 碰撞規則（PRD §3.1、§3.4）。

輸出：
  data/external/universe.csv          — ticker, name_short, sector, listing_date, ...
  data/external/market_venue.csv      — ticker, venue, effective_from, effective_to
  config/universe.yaml                — 名稱變體與比對模式（Matcher 的輸入）
  audit/universe_provenance.csv       — 選樣規則與來源
  audit/venue_resolution_report.csv   — 每檔的市場別解析證據
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

import yaml

TWSE_BASICS = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_BASICS = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

# PRD §3.4 的十個碰撞群組。群組內先試長名，短名命中須有上下文證據。
COLLISION_GROUPS: dict[str, list[str]] = {
    "統一系": ["1216", "2912"],
    "台塑系": ["1301", "6505"],
    "中鋼系": ["2002", "2013", "2014"],
    "東和系": ["1414", "2006"],
    "三商系": ["2905", "2427"],
    "永豐系": ["1907", "6790"],
    "華新系": ["1605"],
    "中化系": ["1762"],
    "台亞台聚": ["2340", "1304"],
    "長榮系": ["2603"],
}

# PRD §3.4 的通用詞降級起始清單（只用代號比對）。
# 須經 §3.10 量化篩檢驗證後定案；此處為起始值。
GENERIC_DEMOTION: dict[str, str] = {
    # 純常用詞
    "1108": "幸福", "1109": "信大", "1210": "大成", "1203": "味王",
    "1414": "東和", "1418": "東華", "1423": "利華", "1506": "正道",
    "1514": "亞力", "1515": "力山", "1517": "利奇", "1471": "首利",
    "1615": "大山", "1611": "中電", "1609": "大亞", "2010": "春源",
    "2012": "春雨", "6923": "中台", "4994": "傳奇", "3029": "零壹",
    "3130": "一零四", "9937": "全國", "2706": "第一店", "2762": "世界健身-KY",
    "2471": "資通", "2468": "華經", "2390": "云辰", "2423": "固緯",
    # 消費品牌污染
    "2912": "統一超", "8454": "富邦媒", "2727": "王品", "2723": "美食-KY",
    "2707": "晶華", "2704": "國賓", "2705": "六福", "2430": "燦坤",
    "9911": "櫻花", "2911": "麗嬰房", "2903": "遠百", "2908": "特力",
    "2910": "統領", "1432": "大魯閣", "1736": "喬山", "7722": "LINEPAY",
    "6902": "GOGOLOOK",
    # 地名／地理詞
    "1303": "南亞", "2101": "南港",
    # 集團泛稱
    "1402": "遠東新", "2845": "遠東銀", "2915": "潤泰全", "2913": "農林",
}

# 名稱本身不足以指涉該股，需改用更長的寫法（PRD §3.4 長榮系）
NAME_OVERRIDES: dict[str, list[str]] = {
    "2603": ["長榮海運"],          # 長榮大學／長榮航空／長榮桂冠
    "1605": ["華新麗華"],          # 「華新」與華新科衝突
    "1762": ["中化生"],
}

# 額外可接受的寫法（暱稱、俗稱）。只加入無歧義者。
EXTRA_VARIANTS: dict[str, list[str]] = {
    "2330": ["台積電", "台積"],
    "2317": ["鴻海"],
    "2454": [],
    "2002": ["中鋼"],
    "2609": ["陽明海運"],
    "2615": ["萬海"],
}


def _get_json(url: str, cache: Path | None = None,
              max_retries: int = 4) -> list[dict]:
    """抓取並落地快取（PRD §4.3 第 2 點）。

    快取存在時直接讀取——重建不應依賴網路，也讓結果可離線重現。上游偶發的
    IncompleteRead 會重試。
    """
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = urllib.request.urlopen(url, timeout=120).read()
            data = json.loads(raw.decode("utf-8-sig"))
        except Exception as exc:  # noqa: BLE001 — 含 IncompleteRead、URLError
            last_exc = exc
            if attempt < max_retries - 1:
                print(f"    抓取失敗（{type(exc).__name__}），{5 * (attempt + 1)}s 後重試")
                time.sleep(5 * (attempt + 1))
            continue
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data
    raise RuntimeError(f"無法取得 {url}：{last_exc}")


def _roc_to_iso(value: str) -> str:
    """民國日期（1150904）轉 ISO。西元 8 碼（19620209）原樣轉換。"""
    value = (value or "").strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    if len(value) == 7 and value.isdigit():
        return f"{int(value[:3]) + 1911:04d}-{value[3:5]}-{value[5:]}"
    return ""


def build(universe_csv: Path, out_data: Path, out_config: Path, out_audit: Path) -> None:
    rows = list(csv.DictReader(open(universe_csv, encoding="utf-8-sig")))
    assert len(rows) == 267, f"宇宙應為 267 檔，實得 {len(rows)}"

    cache_dir = Path("data/raw/company_basics")
    twse = {r["公司代號"]: r
            for r in _get_json(TWSE_BASICS, cache_dir / "twse_t187ap03_L.json")}
    tpex = {r["SecuritiesCompanyCode"]: r
            for r in _get_json(TPEX_BASICS, cache_dir / "tpex_t187ap03_O.json")}

    out_data.mkdir(parents=True, exist_ok=True)
    out_config.mkdir(parents=True, exist_ok=True)
    out_audit.mkdir(parents=True, exist_ok=True)

    universe_rows, venue_rows, venue_report, unresolved = [], [], [], []
    cfg_tickers: dict[str, dict] = {}

    # ticker → 所屬碰撞群組
    in_group: dict[str, str] = {}
    for gname, members in COLLISION_GROUPS.items():
        for t in members:
            in_group[t] = gname

    for row in rows:
        ticker = row["ticker"].strip()
        name = row["name_short"].strip()
        sector = row["sector"].strip()

        if ticker in twse:
            src, venue = twse[ticker], "TWSE"
            listing = _roc_to_iso(src["上市日期"])
            registration = src["外國企業註冊地國"].strip()
            official = src["公司簡稱"].strip()
        elif ticker in tpex:
            src, venue = tpex[ticker], "TPEx"
            listing = _roc_to_iso(src["DateOfListing"])
            registration = src["Registration"].strip()
            official = src["CompanyAbbreviation"].strip()
        else:
            unresolved.append(ticker)
            continue

        is_ky = registration not in ("－", "－ ", "", "-")
        universe_rows.append({
            "ticker": ticker, "name_short": name, "sector": sector,
            "listing_date": listing, "delisting_date": "",
            "venue": venue, "is_ky": int(is_ky), "registration": registration,
        })
        venue_rows.append({
            "ticker": ticker, "venue": venue,
            "effective_from": listing, "effective_to": "",
        })
        venue_report.append({
            "ticker": ticker, "name_short": name, "venue": venue,
            "source": "TWSE t187ap03_L" if venue == "TWSE" else "TPEx mopsfin_t187ap03_O",
            "official_abbreviation": official,
            "name_matches_official": int(official == name),
            "listing_date": listing, "is_ky": int(is_ky),
        })

        # --- 名稱變體 ---
        base = name.replace("-KY", "").replace("*", "").strip()
        texts: list[str] = []
        if ticker in NAME_OVERRIDES:
            texts.extend(NAME_OVERRIDES[ticker])
        else:
            texts.append(base)
            if name != base:
                texts.append(name)  # 保留 -KY 全寫
        texts.extend(EXTRA_VARIANTS.get(ticker, []))

        # 群組短邊需要上下文；群組內最長名稱可直接比對
        group = in_group.get(ticker)
        if group:
            members = COLLISION_GROUPS[group]
            longest = max(
                (r2["name_short"].strip().replace("-KY", "").replace("*", "")
                 for r2 in rows if r2["ticker"].strip() in members),
                key=len,
            )
            mode = "name" if base == longest and len(members) > 1 else "name_with_context"
        else:
            mode = "name" if len(base) >= 3 else "name_with_context"

        seen, variants = set(), []
        for t in texts:
            if t and t not in seen:
                seen.add(t)
                variants.append({"text": t, "match_mode": mode, "priority": len(t)})

        cfg_tickers[ticker] = {
            "name_short": name, "sector": sector, "venue": venue,
            "listing_date": listing, "is_ky": bool(is_ky),
            "collision_group": group,
            "variants": variants,
        }

    if unresolved:
        raise RuntimeError(f"市場別未解析（P0 阻斷性）：{unresolved}")

    _write_csv(out_data / "universe.csv", universe_rows)
    _write_csv(out_data / "market_venue.csv", venue_rows)
    _write_csv(out_audit / "venue_resolution_report.csv", venue_report)

    cfg = {
        "provenance": {
            "source": "既有爬蟲產物 data/universe_267.csv",
            "selection_rule": "每個 TWSE 產業別依證券代號升冪取前 10 檔；不足 10 檔者取全部",
            "n_tickers": len(cfg_tickers),
            "n_sectors": len({v["sector"] for v in cfg_tickers.values()}),
            "note": "本清單不是指數、不是市值前 267 大、不具市場代表性（PRD §1.4、§11）",
        },
        "collision_groups": COLLISION_GROUPS,
        "code_only_tickers": sorted(GENERIC_DEMOTION),
        "code_only_reason": GENERIC_DEMOTION,
        "tickers": cfg_tickers,
    }
    (out_config / "universe.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    _write_csv(out_audit / "universe_provenance.csv", [{
        "field": k, "value": v} for k, v in cfg["provenance"].items()])

    print(f"universe.csv {len(universe_rows)} 列；"
          f"TWSE {sum(r['venue'] == 'TWSE' for r in universe_rows)}、"
          f"TPEx {sum(r['venue'] == 'TPEx' for r in universe_rows)}；"
          f"-KY {sum(r['is_ky'] for r in universe_rows)}；"
          f"通用詞降級 {len(GENERIC_DEMOTION)} 檔")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    build(Path("data/universe_267.csv"), Path("data/external"),
          Path("config"), Path("audit"))
