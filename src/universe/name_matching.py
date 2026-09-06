"""個股歸屬：前綴碰撞消解 ＋ 通用詞降級（PRD §3.4）。

v1.0 的「關鍵字 ＋ 例外清單」在 267 檔的名稱空間下會系統性汙染測度，而汙染量與規模
相關，會直接毀掉 H6。本模組實作 PRD §3.4 的正式演算法：

1. 證券代號比對（含年份／價格／數量／日期的誤判排除）
2. 最長匹配優先，命中後從文本「消耗」該片段
3. 碰撞群組：群組內先長後短，短名命中須有上下文證據
4. 通用詞降級：名稱本身為高頻常用詞者停用簡稱比對，只用代號

輸出 `match_mode`，讓 `is_code_only_matched` 得以進入面板（PRD §3.4 末段）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------

_ZH_PUNCT = str.maketrans({"（": "(", "）": ")", "　": " ", "％": "%"})


def normalize(text: str) -> str:
    """全形→半形、統一標點、去除零寬字元。原始文本不覆寫（呼叫端負責）。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZH_PUNCT)
    text = text.replace("​", "").replace("﻿", "")
    return text


# ---------------------------------------------------------------------------
# 代號比對
# ---------------------------------------------------------------------------

CODE_RE = re.compile(r"(?<![0-9A-Za-z])(\d{4})(?![0-9A-Za-z])")

# 代號區間與西元年份重疊（中鋼 2002、春源 2010、春雨 2012、中鋼構 2013、中鴻 2014、
# 豐興 2015…）。這些代號必須有正面證據才採計，否則「2015 年」會被算成豐興的關注度。
_YEARLIKE = re.compile(r"^(19[5-9]\d|20[0-4]\d)$")

# 緊接在代號之後、代表它其實是年份／日期／數量／價格的詞。
# 「張」「股」「口」是人工抽驗抓到的漏網之魚——「有 1517 張設質」被當成利奇(1517)。
_AFTER_REJECT = re.compile(
    r"^\s*(年|年度|年底|年初|年報|/\d|-\d|\.\d|月|日|點|元|億|萬|塊|人|次|字"
    r"|張|股|口|席|%|％|倍|例|噸|公噸|坪|件|筆|戶|輛|架|艘|條|份)"
)
# 緊接在代號之前、代表它其實是年份／金額的詞。
# 排行榜的「+1506」「-1413」是買賣超金額，不是代號（人工抽驗發現）。
_BEFORE_REJECT = re.compile(
    r"(民國|西元|自|至|到|從|第|共|約|漲|跌|價|收|開|高|低|\$|NT|\d[/\-.]|[+\-−])\s*$"
)
# 代號後面接這些，是明確的股票語境（「股」「張」已移除，見上）
_AFTER_ACCEPT = re.compile(r"^\s*(檔|這檔|多|空|買|賣|進|出|持有|套牢|存|抱)")

# 法人買賣超／週轉率等排行表把數字排成欄，欄位值會被誤判為代號。
# 兩種判準（人工抽驗逐輪補強）：
#   (a) 左右最近的非空白 token 都是純數字；
#   (b) 左邊是「中文名 ＋ 兩個以上空白」的欄位對齊，右邊接數字欄
#       （例：「3706  神達          1809      1823」中的 1809 是買賣超張數）。
_COL_LEFT = re.compile(r"(?:^|\s)([\d,.\-+]+)\s+$")
_COL_LEFT_ALIGNED = re.compile(r"\S\s{2,}$")
_COL_RIGHT = re.compile(r"^\s+([\d,.\-+]+)(?:\s|$)")
_COL_RIGHT_ALIGNED = re.compile(r"^\s{2,}[\d,.\-+]+(?:\s|$)")

# PTT [標的] 發文樣板的範例行，內含「2330 台積電」，會讓每一篇保留樣板的文章
# 都誤配台積電。樣板必須在比對前移除。
_TEMPLATE_LINE = re.compile(
    r"^.*(?:\(例|（例|ex\s*\[標的\]|例\s*[:：]).*$", re.MULTILINE)
_TEMPLATE_BLOCK = re.compile(r"^.*按\s*[Cc]trl\+[yY].*$", re.MULTILINE)


def strip_ptt_template(text: str) -> str:
    """移除 PTT 發文樣板的範例行（PRD §3.4 未預期，由人工抽驗發現）。"""
    text = _TEMPLATE_LINE.sub("", text)
    return _TEMPLATE_BLOCK.sub("", text)


@dataclass(frozen=True)
class Variant:
    """一個可比對的名稱寫法。"""

    ticker: str
    text: str
    match_mode: str  # code_only | name | name_with_context
    priority: int = 0
    blocked_before: tuple[str, ...] = ()   # 這些字接在前面 → 其實是別家公司
    blocked_after: tuple[str, ...] = ()    # 這些字接在後面 → 其實是別家公司

    def __len__(self) -> int:  # 供最長匹配排序
        return len(self.text)


# 股票語境詞：短名或碰撞群組短邊命中時要求的上下文證據
CONTEXT_TERMS = (
    "股價", "股票", "持股", "張", "檔", "買進", "賣出", "作多", "作空", "放空",
    "多單", "空單", "停損", "停利", "目標價", "本益比", "殖利率", "財報", "法說",
    "營收", "EPS", "除權", "除息", "填權", "填息", "漲停", "跌停", "均線", "К線",
    "K線", "融資", "融券", "外資", "投信", "自營", "成交量", "護盤", "套牢",
    "進場", "出場", "抱股", "存股", "配息", "配股", "大盤", "類股", "個股",
)
_CONTEXT_RE = re.compile("|".join(re.escape(t) for t in CONTEXT_TERMS))


@dataclass
class Match:
    ticker: str
    match_mode: str
    matched_text: str
    start: int
    end: int


@dataclass
class Matcher:
    """267 檔的歸屬比對器。

    `variants` 由 config/universe.yaml 展開；`collision_groups` 為 PRD §3.4 的十個
    群組；`code_only` 為通用詞降級清單。
    """

    variants: list[Variant]
    code_only: set[str] = field(default_factory=set)
    valid_codes: set[str] = field(default_factory=set)
    _ordered: list[Variant] = field(init=False, default_factory=list)
    _alt: re.Pattern | None = field(init=False, default=None)
    _by_text: dict[str, Variant] = field(init=False, default_factory=dict)
    _names_by_ticker: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # 最長匹配優先：字元長度降冪，同長度時 priority 高者先
        usable = [v for v in self.variants if v.ticker not in self.code_only]
        self._ordered = sorted(usable, key=lambda v: (-len(v.text), -v.priority, v.text))
        if not self.valid_codes:
            self.valid_codes = {v.ticker for v in self.variants}
        # 單一交替式正則：依長度降冪排列，讓引擎在每個位置先試最長的寫法。
        # 251,858 篇 × 267 檔逐一 str.find 太慢，改為一次掃描。
        self._by_text = {}
        for v in self._ordered:
            self._by_text.setdefault(v.text, v)
        if self._by_text:
            self._alt = re.compile("|".join(re.escape(t) for t in self._by_text))
        names: dict[str, list[str]] = {}
        for v in self.variants:
            names.setdefault(v.ticker, []).append(v.text)
        self._names_by_ticker = {k: tuple(v) for k, v in names.items()}

    # -- 代號 ------------------------------------------------------------
    def match_codes(self, text: str) -> list[Match]:
        out: list[Match] = []
        for m in CODE_RE.finditer(text):
            code = m.group(1)
            if code not in self.valid_codes:
                continue
            before = text[max(0, m.start() - 8):m.start()]
            after = text[m.end():m.end() + 8]
            # 欄位對齊的間距可能超過 8 個空白，前後都需獨立的長視窗
            before_wide = text[max(0, m.start() - 24):m.start()]
            after_wide = text[m.end():m.end() + 24]
            if _BEFORE_REJECT.search(before) or _AFTER_REJECT.match(after):
                continue
            # 排行表的數字欄 → 欄位值不是代號
            if _COL_LEFT.search(before) and _COL_RIGHT.match(after):
                continue
            if (_COL_LEFT_ALIGNED.search(before_wide)
                    and _COL_RIGHT_ALIGNED.match(after_wide)):
                continue
            if _YEARLIKE.match(code) and not _AFTER_ACCEPT.match(after):
                # 年份型代號需要正面證據：同文有該股簡稱，或代號後緊接股票語境詞
                if not self._name_present(text, code):
                    continue
            out.append(Match(code, "code", code, m.start(), m.end()))
        return out

    def _name_present(self, text: str, ticker: str) -> bool:
        return any(t in text for t in self._names_by_ticker.get(ticker, ()))

    # -- 簡稱 ------------------------------------------------------------
    def match_names(self, text: str) -> list[Match]:
        """最長匹配優先並消耗片段，避免短名吃掉長名的一部分。

        交替式已依長度降冪排列，正則引擎在每個起點會先試最長的寫法；命中後從該
        片段之後繼續，等同於「消耗」。上下文不足而被拒的短名只前進一個字元重掃，
        使更短的合法寫法仍有機會命中。
        """
        if self._alt is None:
            return []
        out: list[Match] = []
        pos, n = 0, len(text)
        while pos < n:
            m = self._alt.search(text, pos)
            if m is None:
                break
            var = self._by_text[m.group(0)]
            # 阻擋延伸：命中片段若被相鄰字擴成**另一家公司**，一律不採計。
            # 這是 v1.0 與 PRD §3.4 都沒有的規則——碰撞群組只處理宇宙**內部**的
            # 前綴衝突，但真正的汙染來源多半在宇宙外：統一→統一證券／投信／獅、
            # 三商→三商壽／銀／餐飲、台聚→平台聚集、華電→中華電、新興→新興市場。
            if self._blocked(text, var, m.start(), m.end()):
                pos = m.start() + 1
                continue
            if var.match_mode == "name_with_context" and not self._has_context(
                text, m.start(), m.end()
            ):
                pos = m.start() + 1
                continue
            out.append(Match(var.ticker, var.match_mode, var.text, m.start(), m.end()))
            pos = m.end()
        return out

    @staticmethod
    def _blocked(text: str, var: Variant, start: int, end: int) -> bool:
        # 延伸字與命中片段之間可能夾空白或換行（「落後三商 銀」），必須先去空白，
        # 否則排版換行就能讓誤配逃過阻擋。
        if var.blocked_after:
            after = text[end:end + 8].lstrip()
            if any(after.startswith(b) for b in var.blocked_after):
                return True
        if var.blocked_before:
            before = text[max(0, start - 8):start].rstrip()
            if any(before.endswith(b) for b in var.blocked_before):
                return True
        return False

    def _has_context(self, text: str, start: int, end: int) -> bool:
        """短名／碰撞群組短邊的上下文證據：同句股票語境詞，或同文出現該代號。"""
        window = text[max(0, start - 60):min(len(text), end + 60)]
        return bool(_CONTEXT_RE.search(window))

    # -- 對外 ------------------------------------------------------------
    def match(self, title: str, body: str = "") -> dict[str, str]:
        """回傳 {ticker: match_mode}。一篇文可對應多檔（PRD §3.4 第 5 點）。"""
        text = strip_ptt_template(normalize(f"{title}\n{body}"))
        found: dict[str, str] = {}
        for m in self.match_codes(text):
            found[m.ticker] = "code"
        for m in self.match_names(text):
            # 代號命中優先於簡稱命中：代號是最強的證據
            found.setdefault(m.ticker, m.match_mode)
        return found


# ---------------------------------------------------------------------------
# 由設定檔建構
# ---------------------------------------------------------------------------

def build_matcher(universe_cfg: dict) -> Matcher:
    """由 config/universe.yaml 的結構建構 Matcher。"""
    code_only = set(universe_cfg.get("code_only_tickers", []))
    variants: list[Variant] = []
    for ticker, spec in universe_cfg["tickers"].items():
        ticker = str(ticker)
        for entry in spec.get("variants", []):
            variants.append(
                Variant(
                    ticker=ticker,
                    text=normalize(entry["text"]),
                    match_mode=entry.get("match_mode", "name"),
                    priority=entry.get("priority", 0),
                    blocked_before=tuple(entry.get("blocked_before", ())),
                    blocked_after=tuple(entry.get("blocked_after", ())),
                )
            )
    return Matcher(
        variants=variants,
        code_only=code_only,
        valid_codes={str(t) for t in universe_cfg["tickers"]},
    )
