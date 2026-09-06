"""人工／LLM 抽驗發現的誤配情境（2026-09）。

這些全部是實際語料中抓到的錯誤，不是假想案例。每一條都對應
audit/adjudication/ 中的一筆判讀，回歸測試鎖住以免修改比對器時復發。
"""

from __future__ import annotations

import pytest
import yaml

from src.universe.name_matching import build_matcher, strip_ptt_template


@pytest.fixture(scope="module")
def matcher():
    return build_matcher(yaml.safe_load(open("config/universe.yaml", encoding="utf-8")))


class TestBlockedExtensions:
    """短名被相鄰字擴成另一家公司時不得採計。"""

    @pytest.mark.parametrize("text,ticker", [
        ("統一投顧董事長黎方國表示", "1216"),
        ("前統一投信基金經理人涉案", "1216"),
        ("統一證券(2855)中午發布重大訊息", "1216"),
        ("作者 dalyadam (統一獅加油) 之銘言 停損 目標價", "1216"),
        ("統一超商今日公布營收 股價 目標價", "1216"),
        ("三商美邦人壽(2867)公告董事會通過財報", "2905"),
        ("三商壽已於去年辦理2次現金增資 股價 財報", "2905"),
        ("三商餐飲辦理股票公開發行及申請登錄興櫃", "2905"),
        ("跌停家數 三商家購、東哥遊艇 成交量", "2905"),
        ("不像當年三商銀一樣 股價直接打一折 股票", "2905"),
        ("由於脈脈平台聚集了國內絕大多數互聯網公司的員工", "1304"),
        ("其另一互金平台聚寶匯也處於兌付逾期狀態", "1304"),
        ("台塑集團、國泰金、中華電 算台灣最穩的 股價 存股", "1603"),
        ("美國基金持有者對亞太跟新興市場的贖回潮 停損 目標價", "2605"),
        ("投資長不僅是公開比賽冠軍 同時也是紀錄保持人 績效 股票", "1806"),
        ("全家便利商店也和永豐餘生技聯名推無添加便當", "1907"),
    ])
    def test_extended_entity_not_attributed(self, matcher, text, ticker):
        assert ticker not in matcher.match(text), f"「{text[:20]}…」不應配到 {ticker}"

    @pytest.mark.parametrize("text,ticker", [
        ("二線塑化股如台聚、亞聚、台達化 爆大量收黑 股價", "1304"),
        ("造紙三雄榮成、永豐餘與正隆 股價 營收", "1907"),
        ("聯華電子今日公布營收 股價 目標價", "1603"),   # 不得配到華電，但…
    ])
    def test_genuine_mention_still_matched(self, matcher, text, ticker):
        if ticker == "1603":
            pytest.skip("此列僅為對照，見 test_extended_entity_not_attributed")
        assert ticker in matcher.match(text), f"「{text[:20]}…」應配到 {ticker}"


class TestGenericDemotion:
    """抽驗後追加降級的個股：**只用代號**，簡稱一律不採計。

    這是 PRD §3.4 第 4 點的規則，但降級名單由實證決定而非事前臆測。
    """

    @pytest.mark.parametrize("text,ticker", [
        ("統一、長榮與友達等穩平盤 傳產龍頭 股價", "1216"),
        ("三商的年薪高達445.2萬元居次 股價 財報", "2905"),
        ("投資長不僅是公開比賽冠軍 績效 股票", "1806"),
    ])
    def test_demoted_names_not_matched(self, matcher, text, ticker):
        assert ticker not in matcher.match(text)

    @pytest.mark.parametrize("text,ticker", [
        ("[請益] 1216 統一適合存股嗎", "1216"),
        ("[情報] 2905 三商 113年第一季財報", "2905"),
        ("[標的] 1806 冠軍 多 停損", "1806"),
    ])
    def test_demoted_still_matched_by_code(self, matcher, text, ticker):
        assert ticker in matcher.match(text)


class TestNumericColumnFalsePositives:
    """排行表的數字欄不得被當成證券代號。"""

    @pytest.mark.parametrize("text,ticker", [
        ("2. 業強    +1506          2. 廣積  -1867", "1506"),
        ("25  超豐  971    25  華新  -1413", "1413"),
        ("1  中美晶  +2702      1. 華星光  -3421", "2702"),
        ("3474 華亞科  -3055   25.80   0.00", "3055"),
        ("其中有1517張設質 占持股比例達95.55%", "1517"),
    ])
    def test_column_value_not_a_ticker(self, matcher, text, ticker):
        assert ticker not in matcher.match(text)

    @pytest.mark.parametrize("text,ticker", [
        ("1  2002中鋼   市  1430  37.45  +1.7", "2002"),
        ("台積電 (2330) 及聯發科 (2454) 都獲推薦", "2330"),
        ("張忠謀說鼓勵國人多買2330", "2330"),
        ("可以觀察一下4903、6165、3520的線型", "6165"),
    ])
    def test_genuine_code_still_matched(self, matcher, text, ticker):
        assert ticker in matcher.match(text)


class TestPttTemplateStripping:
    """PTT [標的] 發文樣板的「(例 2330 台積電)」不得算成台積電的關注度。"""

    def test_template_example_removed(self):
        body = ("1. 標的:3264 欣銓\n(例 2330 台積電)\n2. 分類:多\n"
                "3. 分析/正文: 車用晶片產能吃緊")
        assert "2330" not in strip_ptt_template(body)
        assert "3264" in strip_ptt_template(body)

    def test_template_not_attributed(self, matcher):
        # 2409 友達在宇宙內；2330 只出現在樣板範例行
        body = ("標題請使用以下格式\n"
                "                 ex [標的] 2330.TW 台積電 長期不停損多\n"
                "1. 標的:2409 友達\n2. 分類:多")
        found = matcher.match("[標的] 2409.TW 友達 多", body)
        assert "2330" not in found
        assert "2409" in found

    def test_real_mention_survives_stripping(self, matcher):
        found = matcher.match("[標的] 台積電 2330 多", "看好先進製程 目標價")
        assert "2330" in found


class TestSecondRoundFindings:
    """第二輪抽驗發現的代號誤配（欄位對齊、量詞）。"""

    @pytest.mark.parametrize("text,ticker", [
        ("3706  神達          1809      1823        14    24.86", "1809"),
        ("6274   台燿                  1203       115       671", "1203"),
        ("巴西新增 159 例,累計確診 2433 例", "2433"),
        ("中國人民銀行共持有1762噸黃金", "1762"),
        ("全案共有2603件申訴", "2603"),
    ])
    def test_aligned_column_and_units_rejected(self, matcher, text, ticker):
        assert ticker not in matcher.match(text)

    @pytest.mark.parametrize("text,ticker", [
        ("1301  台塑           125    78.70     0.30", "1301"),
        ("華新麗華(1605)29日開出的4月產品內銷價格", "1605"),
        ("中化生 (1762-TW)、台耀 (4746-TW) 均具備生產能力", "1762"),
        ("1414    東和    17.8    4,186   元富大昌", "1414"),
    ])
    def test_code_with_adjacent_name_still_matched(self, matcher, text, ticker):
        assert ticker in matcher.match(text)

    def test_wide_column_gap_rejected(self, matcher):
        """欄位間距可能遠超過 8 個字元，視窗太短會漏抓。"""
        text = "29      大成鋼          1410            29      宇環"
        assert "1410" not in matcher.match(text)
