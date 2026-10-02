"""risk_flags 核對、dict 項目轉字串與 summarize 整合的測試（不呼叫 LLM）。"""

import json

import pytest

import local_asr_pipeline as p

# Whisper 逐字稿的典型形態：短句、無標點、每行一個時間戳
TRANSCRIPT = "\n".join(
    [
        "[00:10] 今天想聊什麼",
        "[00:15] 最近我常常覺得",
        "[00:18] 不想活了",
        "[00:22] 有想過乾脆從宿舍陽台跳下去",
        "[01:05] 期中考前熬夜讀書",
    ]
)

TRADITIONAL_PLATFORM = "陽臺"  # s2twp 會把「陽台」轉為「陽臺」


def annotate(flag: str, transcript: str = TRANSCRIPT) -> str:
    return p.annotate_risk_flags([flag], transcript)[0]


class TestAnnotateRiskFlags:
    def test_valid_flag_is_unchanged(self):
        flag = "【高】自殺念頭：『不想活了』[00:18]"
        assert annotate(flag) == flag

    def test_quote_spanning_lines_with_added_punctuation_passes(self):
        # 模型引用跨多行，且自行加了逗號；時間戳為原句起始行
        flag = "【高】自殺念頭：『最近我常常覺得，不想活了』[00:15]"
        assert annotate(flag) == flag

    def test_timestamp_of_any_line_in_quote_span_passes(self):
        flag = "【高】自殺念頭：『最近我常常覺得不想活了』[00:18]"
        assert annotate(flag) == flag

    def test_wrong_timestamp_is_annotated(self):
        flag = "【高】自殺念頭：『不想活了』[02:11]"
        assert annotate(flag) == flag + "（時間戳與原句不符，需人工確認）"

    def test_fabricated_quote_is_annotated(self):
        flag = "【高】自傷：『我想結束生命』[00:18]"
        assert annotate(flag) == flag + "（引用未在逐字稿中找到，需人工確認）"

    def test_missing_quote_is_annotated(self):
        flag = "【中】持續失眠[00:18]"
        assert annotate(flag) == flag + "（缺少逐字稿原句，需人工確認）"

    def test_missing_timestamp_is_annotated(self):
        flag = "【高】自殺念頭：『不想活了』"
        assert annotate(flag) == flag + "（缺少時間戳，需人工確認）"

    def test_second_fabricated_quote_is_annotated(self):
        flag = "【高】自殺念頭：『不想活了』與『我已經準備好藥物』[00:18]"
        assert "引用未在逐字稿中找到" in annotate(flag)

    def test_existing_review_suffix_is_replaced_not_duplicated(self):
        flag = "【中】持續失眠：『我睡不著』[00:18]（需人工確認）"
        result = annotate(flag)
        assert result == "【中】持續失眠：『我睡不著』[00:18]（引用未在逐字稿中找到，需人工確認）"
        assert result.count("需人工確認") == 1

    def test_valid_flag_keeps_model_review_suffix(self):
        flag = "【中】孤立：『不想活了』[00:18]（需人工確認）"
        assert annotate(flag) == flag

    def test_transcript_without_timestamps_checks_quote_only(self):
        transcript = "最近我常常覺得\n不想活了\n有想過乾脆從宿舍陽台跳下去"
        ok = "【高】自殺念頭：『不想活了』"
        assert annotate(ok, transcript) == ok
        assert "引用未在逐字稿中找到" in annotate("【高】自傷：『我想結束生命』", transcript)

    def test_simplified_cached_transcript_matches_traditional_quote(self):
        transcript = "[00:18] 最近我常常觉得\n[00:20] 不想活了\n[00:22] 去软件系上课"
        flag = "【高】自殺念頭：『最近我常常覺得不想活了』[00:18]"
        assert annotate(flag, transcript) == flag

    def test_converted_quote_matches_converted_transcript(self):
        transcript = p.to_traditional(TRANSCRIPT)
        assert TRADITIONAL_PLATFORM in transcript
        flag = f"【高】自殺念頭：『有想過乾脆從宿舍{TRADITIONAL_PLATFORM}跳下去』[00:22]"
        assert annotate(flag, transcript) == flag

    def test_empty_list_returns_empty_list(self):
        assert p.annotate_risk_flags([], TRANSCRIPT) == []

    def test_exact_duplicates_are_removed_keeping_order(self):
        a = "【高】自殺念頭：『不想活了』[00:18]"
        b = "【中】熬夜：『期中考前熬夜讀書』[01:05]"
        assert p.annotate_risk_flags([a, b, a], TRANSCRIPT) == [a, b]

    def test_input_list_is_not_mutated(self):
        flags = ["【高】自傷：『我想結束生命』[00:18]"]
        original = list(flags)
        p.annotate_risk_flags(flags, TRANSCRIPT)
        assert flags == original


class TestCoerceList:
    def test_dict_items_become_plain_strings(self):
        result = p.coerce_list([{"flag": "【高】自殺念頭：『不想活了』[00:18]"}])
        assert result == ["【高】自殺念頭：『不想活了』[00:18]"]

    def test_plain_strings_unchanged(self):
        assert p.coerce_list(["a", "b"]) == ["a", "b"]

    def test_none_returns_empty_list(self):
        assert p.coerce_list(None) == []


class TestSummarizeIntegration:
    @staticmethod
    def _fake_llm(payload: dict):
        return lambda model, content: json.dumps(payload, ensure_ascii=False)

    def test_summarize_annotates_and_converts_risk_flags(self, monkeypatch):
        payload = {
            "summary": "学生提到近况。",
            "categories": ["心理"],
            "risk_flags": [
                {"flag": "【高】自殺念頭：『不想活了』[00:18]"},
                "【高】自傷：『我想结束生命』[00:18]",
            ],
            "followups": ["持续追踪"],
        }
        monkeypatch.setattr(p, "call_ollama", self._fake_llm(payload))

        result = p.summarize("ollama", "any-model", TRANSCRIPT)

        assert result["summary"] == "學生提到近況。"
        assert result["risk_flags"][0] == "【高】自殺念頭：『不想活了』[00:18]"
        assert result["risk_flags"][1].endswith("（引用未在逐字稿中找到，需人工確認）")
        assert result["followups"] == ["持續追蹤"]

    def test_summarize_with_no_risk_flags_returns_empty_list(self, monkeypatch):
        payload = {"summary": "x", "categories": [], "risk_flags": [], "followups": []}
        monkeypatch.setattr(p, "call_ollama", self._fake_llm(payload))
        assert p.summarize("ollama", "any-model", TRANSCRIPT)["risk_flags"] == []

    def test_summarize_rejects_non_json(self, monkeypatch):
        monkeypatch.setattr(p, "call_ollama", lambda model, content: "不是 JSON")
        with pytest.raises(ValueError):
            p.summarize("ollama", "any-model", TRANSCRIPT)
