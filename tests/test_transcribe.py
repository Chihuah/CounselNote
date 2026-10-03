"""轉錄輔助函式的測試（不載入 Whisper 模型）。"""

import local_asr_pipeline as p

PROMPT = "以下是大學老師與大一學生的一對一輔導談話逐字稿。請用正體中文理解口語，保留教育場域語境。"


class TestIsPromptEcho:
    def test_echoed_sentence_is_detected(self):
        assert p.is_prompt_echo("請用正體中文理解口語，保留教育場域語境。", PROMPT)

    def test_echo_detected_regardless_of_punctuation_and_script(self):
        assert p.is_prompt_echo("请用正体中文理解口语 保留教育场域语境", PROMPT)

    def test_short_utterance_inside_prompt_is_kept(self):
        # 「老師」「大一」等短詞雖出現在提示詞中，仍是正常對話
        assert not p.is_prompt_echo("老師", PROMPT)
        assert not p.is_prompt_echo("對，大一", PROMPT)

    def test_normal_speech_is_kept(self):
        assert not p.is_prompt_echo("所以暑假有去哪裡玩嗎", PROMPT)

    def test_no_prompt_never_filters(self):
        assert not p.is_prompt_echo("請用正體中文理解口語，保留教育場域語境。", None)


def test_format_mmss():
    assert p.format_mmss(0) == "00:00"
    assert p.format_mmss(857.1) == "14:17"
