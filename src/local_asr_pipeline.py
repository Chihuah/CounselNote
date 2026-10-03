"""
CounselNote 本地 ASR 與摘要自動化工具。

功能：
- 將單一音檔或整個資料夾批次進行轉錄（ASR）與摘要匯出
- 利用現有的 TXT/JSON 快取，必要時可加上 --force 強制重跑
- 支援 LM Studio、Ollama 兩種本地 LLM，結果寫入指定輸出資料夾

基本用法：
    python src/local_asr_pipeline.py <path> [選項]

常用選項：
    --provider {lmstudio, ollama}    選擇摘要所使用的 LLM 服務
    --lmstudio_model NAME            指定 LM Studio 模型（預設 qwen2.5-7b-instruct）
    --ollama_model NAME              指定 Ollama 模型（預設 qwen3:4b）
    --out_dir PATH                   輸出資料夾，預設 outputs
    --ext LIST                       批次模式支援的副檔名清單（逗號分隔）
    --force                          即使已有 TXT/JSON 也重新轉錄與摘要

範例：
    python src/local_asr_pipeline.py D:/audio --provider ollama
    python src/local_asr_pipeline.py D:/audio/case.mp3 --provider lmstudio --force
"""

import os
import re
import json
import argparse
import gc
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import importlib.util
import requests
from opencc import OpenCC


def register_nvidia_dll_dirs() -> None:
    """Windows 下把 pip 安裝的 nvidia-cublas/cudnn-cu12 的 bin 目錄加入 DLL 搜尋路徑。"""
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations if spec else None) or []:
        for name in sorted(os.listdir(root)):
            bin_dir = os.path.join(root, name, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


register_nvidia_dll_dirs()
from faster_whisper import WhisperModel  # noqa: E402  需在 DLL 路徑註冊後匯入

# ======== 預設參數 ========
ASR_MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"  # RTX 5080（16GB）可直接用 float16；VRAM 不足時改 int8_float16 或 int8
LMSTUDIO_API = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "qwen2.5-7b-instruct"
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_NUM_CTX = 16384    # Ollama 預設 4096，長逐字稿會被截掉開頭的指令
_TO_TRADITIONAL = OpenCC("s2twp")  # 簡體 → 臺灣正體（含慣用詞，如 软件→軟體）


def to_traditional(text: str) -> str:
    """將文字中的簡體字轉成臺灣正體；已是正體的內容維持不變。"""
    return _TO_TRADITIONAL.convert(text)

# Whisper 的 initial_prompt。預設不使用：實測指令式提示詞會在開頭靜音或聽不清處被原樣吐回，
# 並經由前文條件化一路延續，吞掉前兩分鐘的真實內容；正體轉換已由 OpenCC 處理。
# 若要設定，請用逐字稿風格的短句而非指令；被吐回的片段會由 is_prompt_echo 濾除。
INIT_PROMPT: Optional[str] = None
SUMMARY_PROMPT = """
你是一位學輔紀錄助理。請根據「逐字稿」產出結構化結果。

請只輸出一個 JSON 物件（不得有任何前後解說文字、標點或程式碼圍欄），鍵與型別 **必須** 完全符合以下規範：
{
  "summary": "<以繁體中文撰寫，350–450字。請在摘要中清楚區分老師與學生的觀點／重點>",
  "categories": [ "<從下列候選值中擇一或多個：課業、生活、交友、心理、生涯、課外活動>", ... ],
  "risk_flags": [ "<字串陣列，可為空；每項格式：【高】或【中】＋風險描述＋『逐字稿原句』＋[mm:ss]，判斷標準見下方「risk_flags 判斷標準」>", ... ],
  "followups": [ "<教師後續追蹤行動建議（條列列點）>", ... ]
}

規則：
- 不得輸出任何解說、步驟、分析、推理或思考內容；**只允許輸出單一 JSON 物件**。
- 只能輸出 **一個** JSON 物件，且必須是合法 JSON（雙引號、逗號與括號位置正確）。
- "categories" 陣列的元素必須只來自以下候選值：["課業","生活","交友","心理","生涯","課外活動"]；可多選或給空陣列。
- "summary" 必須 350–450 個中文字，避免個資；必要時可用「學生」「老師」指稱。
- 不得新增除了 "summary","categories","risk_flags","followups" 以外的鍵。

risk_flags 判斷標準（學生安全優先：寧可標記並註明需人工確認，也不要漏報）：
- 【高】一律標記：自傷或自殺念頭／計畫；傷害他人；遭受暴力、霸凌或性騷擾；藥物或酒精濫用。
- 【中】須有明確證據才標記：持續兩週以上的失眠、情緒低落或焦慮；飲食或體重明顯變化；長期缺課或有退學念頭；嚴重人際孤立；經濟困難影響就學。
- 不標記：一般課業壓力、偶爾熬夜、已有醫療處置且穩定的狀況（例如按時服藥與回診）。
- 每一項必須是單純的字串（不得使用物件或巢狀結構），格式為「【高/中】<風險描述>：『<逐字稿原句>』[mm:ss]」。
- 原句必須逐字出自逐字稿，不得改寫或編造；[mm:ss] 必須逐字複製該原句所在那一行行首的時間戳。
- 等級須與上方分級對應，風險描述須與所引用的原句內容相符，不可把中風險情況標成高風險。
- 不確定是否構成風險、但可能有安全疑慮時，仍須標記，並在該項結尾加註「（需人工確認）」。
- 逐字稿中沒有任何符合上述條件的內容時，"risk_flags" 輸出空陣列 []。
"""
# ===========================

_WHISPER_MODEL = None
_WHISPER_MODEL_DEVICE = None
_WHISPER_MODEL_COMPUTE = None

def get_whisper_model() -> Tuple[WhisperModel, str, str]:
    """載入 Whisper 模型，若 CUDA 失敗則改用 CPU。"""
    global _WHISPER_MODEL, _WHISPER_MODEL_DEVICE, _WHISPER_MODEL_COMPUTE
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL, _WHISPER_MODEL_DEVICE, _WHISPER_MODEL_COMPUTE

    device = DEVICE
    compute = COMPUTE_TYPE
    print(f"[info] 載入 Whisper 模型 {ASR_MODEL_SIZE}（首次執行需下載約 3GB，期間不會顯示進度）…")
    load_start = time.time()
    try:
        model = WhisperModel(ASR_MODEL_SIZE, device=device, compute_type=compute)
    except Exception as exc:
        if device != "cuda":
            raise
        fallback_device = "cpu"
        fallback_compute = "int8"
        print(f"[warn] CUDA 初始化失敗（{exc}），改用 CPU int8 重新載入模型。")
        model = WhisperModel(ASR_MODEL_SIZE, device=fallback_device, compute_type=fallback_compute)
        device = fallback_device
        compute = fallback_compute
    print(f"[info] Whisper 模型載入完成（{time.time() - load_start:.1f}s）")

    _WHISPER_MODEL = model
    _WHISPER_MODEL_DEVICE = device
    _WHISPER_MODEL_COMPUTE = compute
    return model, device, compute

def release_whisper_model() -> None:
    """釋放 Whisper 模型與其佔用的 VRAM，讓 LLM 摘要可使用完整顯示記憶體。"""
    global _WHISPER_MODEL, _WHISPER_MODEL_DEVICE, _WHISPER_MODEL_COMPUTE
    _WHISPER_MODEL = None
    _WHISPER_MODEL_DEVICE = None
    _WHISPER_MODEL_COMPUTE = None
    gc.collect()

def format_mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def is_prompt_echo(text: str, prompt: Optional[str] = INIT_PROMPT) -> bool:
    """判斷片段是否為 Whisper 吐回的 initial_prompt（片段整段出自提示詞，且至少 6 字以免誤刪短句）。"""
    if not prompt:
        return False
    needle = _normalize(to_traditional(text))
    return len(needle) >= 6 and needle in _normalize(to_traditional(prompt))

def transcribe(audio_path: str) -> Tuple[str, float]:
    """faster-whisper 語音轉文字"""
    model, device_used, compute_used = get_whisper_model()
    print(f"[info] start transcription device={device_used} compute={compute_used}: {audio_path}")
    segments, info = model.transcribe(
        audio_path,
        language="zh",
        initial_prompt=INIT_PROMPT,
        vad_filter=True,
        beam_size=5,
    )
    total = format_mmss(info.duration)
    lines = []
    dropped = 0
    next_pct = 10
    for seg in segments:
        text = to_traditional((seg.text or "").strip())
        if is_prompt_echo(text):
            dropped += 1
        else:
            lines.append(f"[{format_mmss(seg.start)}] {text}")
        pct = seg.end / info.duration * 100 if info.duration else 100
        if pct >= next_pct:
            print(f"[info] 轉錄進度 {min(pct, 100):.0f}%（{format_mmss(seg.end)} / {total}）")
            next_pct = (int(pct) // 10 + 1) * 10
    if dropped:
        print(f"[warn] 已濾除 {dropped} 段 Whisper 吐回提示詞的片段")
    transcript = "\n".join(lines)
    print(f"[info] transcription finished {audio_path} ({info.duration:.1f}s)")
    return transcript, info.duration

def call_lmstudio(model: str, content: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是嚴謹的學輔摘要助手。"},
            {"role": "user", "content": SUMMARY_PROMPT + "\n\n逐字稿：\n" + content}
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": False
    }
    r = requests.post(LMSTUDIO_API, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_ollama(model: str, content: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是嚴謹的學輔摘要助手。"},
            {"role": "user", "content": SUMMARY_PROMPT + "\n\n逐字稿：\n" + content}
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_ctx": OLLAMA_NUM_CTX}
    }
    r = requests.post(OLLAMA_API, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]

def extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start_idx, end_idx = text.find("{"), text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx+1]
    return text

def _item_to_str(item) -> str:
    """列表項目轉字串；若模型誤輸出物件（dict），改取其字串值而非 dict 的 repr。"""
    if isinstance(item, dict):
        return " ".join(str(v) for v in item.values())
    return str(item)

def coerce_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [_item_to_str(i) for i in x]
    if isinstance(x, str):
        parts = [p.strip() for p in re.split(r"[，,]\s*", x) if p.strip()]
        return parts if parts else ([x] if x else [])
    return [str(x)]

def text_after_last_think(text: str) -> str:
    """
    回傳最後一個 </think> 標籤之後的文字。
    若不存在 </think>（大小寫不拘），則回傳原文。
    """
    matches = list(re.finditer(r"</\s*think\s*>", text, flags=re.IGNORECASE))
    if not matches:
        return text
    last_end = matches[-1].end()
    return text[last_end:].lstrip()

_FLAG_QUOTE = re.compile(r"『(.+?)』")
_FLAG_STAMP = re.compile(r"\[(\d{2}:\d{2})\]")
_STAMP_LINE = re.compile(r"\[(\d{2}:\d{2})\] ?(.*)")
_IGNORED_CHARS = re.compile(r"[\s，。、！？；：,.!?;:「」『』（）()…～~—-]")
_NEED_REVIEW = "（需人工確認）"

def _normalize(text: str) -> str:
    """移除空白與標點，供原句比對（Whisper 逐字稿多半無標點，模型引用時可能自行加上）。"""
    return _IGNORED_CHARS.sub("", text)

def index_transcript(transcript: str) -> Tuple[str, List[Optional[str]]]:
    """
    將逐字稿攤平成連續文字，並記錄每個字元所屬行的時間戳（無時間戳的行為 None）。
    逐字稿先轉為正體，以相容舊版簡體快取。
    """
    chars: List[str] = []
    owners: List[Optional[str]] = []
    for line in to_traditional(transcript).splitlines():
        m = _STAMP_LINE.match(line)
        stamp, body = (m.group(1), m.group(2)) if m else (None, line)
        normalized = _normalize(body)
        chars.extend(normalized)
        owners.extend([stamp] * len(normalized))
    return "".join(chars), owners

def check_risk_flag(flag: str, flat: str, owners: List[Optional[str]]) -> Optional[str]:
    """核對單項 risk_flag 的原句與時間戳；通過回傳 None，否則回傳失敗原因。"""
    quotes = _FLAG_QUOTE.findall(flag)
    if not quotes:
        return "缺少逐字稿原句"
    first_span: set = set()
    for i, quote in enumerate(quotes):
        needle = _normalize(quote)
        pos = flat.find(needle) if needle else -1
        if pos < 0:
            return "引用未在逐字稿中找到"
        if i == 0:
            first_span = {o for o in owners[pos:pos + len(needle)] if o}
    if not first_span:
        return None  # 逐字稿沒有時間戳（如舊版 TXT），僅核對原句
    stamp = _FLAG_STAMP.search(flag)
    if not stamp:
        return "缺少時間戳"
    if stamp.group(1) not in first_span:
        return "時間戳與原句不符"
    return None

def annotate_risk_flags(flags: List[str], transcript: str) -> List[str]:
    """核對每項 risk_flag（完全相同者只保留第一項）；未通過者保留原項並在結尾加註原因，回傳新的列表。"""
    flat, owners = index_transcript(transcript)
    result: List[str] = []
    for flag in dict.fromkeys(flags):
        reason = check_risk_flag(flag, flat, owners)
        if reason is None:
            result.append(flag)
        else:
            result.append(f"{flag.removesuffix(_NEED_REVIEW)}（{reason}，需人工確認）")
    return result

def summarize(provider: str, model: str, transcript: str) -> Dict[str, Any]:
    raw = call_lmstudio(model, transcript) if provider == "lmstudio" else call_ollama(model, transcript)
    # 先擷取最後一個 </think> 之後的內容（避開可能的思考過程雜訊，例如qwen3:4b）
    raw = text_after_last_think(raw)

    json_str = extract_json_block(raw)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 回傳非 JSON：{e}\n原始：\n{raw}")
    return {
        "summary": to_traditional(str(data.get("summary", "")).strip()),
        "categories": coerce_list(data.get("categories", [])),
        "risk_flags": annotate_risk_flags(
            [to_traditional(s) for s in coerce_list(data.get("risk_flags", []))], transcript
        ),
        "followups": [to_traditional(s) for s in coerce_list(data.get("followups", []))]
    }

def preferred_transcript_path(audio_path: str, out_dir: str) -> str:
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(out_dir, base + ".txt")

def find_existing_transcript(audio_path: str, out_dir: str) -> Optional[str]:
    cand1 = preferred_transcript_path(audio_path, out_dir)
    if os.path.isfile(cand1):
        return cand1
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cand2 = os.path.join(os.path.dirname(audio_path), base + ".txt")
    return cand2 if os.path.isfile(cand2) else None







def process_one(audio_path: str, provider: str, lmstudio_model: str, ollama_model: str, out_dir: str, force: bool) -> Tuple[bool, str]:
    trans_duration = 0.0
    summary_duration = 0.0
    trans_start: Optional[float] = None
    summary_start: Optional[float] = None
    summary_started = False
    try:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        existing_txt = find_existing_transcript(audio_path, out_dir)

        trans_start = time.time()
        if existing_txt and not force:
            with open(existing_txt, "r", encoding="utf-8") as f:
                transcript_txt = f.read()
            print(f"⏭️ 已存在逐字稿：{existing_txt}（--force 未指定，跳過 ASR）")
            seconds = 0
            trans_duration = 0.0
        else:
            try:
                transcript_txt, seconds = transcribe(audio_path)
            finally:
                release_whisper_model()
            out_txt = preferred_transcript_path(audio_path, out_dir)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(transcript_txt)
            trans_duration = time.time() - trans_start
            print(f"🎧 執行 ASR：{audio_path}")

        out_json = os.path.join(out_dir, base + ".json")
        if os.path.isfile(out_json) and not force:
            print(f"⏭️ 已存在摘要：{out_json}（--force 未指定，跳過摘要）")
            summary_duration = 0.0
        else:
            summary_start = time.time()
            summary_started = True
            s = summarize(provider, lmstudio_model if provider == "lmstudio" else ollama_model, transcript_txt)
            summary_duration = time.time() - summary_start

            final_obj = {
                "file": audio_path,
                "processed_at": datetime.fromtimestamp(os.path.getmtime(audio_path)).isoformat(timespec="seconds"),
                "duration_sec": int(seconds),
                "transcript_txt": transcript_txt,
                "summary": s["summary"],
                "categories": s["categories"],
                "risk_flags": s["risk_flags"],
                "followups": s["followups"]
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(final_obj, f, ensure_ascii=False, indent=2)

        print(f"[time] transcript {trans_duration:.1f}s | summary {summary_duration:.1f}s")
        print(f"✅ 完成：{audio_path} → {out_json}")
        return True, out_json
    except Exception as e:
        now = time.time()
        if trans_start is not None and trans_duration == 0.0:
            trans_duration = now - trans_start
        if summary_started and summary_start is not None and summary_duration == 0.0:
            summary_duration = now - summary_start
        print(f"[time] transcript {trans_duration:.1f}s | summary {summary_duration:.1f}s (失敗)")
        return False, f"{audio_path} 失敗：{e}"

def find_audio_files(path: str, exts: List[str]) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files: List[str] = []
    for root, _, names in os.walk(path):
        for name in names:
            if name.lower().endswith(tuple(exts)):
                files.append(os.path.join(root, name))
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="ASR→LLM 摘要（支援批次，--force 可強制重跑 ASR）")
    parser.add_argument("path")
    parser.add_argument("--provider", choices=["lmstudio", "ollama"], default="ollama")
    parser.add_argument("--lmstudio_model", default=LMSTUDIO_MODEL)
    parser.add_argument("--ollama_model", default=OLLAMA_MODEL)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--ext", default="mp3,wav,m4a,aac,flac")
    parser.add_argument("--force", action="store_true", help="即使已有逐字稿也強制重跑 ASR")
    args = parser.parse_args()

    exts = [("." + ext.strip().lstrip(".")).lower() for ext in args.ext.split(",") if ext.strip()]
    targets = find_audio_files(args.path, exts)
    if not targets:
        raise FileNotFoundError("找不到音檔")

    print(f"🔍 找到 {len(targets)} 個音檔，force={args.force}")
    overall_start = time.time()
    ok, fail = 0, 0
    errors: List[str] = []

    for idx, fp in enumerate(targets, 1):
        print(f"\n[{idx}/{len(targets)}] 處理：{fp}")
        success, info = process_one(fp, args.provider, args.lmstudio_model, args.ollama_model, args.out_dir, args.force)
        if success:
            ok += 1
        else:
            fail += 1
            errors.append(info)
            print("❌", info)

    total_elapsed = time.time() - overall_start
    print(f"\n===== 批次總結 =====\n成功：{ok}, 失敗：{fail}")
    print(f"[time] 全部處理耗時 {total_elapsed:.1f}s")

    if errors:
        print("失敗清單：")
        for err in errors:
            print(" -", err)

if __name__ == "__main__":
    main()
