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
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import requests
from faster_whisper import WhisperModel

# ======== 預設參數 ========
ASR_MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"     # 1080Ti 建議 int8 或 int8_float16
LMSTUDIO_API = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "qwen2.5-7b-instruct"
OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"

INIT_PROMPT = (
    "以下是大學老師與大一學生的一對一輔導談話逐字稿。請用正體中文理解口語，保留教育場域語境。"
)
SUMMARY_PROMPT = """
你是一位學輔紀錄助理。請根據「逐字稿」產出結構化結果。

請只輸出一個 JSON 物件（不得有任何前後解說文字、標點或程式碼圍欄），鍵與型別 **必須** 完全符合以下規範：
{
  "summary": "<以繁體中文撰寫，350–450字。請在摘要中清楚區分老師與學生的觀點／重點>",
  "categories": [ "<從下列候選值中擇一或多個：課業、生活、交友、心理、生涯、課外活動>", ... ],
  "risk_flags": [ "<可留空陣列；如有，列出具體風險點，如：長期失眠、疑似焦慮、自傷念頭等>", ... ],
  "followups": [ "<教師後續追蹤行動建議（條列列點）>", ... ]
}

規則：
- 不得輸出任何解說、步驟、分析、推理或思考內容；**只允許輸出單一 JSON 物件**。
- 只能輸出 **一個** JSON 物件，且必須是合法 JSON（雙引號、逗號與括號位置正確）。
- "categories" 陣列的元素必須只來自以下候選值：["課業","生活","交友","心理","生涯","課外活動"]；可多選或給空陣列。
- "summary" 必須 350–450 個中文字，避免個資；必要時可用「學生」「老師」指稱。
- 不得新增除了 "summary","categories","risk_flags","followups" 以外的鍵。
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

    _WHISPER_MODEL = model
    _WHISPER_MODEL_DEVICE = device
    _WHISPER_MODEL_COMPUTE = compute
    return model, device, compute

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
    lines = []
    for seg in segments:
        mm = int(seg.start // 60)
        ss = int(seg.start % 60)
        ts = f"[{mm:02d}:{ss:02d}] "
        lines.append(ts + (seg.text or "").strip())
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
        "options": {"temperature": 0.2}
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

def coerce_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
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
        "summary": str(data.get("summary", "")).strip(),
        "categories": coerce_list(data.get("categories", [])),
        "risk_flags": coerce_list(data.get("risk_flags", [])),
        "followups": coerce_list(data.get("followups", []))
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
            transcript_txt, seconds = transcribe(audio_path)
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
