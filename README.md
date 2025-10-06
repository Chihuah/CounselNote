# CounselNote

**Windows / GPU 本地端輔導談話逐字稿處理工具**

1. 使用 faster-whisper 完成語音轉文字（ASR）。
2. 透過本地 LLM（LM Studio 或 Ollama）產出逐字稿摘要與分類。
3. 輸出標準化 JSON，包含 `summary`、`categories`、`risk_flags`、`followups`。

> 所有資料皆在本地端處理，不需上傳雲端。適合需要資料不離機、重視個資與合規的教育現場與研究環境。

---

## 功能總覽

- **批次支援**：給定資料夾即可遞迴處理所有支援的音檔。
- **ASR 快取**：若既有 `outputs/<檔名>.txt` 或音檔同層的 TXT，即可跳過轉錄（可用 `--force` 強制重跑）。
- **摘要快取**：若既有 `outputs/<檔名>.json`，則跳過摘要生成（同樣可用 `--force` 覆寫）。
- **狀態日誌**：輸出每個檔案的轉錄與摘要耗時，以及整批處理總耗時。
- **JSON 結構**：方便匯入校務系統或資料庫，亦可搭配 CSV 匯出工具。

---

## 📁 專案資料夾結構

```text
CounselNote/                        # 專案根目錄
├─ AGENTS.md                        # 協作者Codex CLI指南，說明流程與規範
├─ README.md                        # 專案概述、安裝與使用教學
├─ requirements.txt                 # Python 依賴套件清單
├─ src/                             # 主要程式碼
│  ├─ local_asr_pipeline.py         # 轉錄＋摘要 CLI 主流程 (含快取邏輯、計時)
│  └─ merge_json_to_csv.py          # 將 outputs/ 中的摘要 JSON 合併為 CSV 的小工具
└─ outputs/                         # 產生的逐字稿與摘要結果（已在 .gitignore 中忽略內容）
```

---

## 安裝指南

### 1. 取得原始碼

```powershell
# 範例路徑
D:\> git clone <your-repo-url> CounselNote
D:\> cd CounselNote
```

### 2. 建立虛擬環境（Windows）

```powershell
python --version
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. 安裝依賴套件

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> GPU 使用者建議安裝 **CUDA** 並更新 NVIDIA 顯示卡驅動程式。

---

## 準備在地 LLM

### 方案 A：LM Studio

1. 安裝 LM Studio，下載一個 **Instruct** 類型的中文版模型（如 `qwen2.5-7b-instruct`）。
2. 啟動 Local Server（OpenAI Compatible API），確認端點為 `http://localhost:1234/v1`。
3. 執行程式時帶入 `--provider lmstudio --lmstudio_model <模型名稱>`。

### 方案 B：Ollama（建議）

1. 安裝 Ollama：<https://ollama.com/>
2. 下載模型範例：
   ```powershell
   ollama pull qwen3:4b
   ```
3. 執行程式時帶入 `--provider ollama --ollama_model qwen3:4b`。

---

## 使用說明

### 處理單一音檔

```powershell
.venv\Scripts\Activate.ps1
python src\local_asr_pipeline.py D:\audio\case_20241003.mp3 --provider lmstudio --lmstudio_model qwen2.5-7b-instruct
```

### 批次處理資料夾

```powershell
python src\local_asr_pipeline.py D:\audio --provider ollama --ollama_model qwen3:4b --ext mp3,wav,m4a,aac,flac
```

### 強制重跑轉錄與摘要

```powershell
python src\local_asr_pipeline.py D:\audio --force
```

### JSON 合併至 CSV 檔

此外，`src/merge_json_to_csv.py` 可將 `outputs/` 內的摘要 JSON 合併為單一 CSV。執行範例：

```powershell
python src/merge_json_to_csv.py outputs -o summaries.csv
```

---

## local_asr_pipeline.py 主要參數

| 參數               | 說明                         | 預設值                 |
| ------------------ | ---------------------------- | ---------------------- |
| `path`             | 單檔或資料夾路徑             | 必填                   |
| `--provider`       | `lmstudio` 或 `ollama`       | `ollama`               |
| `--lmstudio_model` | LM Studio 模型名稱           | `qwen2.5-7b-instruct`  |
| `--ollama_model`   | Ollama 模型名稱              | `qwen3:4b`             |
| `--out_dir`        | 輸出資料夾                   | `outputs`              |
| `--ext`            | 批次模式下的副檔名清單       | `mp3,wav,m4a,aac,flac` |
| `--force`          | 即使存在 TXT/JSON 也強制重跑 | `False`                |

---

## 輸出格式

- 每個音檔會對應一份 `outputs/<檔名>.txt`（逐字稿）與 `outputs/<檔名>.json`（摘要）。
- `processed_at` 使用來源音檔的最後修改時間。
- JSON 範例：

```json
{
  "file": "D:/audio/case_20241003.mp3",
  "processed_at": "2025-10-05T12:00:00",
  "duration_sec": 3725,
  "summary": "150-250 字內的重點概述，拆解師生觀點。",
  "categories": ["課業", "心理"],
  "risk_flags": ["提及長期失眠"],
  "followups": ["兩週後回訪睡眠狀況", "轉介校內諮商初談"]
}
```

---

## 核心功能：摘要提示詞

下方是用於生成結構化摘要的提示詞。它要求模型僅輸出一個格式正確的 JSON 物件，確保欄位、型別與內容都符合輔導紀錄需求：

```text
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
```

---

## 性能建議（以 nVidia 1080Ti 為例）

- `ASR_MODEL_SIZE` 可依 GPU 記憶體調整：`large-v3` 精準、`medium` 較快。
- `compute_type` 建議 `int8` 或 `int8_float16`。
- `beam_size` 可降低至 `1~3` 以提升速度。
- 預設啟用 `vad_filter=True` 以去除靜音段落。

---

## 常見問題（FAQ）

- **Q：找不到 GPU？**
  A：請更新 NVIDIA 驅動並安裝 CUDA Runtime；若仍失敗可將 `DEVICE` 改為 `cpu`。
- **Q：LLM 回傳不是 JSON？**
  A：請確認選用 Instruct 模型並保持 `temperature=0.2`；仍失敗可檢查日誌並調整提示詞。
- **Q：依賴版本不相容？**
  A：請將目前環境輸出：
  ```powershell
  pip freeze > requirements.txt
  ```

---

## Roadmap

- 本程式可選擇使用 LM Studio 之本地模型進行摘要總結，但尚未經過詳細測試，可能仍有錯誤存在，尚待除錯。而經實測， Ollama qwen3:4b 可用，但由於預設具備思考模式，因此需後處理相關標籤：`</think>`。例如`local_asr_pipeline.py`裡第 165 行的函式`text_after_last_think`。
- 後續製作 GUI 介面（Tkinter / PySimpleGUI），方便操作。
- 試用其他 ASR 模型，如`Qwen3-ASR-Flash`。
- SRT 字幕輸出（含毫秒時間軸）。
- 說話者分離（speaker diarization）。
- JSON Schema 驗證與 `.jsonl` 匯出。

---

## 專案開發背景/動機

本專案原為一份在 Google Colab 上運作的 ipynb 筆記本。當時的流程是用 faster-whisper（large-v3、中文初始提示）把錄音轉成逐字稿，再把文字交給 Gemini 進行摘要。就技術驗證而言，這條路線很快證明可行：轉錄品質穩定、摘要足以生成會議紀要；而且只要換 file_index 就能處理不同錄音。

但當流程落地到校園情境，敏感與隱私成了關鍵考量：輔導內容屬於高度敏感個資，不宜離開校內/本機環境；把逐字稿送到雲端 LLM 進一步處理，資料外流風險與合規壓力偏高。因此，專案轉向「本地端全流程」：以 faster-whisper 在本機 GPU 完成 ASR，再透過 LM Studio 或 Ollama 調用開源 LLM 做摘要與分類，全程不出機。為了讓日常操作更順手，進一步加入：資料夾遞迴批次處理與逐字稿快取偵測（同名 .txt 即跳過 ASR），或 `--force` 參數（必要時強制重跑 ASR）。

最終形成了 CounselNote：一個以隱私優先、可維運為核心的本地端工具，能把輔導會談錄音穩定地轉成逐字稿（TXT）與結構化摘要（JSON），並為後續的去識別化、SRT、說話者分離與報表分析預留擴充空間。

本專案之開發過程，先以 ChatGPT 5 網頁對話，草擬核心程式碼，再到本機端使用 OpenAI Codex CLI(model: gpt-5-codex) 完成後續程式碼編寫和除錯，以及補充說明文件。最終大約使用了 553K tokens。

---

## 授權

本專案以 MIT License 發布。使用、修改或散布程式碼時，請保留原始的著作權與授權聲明，並在相關文件中標示出處。詳細條款請參考根目錄的 LICENSE 檔案。
