#!/usr/bin/env python3
"""
merge_json_to_csv.py

蒐集指定資料夾內的摘要 JSON，整併成單一 CSV。
transcript_txt 欄位會被排除。
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Dict, Any

FIELD_ORDER = [
    "file",
    "processed_at",
    "duration_sec",
    "summary",
    "categories",
    "risk_flags",
    "followups",
]

def load_json_files(folder: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from *.json files under folder (sorted by name)."""
    paths = sorted(folder.glob("*.json"))
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.pop("transcript_txt", None)  # drop verbose field
        yield data

def normalize_value(value: Any) -> str:
    """Convert JSON values into CSV-friendly strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)

def export_to_csv(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """Write records to CSV with predefined column order."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, FIELD_ORDER, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {field: normalize_value(record.get(field, "")) for field in FIELD_ORDER}
            writer.writerow(row)

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSON summaries into a single CSV.")
    parser.add_argument("json_dir", type=Path, help="包含摘要 JSON 檔的資料夾")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("summaries.csv"),
        help="輸出 CSV 路徑（預設 summaries.csv）",
    )
    args = parser.parse_args()

    records = list(load_json_files(args.json_dir))
    if not records:
        raise SystemExit(f"資料夾 {args.json_dir} 找不到任何 JSON 摘要。")

    export_to_csv(records, args.output)
    print(f"✅ 已匯出 {len(records)} 筆摘要到 {args.output}")
    
if __name__ == "__main__":
    main()
