# Repository Guidelines

## Project Structure & Module Organization

Code lives in `src/`, with `src/local_asr_pipeline.py` providing the CLI for batch ASR, summary generation, and the `--force` override. `src/__init__.py` simply exposes the pipeline module so editors treat the directory as a package. Generated transcripts and summaries are written to `outputs/`; keep JSON/TXT artifacts grouped by source audio name and avoid hand-editing them. Retain audio under a separate storage path (e.g., `D:/audio/`) and reference it by absolute path when running the pipeline.

## Build, Test, & Development Commands

Create the virtual environment and install dependencies listed in `requirements.txt`:
`python -m venv .venv && .venv\Scripts\Activate.ps1`
`pip install -r requirements.txt`
Run the pipeline for a folder or single file:
`python src/local_asr_pipeline.py D:/audio --provider ollama`
Force a re-run of both ASR and summarization for a file:
`python src/local_asr_pipeline.py D:/audio/file.mp3 --force`
Export collected JSON summaries into a CSV (UTF-8 BOM for Excel):
`python merge_json_to_csv.py outputs -o summaries.csv`

## Coding Style & Naming Conventions

Follow PEP 8 with 4-space indentation and snake_case symbols. Centralize configuration (model names, endpoints, default folders) near the constants block in `local_asr_pipeline.py`. When adding CLI flags or helper modules, keep public functions typed and documented with concise docstrings. Log status using the existing emoji-prefixed messages for consistency.

## Testing Guidelines

No automated suite ships yet, so create targeted `pytest` cases whenever you extract reusable logic (e.g., JSON parsing or timing utilities). Before merging, run the pipeline on at least one short clip and verify both the transcript TXT and summary JSON for timestamp drift, category labels, and `processed_at` metadata. Document manual test clips in PR notes for traceability.

## Commit & Pull Request Guidelines

Use imperative commit subjects such as `Add timing logs to process_one` or `Skip summary when JSON exists`. Reference related issues in the body when applicable. Pull requests should include: a brief summary, the commands executed (with provider/model variants), sample output paths under `outputs/`, and any configuration deltas (GPU vs CPU fallback, API timeouts). Attach logs when changes touch inference or LLM calls so reviewers can reproduce behavior quickly.
