#!/usr/bin/env python3
"""
Tag all results/ COT and NTCOT entries with the stigma taxonomy using Claude on AWS Bedrock.

Single-run tagging (no ensemble) with checkpoint/resume support.

Usage:
    # Dry run: see what will be processed
    python tag_results.py --dry-run

    # Test with 10 items from one folder
    python tag_results.py --folder deepseek-v31_COT --limit 10 --workers 3

    # Resume same folder (picks up where it left off)
    python tag_results.py --folder deepseek-v31_COT --workers 3

    # Run all folders
    python tag_results.py --workers 3

    # Fresh start (clears previous output)
    python tag_results.py --folder deepseek-v31_COT --clean-start --workers 3
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

from unified_client import create_tagging_client, process_single_item


RESULTS_DIR = REPO_ROOT / "results"
TAXONOMY_FILE = REPO_ROOT / "data" / "taxonomy.json"


def load_taxonomy() -> dict:
    """Load the stigma taxonomy from data/taxonomy.json."""
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Bedrock model IDs for the tagging model
MODEL_ALIASES = {
    "sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "opus": "us.anthropic.claude-opus-4-5-20251101-v1:0",
}

# Logging setup
LOG_FILE = REPO_ROOT / "tag_results.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class RateLimitTracker:
    """Track rate limit errors and signal abort if threshold exceeded."""

    def __init__(self, max_errors: int = 5, window_seconds: int = 60):
        self.max_errors = max_errors
        self.window_seconds = window_seconds
        self._errors: list[float] = []
        self._lock = Lock()
        self.should_abort = False

    def record_error(self, error_msg: str):
        with self._lock:
            now = time.time()
            self._errors.append(now)
            # Only keep errors within the window
            cutoff = now - self.window_seconds
            self._errors = [t for t in self._errors if t >= cutoff]
            log.warning(f"RATE LIMIT ERROR ({len(self._errors)}/{self.max_errors} in {self.window_seconds}s): {error_msg}")
            if len(self._errors) >= self.max_errors:
                log.error(f"ABORTING: {self.max_errors} rate limit errors in {self.window_seconds}s window")
                self.should_abort = True

    @property
    def total_errors(self):
        with self._lock:
            return len(self._errors)


# =============================================================================
# Folder Discovery
# =============================================================================

def discover_folders(base_dir: Path, folder_filter: list[str] | None = None, suffix: str = "") -> list[dict]:
    """Scan results/ for COT and NTCOT dirs (excluding Tagged dirs)."""
    folders = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.endswith("_Tagged"):
            continue
        if not (name.endswith("_COT") or name.endswith("_NTCOT")):
            continue
        if folder_filter and name not in folder_filter:
            continue

        # Derive model name: everything before _COT or _NTCOT
        if name.endswith("_NTCOT"):
            model = name[:-6]
            variant = "NTCOT"
        else:
            model = name[:-4]
            variant = "COT"

        folders.append({
            "input_dir": entry,
            "output_dir": base_dir / f"{name}_Tagged{suffix}",
            "name": name,
            "model": model,
            "variant": variant,
        })
    return folders


# =============================================================================
# JSONL I/O
# =============================================================================

def load_jsonl(jsonl_path: Path, model_name: str) -> list[dict]:
    """Read JSONL file, inject model field from folder name."""
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["model"] = model_name
            items.append(record)
    return items


def count_output_lines(jsonl_path: Path) -> int:
    """Count valid lines in output JSONL (for resume)."""
    if not jsonl_path.exists():
        return 0
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Validate last line isn't corrupted
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    break
    return count


def truncate_to_valid(jsonl_path: Path, valid_count: int) -> None:
    """Truncate output JSONL to only valid_count lines (remove corrupted tail)."""
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if len(lines) >= valid_count:
                break
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def append_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    """Append records to JSONL file."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# Record Building
# =============================================================================

def build_output_record(original: dict, tag_result: dict) -> dict:
    """Merge original fields + tag fields + derived fields."""
    output = dict(original)
    output["tags"] = tag_result.get("tags", [])
    output["analysis_summary"] = tag_result.get("analysis_summary", "")
    tags = output["tags"]
    output["has_stigma"] = "Yes" if tags else "No"
    output["how_many_stigma"] = len(tags)
    output["stigma_score"] = sum(float(t.get("severity", 0)) for t in tags)
    return output


# =============================================================================
# Folder Processing
# =============================================================================

def _is_rate_limit_error(error_msg: str) -> bool:
    """Check if an error message indicates a rate limit."""
    lower = error_msg.lower()
    return any(s in lower for s in ["rate_limit", "rate limit", "429", "too many requests", "overloaded", "throttl"])


def process_folder(
    client,
    folder_info: dict,
    taxonomy: dict,
    workers: int,
    batch_size: int,
    limit: int | None,
    clean_start: bool,
    folder_idx: int,
    total_folders: int,
    rate_tracker: RateLimitTracker | None = None,
) -> bool:
    """Process a single folder. Returns False if aborted due to rate limits."""
    input_dir = folder_info["input_dir"]
    output_dir = folder_info["output_dir"]
    model = folder_info["model"]
    name = folder_info["name"]

    # Find input JSONL file(s)
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.info(f"[{folder_idx}/{total_folders}] {name}: No JSONL files found, skipping")
        return True

    for jsonl_file in jsonl_files:
        if rate_tracker and rate_tracker.should_abort:
            log.error(f"Aborting {name} due to rate limit errors")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / jsonl_file.name

        log.info(f"")
        log.info(f"{'=' * 60}")
        log.info(f"[{folder_idx}/{total_folders}] {name} / {jsonl_file.name}")
        log.info(f"{'=' * 60}")

        # Load input
        items = load_jsonl(jsonl_file, model)
        total_items = len(items)

        # Handle clean start
        if clean_start and output_file.exists():
            output_file.unlink()
            log.info(f"  Clean start: deleted {output_file}")

        # Resume: count already-processed lines
        done = count_output_lines(output_file)
        if done > 0:
            truncate_to_valid(output_file, done)
            log.info(f"  Resuming from item {done}/{total_items}")

        # Apply limit
        end_idx = total_items
        if limit is not None:
            end_idx = min(done + limit, total_items)

        remaining = end_idx - done
        if remaining <= 0:
            log.info(f"  All items already processed ({done}/{total_items})")
            continue

        log.info(f"  Processing items {done+1}..{end_idx} of {total_items} ({remaining} items)")
        log.info(f"  Workers: {workers}, Batch size: {batch_size}")

        # Process in batches
        print_lock = Lock()
        tagged_count = 0

        for batch_start in range(done, end_idx, batch_size):
            if rate_tracker and rate_tracker.should_abort:
                log.error(f"Aborting mid-folder at item {batch_start} due to rate limits")
                return False

            batch_end = min(batch_start + batch_size, end_idx)
            batch_items = items[batch_start:batch_end]

            log.info(f"  [Batch {batch_start+1}-{batch_end}/{end_idx}]")

            batch_results = []

            if workers > 1:
                results_by_idx = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_idx = {
                        executor.submit(process_single_item, client, item, taxonomy): i
                        for i, item in enumerate(batch_items)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            results_by_idx[idx] = result
                            # Check if the result itself contains an API error (rate limit)
                            summary = result.get("analysis_summary", "")
                            if _is_rate_limit_error(summary) and rate_tracker:
                                rate_tracker.record_error(summary)
                        except Exception as e:
                            error_msg = str(e)
                            log.error(f"    Error on item {batch_start + idx}: {error_msg}")
                            if _is_rate_limit_error(error_msg) and rate_tracker:
                                rate_tracker.record_error(error_msg)
                            results_by_idx[idx] = {
                                "tags": [],
                                "analysis_summary": f"API_ERROR: {error_msg}",
                            }
                        with print_lock:
                            tags = results_by_idx[idx].get("tags", [])
                            if tags:
                                tagged_count += 1
                                codes = ", ".join(t["code"] for t in tags)
                                log.info(f"    Item {batch_start + idx + 1}: {codes}")
                # Preserve order
                for i in range(len(batch_items)):
                    batch_results.append(results_by_idx[i])
            else:
                for i, item in enumerate(batch_items):
                    if rate_tracker and rate_tracker.should_abort:
                        log.error(f"Aborting mid-batch due to rate limits")
                        return False
                    result = process_single_item(client, item, taxonomy)
                    batch_results.append(result)
                    summary = result.get("analysis_summary", "")
                    if _is_rate_limit_error(summary) and rate_tracker:
                        rate_tracker.record_error(summary)
                    tags = result.get("tags", [])
                    if tags:
                        tagged_count += 1
                        codes = ", ".join(t["code"] for t in tags)
                        log.info(f"    Item {batch_start + i + 1}: {codes}")

            # Build output records and append
            output_records = []
            for orig, tag_res in zip(batch_items, batch_results):
                output_records.append(build_output_record(orig, tag_res))

            append_jsonl(output_file, output_records)
            log.info(f"    Saved {len(output_records)} records (total: {batch_end}/{end_idx})")

        log.info(f"  Done: {name} — {tagged_count} items had stigma tags")

    return True


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tag results/ COT/NTCOT entries with stigma taxonomy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--folder", action="append", dest="folders", default=None,
        help="Process only this folder (repeatable). E.g. --folder deepseek-v31_COT",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N items per folder (for cost testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel Bedrock calls (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Items per checkpoint save (default: 10)",
    )
    parser.add_argument(
        "--clean-start", action="store_true",
        help="Delete existing output and start fresh",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List folders and counts without processing",
    )
    parser.add_argument(
        "--model", type=str, default="opus",
        help="Tagging model: sonnet, opus, or a full Bedrock model ID (default: opus)",
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix for output folder names, e.g. '_sonnet' → {folder}_Tagged_sonnet/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = RESULTS_DIR
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        sys.exit(1)

    # Discover folders
    folders = discover_folders(base_dir, args.folders, suffix=args.output_suffix)
    if not folders:
        print("No matching COT/NTCOT folders found.")
        sys.exit(1)

    print(f"Found {len(folders)} folders to process:")
    total_items = 0
    for f in folders:
        jsonl_files = sorted(f["input_dir"].glob("*.jsonl"))
        count = 0
        for jf in jsonl_files:
            with open(jf, "r") as fh:
                count += sum(1 for line in fh if line.strip())
        done = 0
        for jf in jsonl_files:
            out_file = f["output_dir"] / jf.name
            done += count_output_lines(out_file)
        total_items += count
        status = f" ({done}/{count} done)" if done > 0 else f" ({count} items)"
        print(f"  {f['name']}{status}")

    print(f"\nTotal: {total_items} items across {len(folders)} folders")

    if args.dry_run:
        return

    # Resolve model
    tagging_model = MODEL_ALIASES.get(args.model, args.model)

    # Load taxonomy and create client
    log.info(f"Tagging model (Bedrock): {tagging_model} (extended thinking enabled)")
    log.info("Loading taxonomy...")
    taxonomy = load_taxonomy()

    log.info("Creating Bedrock client...")

    # Rate limit tracker: abort if 5 rate limit errors in 60 seconds
    rate_tracker = RateLimitTracker(max_errors=5, window_seconds=60)

    # Process each folder
    start_time = time.time()
    aborted = False
    with create_tagging_client(model=tagging_model) as client:
        for i, folder_info in enumerate(folders, 1):
            ok = process_folder(
                client=client,
                folder_info=folder_info,
                taxonomy=taxonomy,
                workers=args.workers,
                batch_size=args.batch_size,
                limit=args.limit,
                clean_start=args.clean_start,
                folder_idx=i,
                total_folders=len(folders),
                rate_tracker=rate_tracker,
            )
            if not ok:
                aborted = True
                break

    elapsed = time.time() - start_time
    log.info("")
    log.info("=" * 60)
    if aborted:
        log.error(f"ABORTED after {elapsed/60:.1f} min due to rate limit errors ({rate_tracker.total_errors} total)")
        log.error("Progress was saved — re-run to resume from where it stopped.")
        sys.exit(1)
    else:
        log.info(f"All folders processed in {elapsed/60:.1f} min!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
