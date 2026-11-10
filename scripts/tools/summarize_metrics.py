#!/usr/bin/env python3
"""Summarize model metric CSV files under REDS4_new.

This script searches recursively for files named `test_metrics_DatasetTest.csv`,
extracts one global summary row per model variant (preferably the `mean` row),
and optionally a per-sequence summary when a CSV contains more than one distinct
sequence folder. It writes:
1) One CSV folder per metric sorted by that metric.
2) A global CSV inside each metric folder.
3) Optional per-sequence CSVs inside each metric folder when multiple sequences
    are present.

Usage:
    python summarize_metrics.py
    python summarize_metrics.py --root /path/to/REDS4_new
    python summarize_metrics.py --pattern "test_metrics_*.csv"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


DEFAULT_PATTERN = "test_metrics_*.csv"
IMAGE_NAME_COLUMNS = ("original_image", "image")


def clean_metric_name(metric: str) -> str:
    """Convert original metric names to short names for output columns/files."""
    metric = metric.strip()
    if "/" in metric:
        metric = metric.split("/")[-1]
    return metric.lower()


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def get_image_value(row: Dict[str, str]) -> str:
    for column in IMAGE_NAME_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            return value
    return ""


def get_sequence_name(row: Dict[str, str]) -> Optional[str]:
    image = get_image_value(row)
    if not image or image.lower() == "mean":
        return None
    sequence = image.split("/", 1)[0].strip()
    return sequence or None


def metric_sort_desc(metric: str) -> bool:
    """Return True if higher values are better for this metric."""
    name = clean_metric_name(metric)
    higher_is_better = {"psnr", "ssim"}
    lower_is_better_tokens = ("loss", "lpips", "dists", "error", "mse", "mae")

    if name in higher_is_better:
        return True
    if any(token in name for token in lower_is_better_tokens):
        return False
    # Default: assume higher is better if metric type is unknown.
    return True


def load_summary_row(csv_path: Path) -> Tuple[Dict[str, float], List[str]]:
    """Load summary metrics from one CSV.

    Preference order:
    1) Row with image == 'mean'
    2) Arithmetic mean over all non-summary rows
    """
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {csv_path}")

        columns = reader.fieldnames or []
        metric_columns = [c for c in columns if c not in IMAGE_NAME_COLUMNS]

        mean_row = None
        for row in rows:
            image = (row.get("image") or "").strip().lower()
            if image == "mean":
                mean_row = row
                break

        if mean_row is not None:
            summary = {}
            for c in metric_columns:
                if is_float(mean_row.get(c, "")):
                    summary[c] = float(mean_row[c])
            return summary, metric_columns

        # Fallback: compute mean from per-sequence rows.
        summary = {}
        for c in metric_columns:
            values = [float(r[c]) for r in rows if is_float(r.get(c, ""))]
            if values:
                summary[c] = mean(values)
        return summary, metric_columns


def load_metric_columns(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV: {csv_path}")

        columns = reader.fieldnames or []
        metric_columns = [c for c in columns if c not in IMAGE_NAME_COLUMNS]
        return metric_columns, rows


def load_all_images(csv_path: Path) -> set:
    """Extract all image names from a CSV (excluding 'mean' row)."""
    images = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = ""
            for column in IMAGE_NAME_COLUMNS:
                value = (row.get(column) or "").strip().lower()
                if value:
                    image = value
                    break
            if image and image != "mean":
                images.add(image)
    return images


def load_images_by_sequence(csv_path: Path) -> Dict[str, set]:
    """Load images grouped by sequence name. Returns {sequence: {images}}."""
    images_by_seq: Dict[str, set] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = get_image_value(row).lower()
            if image == "mean":
                continue
            sequence = get_sequence_name(row)
            if sequence:
                images_by_seq.setdefault(sequence, set()).add(image)
    return images_by_seq


def rows_for_common_images(rows: List[Dict[str, str]], common_images_by_sequence: Dict[str, set]) -> List[Dict[str, str]]:
    """Filter rows to only images that are common within each sequence."""
    filtered_rows: List[Dict[str, str]] = []
    for row in rows:
        image = get_image_value(row).lower()
        if image == "mean":
            continue
        sequence = get_sequence_name(row)
        if sequence and sequence in common_images_by_sequence:
            if image in common_images_by_sequence[sequence]:
                filtered_rows.append(row)
    return filtered_rows


def summarize_rows(rows: List[Dict[str, str]], metric_columns: List[str]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for c in metric_columns:
        values = [float(r[c]) for r in rows if is_float(r.get(c, ""))]
        if values:
            summary[c] = mean(values)
    return summary


def summarize_global_rows(rows: List[Dict[str, str]], metric_columns: List[str]) -> Dict[str, float]:
    for row in rows:
        if get_image_value(row).strip().lower() == "mean":
            summary: Dict[str, float] = {}
            for c in metric_columns:
                if is_float(row.get(c, "")):
                    summary[c] = float(row[c])
            return summary

    return summarize_rows(rows, metric_columns)


def write_csv(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def build_sorted_rows(
    summaries: List[Dict[str, object]],
    metric: str,
) -> List[Dict[str, object]]:
    reverse_metric = metric_sort_desc(metric)

    def sort_value(row: Dict[str, object]) -> float:
        raw_value = row.get(metric, "")
        raw_text = str(raw_value)
        if is_float(raw_text):
            return float(raw_text)
        return float("-inf" if reverse_metric else "inf")

    return sorted(
        summaries,
        key=sort_value,
        reverse=reverse_metric,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metric CSV files by variant.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder to search (default: script folder).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="CSV filename pattern to search recursively.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    csv_files = sorted(root.rglob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No CSV files found with pattern '{args.pattern}' in {root}")

    # First pass: build common_images_by_sequence (per-sequence intersection)
    common_images_by_sequence: Dict[str, set] = {}
    for csv_path in csv_files:
        try:
            images_by_seq = load_images_by_sequence(csv_path)
            for sequence, images in images_by_seq.items():
                if sequence not in common_images_by_sequence:
                    common_images_by_sequence[sequence] = images.copy()
                else:
                    common_images_by_sequence[sequence] &= images  # Intersection per sequence
        except Exception as exc:
            print(f"[WARN] Error reading images from {csv_path}: {exc}")
            continue

    if not common_images_by_sequence:
        raise SystemExit("No common sequences found across all CSV files.")

    print(f"Found {len(csv_files)} CSV files.")
    print(f"Per-sequence common images:")
    for seq in sorted(common_images_by_sequence.keys()):
        print(f"  {seq}: {len(common_images_by_sequence[seq])} images")
    print()

    summaries: List[Dict[str, object]] = []
    sequence_summaries: List[Dict[str, object]] = []
    all_metric_columns: List[str] = []

    # Second pass: extract summaries using per-sequence common images
    for csv_path in csv_files:
        try:
            metric_columns, rows = load_metric_columns(csv_path)
        except Exception as exc:  # Keep script robust if one file is malformed.
            print(f"[WARN] Skipping {csv_path}: {exc}")
            continue

        filtered_rows = rows_for_common_images(rows, common_images_by_sequence)
        if not filtered_rows:
            print(f"[WARN] Skipping {csv_path}: no rows matched the common image sets")
            continue

        summary = summarize_rows(filtered_rows, metric_columns)

        for c in metric_columns:
            if c not in all_metric_columns:
                all_metric_columns.append(c)

        variant = csv_path.parent.relative_to(root).as_posix()
        row: Dict[str, object] = {
            "variant": variant,
            "source_csv": csv_path.relative_to(root).as_posix(),
        }
        row.update(summary)
        summaries.append(row)

        rows_by_sequence: Dict[str, List[Dict[str, str]]] = {}
        for filtered_row in filtered_rows:
            sequence = get_sequence_name(filtered_row)
            if sequence is None:
                continue
            rows_by_sequence.setdefault(sequence, []).append(filtered_row)

        if len(rows_by_sequence) > 1:
            for sequence, sequence_rows in sorted(rows_by_sequence.items()):
                sequence_summary = summarize_rows(sequence_rows, metric_columns)
                sequence_row: Dict[str, object] = {
                    "variant": variant,
                    "sequence": sequence,
                    "source_csv": csv_path.relative_to(root).as_posix(),
                }
                sequence_row.update(sequence_summary)
                sequence_summaries.append(sequence_row)

    if not summaries:
        raise SystemExit("No valid summaries could be extracted.")

    # Write one ranking CSV per metric.
    for metric in all_metric_columns:
        metric_name = clean_metric_name(metric)
        metric_dir = root / "metrics" / metric_name

        sorted_rows = build_sorted_rows(summaries, metric)
        out_path = metric_dir / "global.csv"

        out_rows: List[Dict[str, object]] = []
        for rank, row in enumerate(sorted_rows, start=1):
            out_rows.append(
                {
                    "rank": rank,
                    "variant": row["variant"],
                    "source_csv": row["source_csv"],
                    metric_name: row.get(metric, ""),
                }
            )

        write_csv(out_path, ["rank", "variant", "source_csv", metric_name], out_rows)

        if sequence_summaries:
            sequence_header = ["rank", "variant", "sequence", "source_csv", metric_name]
            sequence_rows_by_name: Dict[str, List[Dict[str, object]]] = {}
            for sequence_row in sequence_summaries:
                sequence_name = str(sequence_row["sequence"])
                sequence_rows_by_name.setdefault(sequence_name, []).append(sequence_row)

            for sequence_name, sequence_rows in sorted(sequence_rows_by_name.items()):
                sequence_out_rows: List[Dict[str, object]] = []
                for rank, row in enumerate(build_sorted_rows(sequence_rows, metric), start=1):
                    sequence_out_rows.append(
                    {
                            "rank": rank,
                            "variant": row["variant"],
                            "sequence": row["sequence"],
                            "source_csv": row["source_csv"],
                            metric_name: row.get(metric, ""),
                        }
                    )

                sequence_out_path = metric_dir / f"{sequence_name}.csv"
                write_csv(sequence_out_path, sequence_header, sequence_out_rows)

    print(f"Valid summaries: {len(summaries)}")
    print("Per-metric folders:")
    for metric in all_metric_columns:
        metric_name = clean_metric_name(metric)
        print(f"- metrics/{metric_name}/global.csv")
    if sequence_summaries:
        print("Per-sequence summaries:")
        sequence_names = sorted({str(row["sequence"]) for row in sequence_summaries})
        for metric in all_metric_columns:
            metric_name = clean_metric_name(metric)
            for sequence in sequence_names:
                print(f"- metrics/{metric_name}/{sequence}.csv")


if __name__ == "__main__":
    main()
