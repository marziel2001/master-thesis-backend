"""Print WER and CER for a reference and hypothesis text file.

Usage::

    python -m scripts.count_metrics reference.txt hypothesis.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jiwer import cer, wer


def calculate_metrics_from_text(
    reference_text: str, hypothesis_text: str
) -> dict[str, float]:
    return {
        "wer": wer(reference_text, hypothesis_text),
        "cer": cer(reference_text, hypothesis_text),
    }


def calculate_metrics_from_file(
    reference_path: str | Path, hypothesis_path: str | Path
) -> dict[str, float]:
    return calculate_metrics_from_text(
        Path(reference_path).read_text(encoding="utf-8"),
        Path(hypothesis_path).read_text(encoding="utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print WER and CER for a reference and hypothesis file."
    )
    parser.add_argument("reference_file", help="Path to the reference text file.")
    parser.add_argument("hypothesis_file", help="Path to the hypothesis text file.")
    args = parser.parse_args()

    metrics = calculate_metrics_from_file(args.reference_file, args.hypothesis_file)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
