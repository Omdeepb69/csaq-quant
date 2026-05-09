"""
examples/wikitext2_export.py — Export WikiText-2 to plain text files.

Creates wikitext2_train.txt and wikitext2_test.txt for use as English
calibration and evaluation corpora in CSAQ research benchmarks.

Usage::

    pip install datasets
    python examples/wikitext2_export.py --output_dir ./data
    # Creates: ./data/wikitext2_train.txt and ./data/wikitext2_test.txt
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(description="Export WikiText-2 to plain text files")
    p.add_argument(
        "--output_dir", default="./data",
        help="Directory to write the text files (default: ./data)",
    )
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required.\n"
            "Install it with: pip install datasets"
        )

    print("Downloading WikiText-2 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    os.makedirs(args.output_dir, exist_ok=True)

    for split, filename in [("train", "wikitext2_train.txt"), ("test", "wikitext2_test.txt")]:
        path = os.path.join(args.output_dir, filename)
        lines = [t for t in ds[split]["text"] if len(t.strip()) > 20]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in lines)
        print(f"  {path}  ({len(lines)} lines)")

    print("Done.")


if __name__ == "__main__":
    main()
