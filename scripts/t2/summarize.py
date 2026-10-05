#!/usr/bin/env python3
"""T2 — tổng hợp CSV từng episode thành bảng theo cấu hình × σ (logic: analysis/robust_summary.py).

    python scripts/t2/summarize.py <csv> [<csv> ...] [--md out.md] [--out-csv out.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.robust_summary import load, summarize, to_markdown, write_summary_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--md", default=None, help="ghi bảng markdown ra file")
    ap.add_argument("--out-csv", default=None, help="ghi bảng tổng hợp dạng csv")
    args = ap.parse_args()
    table = summarize(load(args.csv))
    md = to_markdown(table)
    print(md)
    if args.md:
        Path(args.md).write_text(md + "\n")
    if args.out_csv:
        write_summary_csv(table, args.out_csv)


if __name__ == "__main__":
    main()
