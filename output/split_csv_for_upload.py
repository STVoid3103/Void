"""
split_csv_for_upload.py -- splits a large CSV into several smaller CSV
files, each under a target size, so it fits your 30MB chat upload limit.
Built for nifty_options_daily.csv (~300MB) but works on any CSV.

USAGE:
    python split_csv_for_upload.py data/pit/nifty_options_daily.csv --max-mb 25

WHAT IT DOES:
  - Streams the file line by line (never loads all 300MB into memory).
  - Repeats the header row in every part, so each part is a complete,
    independently-readable CSV on its own (pandas.read_csv works on each
    part directly -- no need to strip/re-add headers when I reassemble).
  - Never splits a data row across two files -- only breaks between whole
    rows, right after the running byte count would exceed --max-mb.

OUTPUT: next to the input file, named
    nifty_options_daily_part01.csv, _part02.csv, _part03.csv, ...
Send them all back, in order. I'll concatenate them (dropping the repeated
headers) to reconstruct the original file exactly.

25MB default, not 30MB: leaves headroom since MB-as-1024^2 vs upload
limits sometimes counting MB-as-10^6 can disagree by ~5%, and this way a
part that lands right at the boundary doesn't get rejected.
"""
from __future__ import annotations

import argparse
import os


def split_csv(input_path: str, max_bytes: int) -> list[str]:
    base, ext = os.path.splitext(input_path)
    out_paths: list[str] = []

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        header = f.readline()
        if not header:
            return out_paths
        header_bytes = len(header.encode("utf-8"))

        part_num = 1
        out_path = f"{base}_part{part_num:02d}{ext}"
        out_f = open(out_path, "w", encoding="utf-8", newline="")
        out_f.write(header)
        current_size = header_bytes
        out_paths.append(out_path)

        for line in f:
            line_size = len(line.encode("utf-8"))
            if current_size + line_size > max_bytes and current_size > header_bytes:
                out_f.close()
                part_num += 1
                out_path = f"{base}_part{part_num:02d}{ext}"
                out_f = open(out_path, "w", encoding="utf-8", newline="")
                out_f.write(header)
                current_size = header_bytes
                out_paths.append(out_path)
            out_f.write(line)
            current_size += line_size

        out_f.close()

    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("--max-mb", type=float, default=25.0,
                    help="target max size per part, in MB (default 25, under the 30MB upload cap)")
    args = ap.parse_args()

    if not os.path.exists(args.input_file):
        print(f"[ERROR] file not found: {args.input_file}")
        return

    max_bytes = int(args.max_mb * 1024 * 1024)
    paths = split_csv(args.input_file, max_bytes)

    if not paths:
        print("[ERROR] input file appears to be empty (no header line found).")
        return

    print(f"Split {args.input_file} into {len(paths)} parts:")
    total_mb = 0.0
    for p in paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        total_mb += size_mb
        print(f"  {p}  ({size_mb:.1f} MB)")
    print(f"\nTotal: {total_mb:.1f} MB across {len(paths)} files. Send them all back, in order "
         f"(part01, part02, ...) -- I'll stitch them back together.")


if __name__ == "__main__":
    main()
