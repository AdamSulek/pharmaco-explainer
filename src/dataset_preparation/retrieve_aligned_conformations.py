import os
import argparse
import logging
from typing import Dict, Iterable, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"ID", "sdf_source", "conf_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["ID", "sdf_source", "conf_idx"]).copy()
    df["ID"] = df["ID"].astype(str)
    df["sdf_source"] = df["sdf_source"].astype(str)
    df["conf_idx"] = df["conf_idx"].astype(int)

    return df[["ID", "sdf_source", "conf_idx"]]


def iter_sdf_records_text(path: str, encoding: str = "utf-8") -> Iterable[Tuple[str, str]]:
    """
    Stream SDF as raw text records.

    Yields:
      (header_line, record_text_including_$$$$)
    """
    header: Optional[str] = None
    buf = []

    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            if header is None:
                # first line of record
                header = line.rstrip("\r\n")
                buf = [line]
                continue

            buf.append(line)

            if line.strip() == "$$$$":
                rec = "".join(buf)
                yield header.strip(), rec
                header = None
                buf = []

    # If file does not end with $$$$, drop trailing partial record silently.


def export_from_one_sdf(
    sdf_path: str,
    wanted_headers: Set[str],
    out_dir: str,
    one_file: bool = False,
    combined_handle=None,
) -> Tuple[int, int]:
    """
    Export records whose first line matches any header in wanted_headers.

    If one_file=True, appends to combined_handle (must be opened in text mode).
    Else writes individual files into out_dir.
    """
    found = 0
    total = len(wanted_headers)

    if total == 0:
        return 0, 0

    os.makedirs(out_dir, exist_ok=True)

    try:
        for header, rec in iter_sdf_records_text(sdf_path):
            if header in wanted_headers:
                found += 1

                if one_file:
                    combined_handle.write(rec)
                else:
                    # header is "ID#conf=N" by convention
                    safe_name = header.replace("/", "_")
                    out_path = os.path.join(out_dir, f"{safe_name}.sdf")
                    with open(out_path, "w", encoding="utf-8") as w:
                        w.write(rec)

                # optional early-stop if everything already found
                if found == total:
                    break

    except Exception as e:
        logging.error(f"Failed reading {sdf_path}: {e}")
        return 0, total

    return found, total


def build_wanted_headers(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Returns mapping:
      sdf_source -> set of headers "ID#conf=conf_idx"
    """
    groups: Dict[str, Set[str]] = {}
    for sdf_source, g in df.groupby("sdf_source", sort=False):
        headers = set(f"{row.ID}#conf={int(row.conf_idx)}" for row in g.itertuples(index=False))
        groups[str(sdf_source)] = headers
    return groups


def main():
    ap = argparse.ArgumentParser(description="Export selected conformers (ID#conf=N) from source SDF files.")
    ap.add_argument("--input", required=True, help="CSV/Parquet with columns: ID,sdf_source,conf_idx")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--workers", type=int, default=1, help="Parallelize across SDF sources (threads)")
    ap.add_argument("--one-file", action="store_true", help="Write all hits into one SDF file")
    ap.add_argument("--one-file-name", default="positives.sdf", help="Name for combined SDF when --one-file is set")
    args = ap.parse_args()

    df = load_table(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    groups = build_wanted_headers(df)
    logging.info(f"Unique SDF sources: {len(groups)}")
    logging.info(f"Total targets: {len(df)}")

    total_found = 0
    total_targets = 0

    combined_path = os.path.join(args.out_dir, args.one_file_name)
    combined_handle = None
    if args.one_file:
        combined_handle = open(combined_path, "w", encoding="utf-8")

    try:
        if args.workers <= 1:
            for i, (sdf_path, wanted) in enumerate(groups.items(), 1):
                found, tot = export_from_one_sdf(
                    sdf_path=sdf_path,
                    wanted_headers=wanted,
                    out_dir=args.out_dir,
                    one_file=args.one_file,
                    combined_handle=combined_handle,
                )
                total_found += found
                total_targets += tot
                logging.info(f"[{i}/{len(groups)}] {os.path.basename(sdf_path)}: {found}/{tot}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {
                    ex.submit(
                        export_from_one_sdf,
                        sdf_path,
                        wanted,
                        args.out_dir,
                        args.one_file,
                        combined_handle,   # NOTE: only safe if one_file=False
                    ): sdf_path
                    for sdf_path, wanted in groups.items()
                }

                if args.one_file:
                    raise ValueError("--one-file is not compatible with --workers > 1 (single shared output handle).")

                for i, fut in enumerate(as_completed(futures), 1):
                    sdf_path = futures[fut]
                    found, tot = fut.result()
                    total_found += found
                    total_targets += tot
                    logging.info(f"[{i}/{len(groups)}] {os.path.basename(sdf_path)}: {found}/{tot}")

    finally:
        if combined_handle is not None:
            combined_handle.close()
            logging.info(f"Combined output: {combined_path}")

    missing = total_targets - total_found
    logging.info(f"DONE: saved {total_found}/{total_targets} records (missing={missing}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
