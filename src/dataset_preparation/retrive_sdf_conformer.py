import logging
import pandas as pd
import os
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_sdf_record_by_id_conf(input_sdf: str, output_sdf: str, mol_id: str, conf_idx: int) -> bool:
    """
    Extract a single SDF record from `input_sdf` whose first line is:
        f"{mol_id}#conf={conf_idx}"
    and write it (up to and including the next '$$$$' line) to `output_sdf`.

    Returns:
        True  – if the record was found and written,
        False – otherwise.
    """
    target_header = f"{mol_id}#conf={conf_idx}"
    in_record = False
    buf = []

    with open(input_sdf, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\r\n")
            if not in_record:
                # Look for the header line for this specific conformer
                if s.strip() == target_header:
                    in_record = True
                    buf.append(line)
            else:
                buf.append(line)
                if s.strip() == "$$$$":
                    break

    # Fallback: close record after 'M  END' if '$$$$' is missing
    if in_record and (not buf or buf[-1].strip() != "$$$$"):
        for i in range(len(buf) - 1, -1, -1):
            if buf[i].strip() == "M  END":
                buf = buf[: i + 1] + ["\n", "$$$$\n"]
                break

    if not buf:
        return False

    with open(output_sdf, "w", encoding="utf-8") as w:
        w.writelines(buf)
    logging.info(f"✅ Extracted: {output_sdf}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export specific conformers from SDF files listed in a partial DataFrame."
    )
    parser.add_argument(
        "--k",
        required=True,
        choices=["k3", "k4", "k5"],
        help="Pharmacophore dataset identifier (k3, k4, or k5).",
    )
    parser.add_argument(
        "--part-idx",
        type=int,
        required=True,
        help="Index of the partial file (e.g. SLURM_ARRAY_TASK_ID).",
    )
    args = parser.parse_args()

    # Load partial parquet: data/<k>/positive_sdf_parts/{part-idx}.parquet
    df_sdf = pd.read_parquet(f"data/{args.k}/positive_sdf_parts/{args.part_idx}.parquet")

    # Output directory: sdf_files/<k>_positive/{part-idx}/
    out_dir = f"sdf_files/{args.k}_positive/{args.part_idx}"
    os.makedirs(out_dir, exist_ok=True)

    for index, row in df_sdf.iterrows():
        sdf_path = row["sdf_source"]
        conf_id = int(row["conf_idx"])

        output_sdf = os.path.join(out_dir, f"{row['ID']}_conf{conf_id}.sdf")

        extract_sdf_record_by_id_conf(
            input_sdf=sdf_path,
            output_sdf=output_sdf,
            mol_id=row["ID"],
            conf_idx=conf_id,
        )

        if index % 1000 == 0:
            logging.info(f"Processed {index} records...")

    logging.info("Done!")
