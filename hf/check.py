import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

files = [
    "3M_ID_smi.parquet",
    "ID_smi_enamine_JJ.parquet",
    "ID_smi_enamine.parquet",
]

dfs = []

for input_file in files:
    logging.info(f"Loading {input_file}")
    df = pd.read_parquet(input_file)

    logging.info(f"{input_file}: rows = {len(df)}, cols = {df.shape[1]}")
    logging.info(f"{input_file}: columns = {list(df.columns)}")

    if len(df) > 0:
        logging.info(f"{input_file}: sample row: {df.iloc[0].to_dict()}")

    # opcjonalnie: dodaj info skąd pochodzi rekord
    df["dataset"] = input_file

    dfs.append(df)

# ===== MERGE =====
logging.info("Concatenating dataframes")
merged_df = pd.concat(dfs, ignore_index=True)

logging.info(
    f"Merged dataframe: rows = {len(merged_df)}, cols = {merged_df.shape[1]}"
)

# ===== OPCJONALNIE: usuwanie duplikatów =====
# po ID
before = len(merged_df)
merged_df = merged_df.drop_duplicates(subset=["ID"])
logging.info(f"Removed {before - len(merged_df)} duplicate IDs")

# albo (jeśli chcesz ostrzej):
# merged_df = merged_df.drop_duplicates(subset=["ID", "smiles"])

# ===== ZAPIS =====
output_file = "pharmaco_explainer_data.parquet"
logging.info(f"Saving merged data to {output_file}")
merged_df.to_parquet(output_file, index=False)

logging.info("DONE")
