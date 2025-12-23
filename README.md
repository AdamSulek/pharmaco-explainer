# pharmaco-explainer

### 1. Generating conformers for pharmacophore alignment

For each molecule, a fixed number of conformers is generated to enable robust pharmacophore alignment.

In this pipeline, up to **50 conformations per molecule** are generated and stored in SDF format.  
These conformers are used as the input ensemble for subsequent pharmacophore alignment and screening.

Conformer generation is performed with:

```bash
python data_preparation/prepare_sdf_for_screening.py \
  --in-parquet path/to/input.parquet \
  --out-dir path/to/output_sdfs \
  --target-confs 50
```

### 2. Pharmacophore alignment and label generation

This step performs **pharmacophore-based alignment** of molecular conformers and assigns labels based on geometric matching.

For each molecule ID, all available conformers are evaluated against a predefined pharmacophore hypothesis.  
Candidate atom assignments (HBA, HBD, aromatic) are enumerated and checked using distance tolerances.

A molecule is labeled as **positive (`y = 1`)** if and only if **exactly one unique pharmacophore configuration** matches across all its conformers.  
Otherwise, the molecule is labeled as **negative (`y = 0`)**.

#### How to run

```bash
python data_preparation/pharmacophore_alignment_pipeline.py \
  --part-dir path/to/sdf_files/part_000 \
  --plots-root path/to/plots \
  --out-root path/to/labels_out \
  --pharm k4 \
  --hypo-json hypothesis/k4.json \
  --cpus 8 \
  --tol-core 1.0 \
  --tol-ar 2.0
```

### 3. (QC): Retrieve positive conformers

- Retrieves **positive conformers** from large SDF files based on pharmacophore alignment results.  
  For each molecule, the conformer selected during labeling (e.g. `winner_conf_id`) is extracted.

- Used for **quality control (QC)** and downstream inspection of conformers that satisfied pharmacophore constraints, without reprocessing full SDF archives.

- The script reads label files for a given pharmacophore mode (`k3`, `k4`, or `k5`) and `part_*`, identifies molecules with `y = 1`, and extracts the corresponding SDF records from the source files into a separate output directory.

- Run on HPC using the provided template:

  ```bash
  sbatch --array=0-52 \
    --export=ALL,K=k4,BASE_DIR=/path/to/project_root \
    hpc/qc_retrieve_positive_conformers.template.sh
