# pharmaco-explainer

## Data Preparation

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
    scripts/qc_retrieve_positive_conformers.sh
  ```

### 4. Graph generation

- Generates **graph-based representations** of molecules from prepared tabular data for a selected pharmacophore setting (`k3`, `k4`, or `k5`).

- Reads an input parquet file (`ks{K}.parquet`) and converts each molecule into a graph format suitable for downstream graph-based or deep learning models.

- Run on HPC using the provided template:

  ```bash
  sbatch --export=ALL,K=4,BASE_DIR=/path/to/project_root \
    scripts/graph_generation.sh
  ```

### 5. MAT / RMAT featurization

- Generates **MAT or RMAT molecular features** using attention-based transformer models from HuggingMolecules.

- Two featurization modes are supported:
  - `no_pos` – features are generated from a **3D conformer internally constructed from SMILES** (native HuggingMolecules behavior).  
    This mode produces **one feature representation per molecule ID**, independent of pharmacophore alignment.
  - `with_pos` – features are generated from the **exact 3D conformer selected during pharmacophore alignment**.  
    Each generated feature corresponds to a **pharmacophore-matched conformer** and therefore represents a **structural positive by construction** (`y = 1`).

- Featurization is performed **independently of dataset splitting and labeling**.  
  Feature objects are stored together with molecule IDs as `(feature, ID)` tuples and can be **reused across multiple datasets** (`k3`, `k4`, `k5`).

- For large-scale libraries (e.g. millions of molecules), MAT/RMAT featurization is computationally expensive and should be **performed once**, followed by **dataset-specific assignment of labels and splits**.

- Run on HPC using the provided template:

  ```bash
  sbatch --array=0-9 \
    --export=ALL,BASE_DIR=/path/to/project_root,K=4,MODEL_TYPE=mat,MODE=no_pos \
    hpc/featurize_mat_rmat.template.sh
  ```

## Run Training

### 1. GCN training

- Trains a **graph convolutional network (GCN)** on precomputed molecular graph representations.

- Uses graph features generated during data preparation and dataset-specific train/validation/test splits (`split_easy`, `split_hard`, or `all`).

- Training is executed per pharmacophore dataset (`k3`, `k4`, `k5`), with checkpoints and results stored separately for each split configuration.

- Run on HPC using the provided template:

  ```bash
  sbatch \
    --export=ALL,K=4,BASE_DIR=/path/to/project_root,SPLIT_TYPE=split_easy \
    scripts/train_gcn.sh
  ```

### 2. MAT / RMAT training

- Trains a **Molecule Attention Transformer (MAT)** or **Relational MAT (RMAT)** on precomputed molecular features stored as `(feature, ID)` tuples.

- Feature computation is performed once at scale; labels (`y`) and dataset splits are applied later based on molecule IDs, enabling efficient reuse across `k3`, `k4`, and `k5` datasets.

- Supports standard training as well as runs augmented with precomputed **with_pos** features, where matched conformers are injected for positive samples.

- Run on HPC using the provided template:

  ```bash
  sbatch \
    --export=ALL,BASE_DIR=/path/to/project_root,MODEL=mat,K=k4,DIFFICULTY=normal \
    scripts/train_mat_or_rmat.sh
  ```

## Explainability

### 2. MAT / RMAT

Run per-atom attributions (Vanilla Gradients or Grad-CAM) for MAT/RMAT models on SLURM.

### Script

`compute_mat_rmat_attributions.sh`

### Usage (SLURM)

Set parameters via `sbatch --export`:

```bash
sbatch --export=ALL,\
METHOD=gradcam,\
MODEL=mat,\
CHECKPOINT=checkpoints_mat/mat_k4_normal_with_pos/best_model.pth,\
TEST_PICKLE=pharmaco_explainer/pickle_dataloaders/mat/k4/normal/test/test.p,\
POSITIVE_PICKLE_POS_PATH=pharmaco_explainer/pickle_dataloaders/mat/k4_positive/k4_positive.p,\
OUTPUT_FILE=results_k4/GradCam_mat_all_with_pos.parquet \
compute_mat_rmat_attributions.sh
```
