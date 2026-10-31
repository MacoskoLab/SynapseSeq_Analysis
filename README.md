# Synapse-seq – Code and Vignettes

**Synapse-seq** is a suite of technologies designed to measure both single-neuron molecular expression profiles and long-range neuroanatomical projections. The **pre-synaptic system** enables detection of distal projections, making it possible to analyze how a neuron's transcriptomic identity relates to its projection behavior.

## 📘 Vignettes

This repository provides walkthroughs for key analytical components of Synapse-seq.  
We recommend starting with the **presynaptic vignette**, which characterizes **cortico-thalamic projections from the primary visual cortex (VISp)**.

## Code

**code/preprocessing**:
 contains the scripts used to deduplicate VT dialouts and transform fastqs into cell(bead) / VT matrices.

**code/CreateScoringDF.ipynb**: walks through the creation of VT scoring for projection analysis.

**code/run_analysis**: reproduces Presynaptic main figures based on data accessed below.

**code/postsynaptic_analysis/post_analysis.ipynb**: contains notebook used to create Postsynaptic experiment figures.


---

## Data Access

All required data is publicly available in a Google Cloud Storage bucket.

### Programmatic Access

Data can be accessed using the Google Cloud SDK [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk) at:
gs://macosko_public/Synapseseq_data_2026

For executing vignettes, required files will be downloaded at beginning of the notebook.

### Manual Access

You can also browse and download the data manually from the following link:  
[Google Cloud Console – Synapseseq Data (2026)](https://console.cloud.google.com/storage/browser/macosko_public/Synapseseq_data_2026)


File Info:

## Presynapse Analysis (gs://macosko_public/Synapseseq_data_2026/Presynaptic)

- **projection_target_vts.pkl**
  Pickle file containing VTs detected at different projection sites.

- **vt_freq_objs.pkl**
  Pickle file containing the number of instances each VT was observed in the deeply sequenced pool. Used for calculating library diversity and creating VT score.
  
### VISp – Presynaptic Primary Visual Cortex Injection Experiment

- **v1_adata.h5ad**  
  H5AD (Anndata) of snRNA-seq obtained from the VISp experiment.
- **v1_vt_df.csv**  
  DataFrame containing relationships of cell-barcode to viral tags (VT).
- **thalamus_combined_obj.pkl**  
  Pickle file encoding RNA expression obtained via Slide-seq at the thalamus.
- **thalamus_bead_location.csv**  
  DataFrame containing the relationship between bead barcode and spatial position.
- **thalamus_vt_df_bead_merged.csv**  
  DataFrame containing mappings of VTs found in the thalamus and the bead on which each molecule was identified.

---

### AC – Presynaptic Anterior Cortex Injection Experiment

- **aca_adata.h5ad**  
  H5AD (Anndata) of snRNA-seq obtained from the AC experiment.
- **aca_vt_df.csv**  
  DataFrame containing relationships of cell-barcode to VTs.
- **str_combined_obj.pkl**  
  Pickle file encoding RNA expression obtained via Slide-seq in the striatum.
- **str_bead_locations.csv**  
  DataFrame containing the relationship between bead barcode and spatial position.
- **str_vt_df_bead_merged.csv**  
  DataFrame containing mappings of VTs found in the striatum and the bead on which each molecule was identified.

---

## Postsynaptic Analysis (gs://macosko_public/Synapseseq_data_2026/Postsynaptic)
### Postsynaptic Hippocampus Experiment
- **pyr_post_processed.csv**  
  DataFrame containing the DBSCAN cluster calls of VTs within the pyramidal layer (CA1, CA2, CA3), along with their spatial position (obtained from Slide-seq).
- **vt_df_dgl.csv**  
  DataFrame containing the DBSCAN cluster calls of VTs within the infrapyramidal horn of the DG.
- **vt_df_dgu.csv**  
  DataFrame containing the DBSCAN cluster calls of VTs within the suprapyramidal horn of the DG.
- **processed_hippo_RNA.qs**  
  Seurat object with spatial transcriptomics measurements made by Slide-seq.
- **rctd_results.csv**  
  DataFrame containing mappings of bead-barcode to cell-type proportions (RCTD results).




### Create environment

The requirements.yaml file can be used to create a conda environment for which we can use to execute the notebooks.

Ensure a conda solver is installed. We recommend mamba. (https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)


The `requirements.yaml` file can be used to create a conda environment for running the notebooks.

```
 mamba env create -f requirements.yaml -n synapseseq_env
 mamba activate synapseseq_env
```

---

## Reproducing Figures

### Presynaptic (`code/run_analysis.py`)

Download inputs into `objects/Presynaptic/` (from the repo root):

```
mkdir -p objects/Presynaptic
gcloud storage cp \
    gs://macosko_public/Synapseseq_data_2026/Presynaptic/{leaf_to_subclass.pkl,projection_target_vts.pkl,vt_freq_objs.pkl,vt_processed_objs_select_cols.pkl,cellular_bg_rates.pkl} \
    objects/Presynaptic/

# AC only
gcloud storage cp \
    gs://macosko_public/Synapseseq_data_2026/Presynaptic/{str_combined_obj.pkl,str_vt_df_bead_merged.csv,medulla_projs_w_umi_support.csv} \
    objects/Presynaptic/
```

Run `code/CreateScoringDF.ipynb` to generate `v1_vt_df.csv`, `aca_vt_df.csv`, and `aca_vt_scoring_df.csv` (or download them directly from the bucket).

```
objects/Presynaptic/
├── leaf_to_subclass.pkl
├── projection_target_vts.pkl
├── vt_freq_objs.pkl
├── vt_processed_objs_select_cols.pkl
├── cellular_bg_rates.pkl
├── str_combined_obj.pkl
├── str_vt_df_bead_merged.csv
├── medulla_projs_w_umi_support.csv
├── v1_vt_df.csv
├── aca_vt_df.csv
└── aca_vt_scoring_df.csv
```

Then, from `code/`:

```
# VISp
python run_analysis.py --vt_df_path ../objects/Presynaptic/v1_vt_df.csv --obj_dir ../objects/Presynaptic --vt_obj_id v1 --out_dir_base ../figs

# AC
python run_analysis.py --vt_df_path ../objects/Presynaptic/aca_vt_df.csv --obj_dir ../objects/Presynaptic --vt_obj_id aca --out_dir_base ../figs --aca_vt_score_path ../objects/Presynaptic/aca_vt_scoring_df.csv
```

Figures are written to `figs/<vt_df name>/`.

### Postsynaptic

Postsynaptic figures are generated with [`code/postsynaptic_analysis/post_analysis.ipynb`](code/postsynaptic_analysis/post_analysis.ipynb).
