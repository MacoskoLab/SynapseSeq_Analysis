# Synapse-seq – Code and Vignettes

**Synapse-seq** is a suite of technologies designed to measure both single-neuron molecular expression profiles and long-range neuroanatomical projections. The **pre-synaptic system** enables detection of distal projections, making it possible to analyze how a neuron's transcriptomic identity relates to its projection behavior.

## 📘 Vignettes

This repository provides walkthroughs for key analytical components of Synapse-seq.  
We recommend starting with the **presynaptic vignette**, which characterizes **cortico-thalamic projections from the primary visual cortex (VISp)**.

---

## Data Access

All required data is publicly available in a Google Cloud Storage bucket.

### Programmatic Access

Data can be accessed using the Google Cloud SDK [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk) at:
gs://macosko_public/Synapseseq_data_2025

For executing vignettes, required files will be downloaded at beginning of the notebook.

### Manual Access

You can also browse and download the data manually from the following link:  
[Google Cloud Console – Synapseseq Data (2025)](https://console.cloud.google.com/storage/browser/macosko_public/Synapseseq_data_2025)

If one is unable to download the required data from the public google cloud bucket, we have also made a google drive link available with the data needed to run the VISp presynaptic vignette.
https://drive.google.com/drive/u/0/folders/1U1V7Rei6ShXLDdc-2EoSN3UIrL0uW_OE


File Info:

## VISp – Presynaptic Primary Visual Cortex Injection Experiment

- **v1_adata.h5ad**  
  H5AD (Anndata) of snRNA-seq obtained from the VISp experiment.
- **v1_processed_vt_df.csv**  
  DataFrame containing relationships of cell-barcode to viral tags (VT).
- **v1_projs.pkl**  
  Pickle file encoding a Python dictionary for VTs obtained at various projection targets.
- **thalamus_combined_obj.pkl**  
  Pickle file encoding RNA expression obtained via Slide-seq at the thalamus.
- **thalamus_bead_location.csv**  
  DataFrame containing the relationship between bead barcode and spatial position.
- **thalamus_vt_df_bead_merged.csv**  
  DataFrame containing mappings of VTs found in the thalamus and the bead on which each molecule was identified.

---

## AC – Presynaptic Anterior Cortex Injection Experiment

- **aca_adata.h5ad**  
  H5AD (Anndata) of snRNA-seq obtained from the AC experiment.
- **aca_processed_vt_df.csv**  
  DataFrame containing relationships of cell-barcode to VTs.
- **aca_projs.pkl**  
  Pickle file encoding a Python dictionary of VTs obtained at various projection targets.
- **str_combined_obj.pkl**  
  Pickle file encoding RNA expression obtained via Slide-seq in the striatum.
- **str_bead_locations.csv**  
  DataFrame containing the relationship between bead barcode and spatial position.
- **str_vt_df_bead_merged.csv**  
  DataFrame containing mappings of VTs found in the striatum and the bead on which each molecule was identified.

---

## Postsynaptic_Hippocampus – Postsynaptic Hippocampus Experiment

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

Next we need to export the environment so it is accessible via jupyter.
```
python -m ipykernel install --user --name synapseseq_env --display-name "synapseseq_env"
```

Then run jupyter-lab
```
jupyter-lab --ip='*' --port=8989
```
