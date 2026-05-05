# Datasets

Please follow the instructions below to prepare all required datasets.

## PCQM4Mv2 (Pretraining)

Download the PCQM4Mv2 dataset in this directory:

```bash
wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
md5sum pcqm4m-v2-train.sdf.tar.gz  # fd72bce606e7ddf36c2a832badeec6ab
tar -xf pcqm4m-v2-train.sdf.tar.gz
```

This will extract `pcqm4m-v2-train.sdf` in this directory.

## Fingerprint Embeddings (Pretraining)

Download the precomputed fingerprint embeddings:

```bash
wget https://zenodo.org/records/20044253/files/downstream_fingerprint_embeddings.parquet
```

## MoleculeNet (Finetuning)

Download `molecule_datasets.zip` and extract it in this directory:

```bash
unzip molecule_datasets.zip -d molecule_datasets/
```

## Expected Directory Structure

```
datasets/
├── README.md
├── pcqm4m-v2-train.sdf
├── downstream_fingerprint_embeddings.parquet
└── molecule_datasets/
    ├── bace/
    │   └── raw/
    ├── bbbp/
    │   └── raw/
    ├── clintox/
    │   └── raw/
    └── ...
```
