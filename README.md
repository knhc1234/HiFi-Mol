# HiFi-Mol
Multi-View Molecular Representation Learning with Hierarchical Graphs and Contextualized Fingerprints

> **Note:** The code and instructions for Fingerprint Pretraining will be released in a future update.

## Environments

Our code is tested and supported under the following environment:

- **Python**: 3.9
- **CUDA**: 11.7
- **OS**: Linux

Create and activate a conda environment, then install the required dependencies:

```bash
conda create -n HiFi-Mol python==3.9
conda activate HiFi-Mol

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_sparse-0.6.17%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_cluster-1.6.1%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_spline_conv-1.2.2%2Bpt113cu117-cp39-cp39-linux_x86_64.whl

pip install torch_scatter-2.1.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.17+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.1+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.2+pt113cu117-cp39-cp39-linux_x86_64.whl
pip install torch-geometric==1.7.2

pip install numpy==1.24.4
pip install rdkit==2023.3.2
pip install scikit-learn==1.0.2
pip install pyarrow
pip install tensorboard
```

## Usage

Run the following steps in order.

> **Tip:** The PCQM4Mv2 download (Step 1), Data Preprocessing (Step 2), and Pretraining (Step 3) can be skipped by downloading the pretrained model weights (`HiFi-Mol_GIN_best.pth`, `HiFi-Mol_GIN_final.pth`) and placing them in the `main/Pretrain/` directory. See [`main/Pretrain/README.md`](Pretrain/README.md) for download links. In that case, still complete the **Fingerprint Embeddings** and **MoleculeNet** downloads in Step 1, then proceed directly to **Step 4**.

### Step 1. Dataset Download

#### PCQM4Mv2 (Pretraining)

Navigate to the `datasets/` directory and download the PCQM4Mv2 dataset:

```bash
cd datasets
wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
md5sum pcqm4m-v2-train.sdf.tar.gz  # fd72bce606e7ddf36c2a832badeec6ab
tar -xf pcqm4m-v2-train.sdf.tar.gz
```

This will extract `pcqm4m-v2-train.sdf` in the `datasets/` directory.

#### Fingerprint Embeddings (Pretraining)

Download the precomputed fingerprint embeddings and place them in the `datasets/` directory:

```bash
wget <URL>  # Link to be updated
```

The file should be located at `datasets/downstream_fingerprint_embeddings.parquet`.

#### MoleculeNet (Finetuning)

Download `molecule_datasets.zip` and extract it so that the dataset is located at `datasets/molecule_datasets/`:

```bash
unzip molecule_datasets.zip -d molecule_datasets/
```

#### Expected Directory Structure

```
datasets/
├── pcqm4m-v2-train.sdf
├── downstream_fingerprint_embeddings.parquet
└── molecule_datasets/
    ├── bace/
    ├── bbbp/
    ├── clintox/
    └── ...
```

### Step 2. Data Preprocessing

Navigate to the `main/` directory and run the preprocessing script to convert the PCQM4Mv2 dataset into hierarchical graph format:

```bash
cd ../main
python PCQM4Mv2_preparation.py
```

### Step 3. Pretraining

Run the pretraining script to train HiFi-Mol on the preprocessed data. The trained model checkpoints will be saved in the `main/Pretrain/` directory.

```bash
python pretrain.py
```

### Step 4. Finetuning

Run the finetuning script to perform downstream property prediction. Use `finetune.py` for classification tasks and `finetune_reg.py` for regression tasks.

> **Note:** For MoleculeNet datasets, only the raw data files are provided. On the first run, hierarchical graph preprocessing will be automatically performed by `main/datasets/molecule_dataset.py`, and the processed data will be saved in the `processed/` folder under each dataset directory (e.g., `datasets/molecule_datasets/bbbp/processed/`). This preprocessing only occurs once.

```bash
# Classification tasks (e.g., BACE, BBBP, Clintox, ...)
python finetune.py

# Regression tasks (e.g., ESOL, FreeSolv, Lipophilicity)
python finetune_reg.py
```

The finetuned model checkpoints will be saved in `main/results/HiFi-Mol_last/`.

## Results

The evaluation results for each dataset will be saved as a CSV file in the `main/` directory:

```
main/
└── HiFi-Mol_{dataset_name}.csv
```
