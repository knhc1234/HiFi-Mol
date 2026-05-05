# HiFi-Mol
Multi-View Molecular Representation Learning with Hierarchical Graphs and Contextualized Fingerprints

## Environments

Create and activate a conda environment, then install the required dependencies:

```bash
conda create -n HiFi-Mol python==3.9
conda activate HiFi-Mol
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install torch-geometric
```
