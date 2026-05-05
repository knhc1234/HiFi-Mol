# HiFi-Mol
Multi-View Molecular Representation Learning with Hierarchical Graphs and Contextualized Fingerprints

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
