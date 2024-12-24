TSvelo is designed to analyze gene expression trajectories and predict dynamic changes in gene expression across time. It leverages temporal information from single-cell RNA-seq data to model transitions between cell states and infer underlying biological processes.

## Requirements

Before running TSvelo, make sure you have the following dependencies installed:

```bash
conda create -n TSvelo_env python=3.8.19
pip install pandas==2.0.3 
pip install anndata==0.9.2
pip install scanpy==1.9.8
pip install numpy==1.24.4
pip install scipy==1.10.1
pip install numba==0.58.1 
pip install matplotlib==3.7.5
pip install scvelo==0.3.2
pip install typing_extensions
pip install torch==2.4.1
pip install torchdiffeq==0.2.4
pip install mygene==3.2.2
pip install leidenalg==0.10.2
pip install pygam==0.9.1 
