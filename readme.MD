# TSvelo


This is the code page for the paper "TSvelo: Comprehensive RNA velocity inference by jointly modeling splicing, transcription and its regulation network".

TSvelo is a comprehensive RNA velocity framework that can jointly model splicing, transcription processes and its regulation network using highly interpretable neural Ordinary Differential Equations (ODEs) with an Expectation-Maximization (EM) solution. TSvelo can precisely capture the transcription-unspliced-spliced 3D dynamics of all genes simultaneously, infer unified latent time shared by genes within single cell, detect key gene regulatory relations and be applied to multi-lineage datasets.

![Image text](https://https://github.com/lijc0804/TSvelo/blob/main/figures/fig1.png)

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
```

TSvelo_run.py provides the demo for runing TSvelo.  TSvelo_analysis.ipynb and TSvelo_branch_analysis.ipynb could visualize results for analyzing datasets with single lineage and multiple lineages, respectively. The package TSvelo can be directly downloaded for usage.


TSvelo needs the TF-target annotation from ENCODE and ChEA database, which are also provided in this page. Please decompress the ENCODE TF-target database file firstly after downloading this package. On linux, you can run: 

```
unzip ENCODE.zip
```
