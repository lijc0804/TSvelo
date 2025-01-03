import os
import sys
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import scvelo as scv
import anndata as ad
import numpy as np
import scanpy as sc
import scipy
import time
from .TSvelo_pp_utils import get_TFs, select_gene, geneid_symbol


def read_data(args):
    if args.dataset_name == 'gastrulation_erythroid':
        adata = scv.datasets.gastrulation_erythroid()   
        adata.obs['clusters'] = adata.obs['celltype'].copy()
    elif args.dataset_name == 'pancreas':
        adata = scv.datasets.pancreas() 
    elif args.dataset_name == 'pons':
        adata = ad.read_h5ad("data/pons/oligo_lite.h5ad")  
        adata.obs['clusters'] = adata.obs['celltype']
        adata.obs['clusters'] = pd.Categorical(adata.obs['clusters'], categories=['OPCs', 'COPs', 'NFOLs', 'MFOLs'], ordered=True)
    elif args.dataset_name == '10x_mouse_brain':
        adata = ad.read_h5ad("data/10x_mouse_brain/adata_rna.h5ad") 
        adata.obs['clusters'] = adata.obs['celltype'].copy()
    elif args.dataset_name == 'dentategyrus':
        adata = ad.read_h5ad("data/DentateGyrus/DentateGyrus.h5ad") 
        adata.obs['clusters'] = adata.obs['ClusterName'].copy()
        adata.obs['clusters'] = adata.obs['clusters'].replace({'Nbl1': 'Neuroblast', 'Nbl2': 'Neuroblast'})
        adata.obs['clusters'] = adata.obs['clusters'].replace({'RadialGlia2': 'RadialGlia'})
        adata.obs['clusters'] = adata.obs['clusters'].replace({'ImmGranule1': 'ImmGranule', 'ImmGranule2': 'ImmGranule'})
        desired_order = ['Neuroblast', 'CA', 'CA1-Sub', 'CA2-3-4', 
                        'ImmGranule', 'Granule',
                        'RadialGlia', 'ImmAstro',
                        'nIPC', 'GlialProg', 'OPC']  
        adata.obs['clusters'] = pd.Categorical(adata.obs['clusters'], categories=desired_order, ordered=True)
    elif args.dataset_name == 'larry':
        adata = ad.read_h5ad("data/larry/larry.h5ad")  
        adata.obs['clusters'] = adata.obs['state_info']
        adata.obsm['X_umap'] = np.stack([np.array(adata.obs['SPRING-x']), np.array(adata.obs['SPRING-y'])]).T

    if args.dataset_name in ['pancreas', 'pons', 'gastrulation_erythroid']:
        adata.uns['clusters_colors'] = np.array(['red', 'orange', 'yellow', 'green','skyblue', 'blue','purple', 'pink', '#8fbc8f', '#f4a460', '#fdbf6f', '#ff7f00', '#b2df8a', '#1f78b4',
            '#6a3d9a', '#cab2d6'][:len(adata.obs['clusters'].cat.categories)], dtype=object)   

    print(adata)
    return adata


def preprocess(args):
    print('Preprocessing', args.dataset_name)    
    adata = read_data(args)
    
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=args.n_top_genes)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=args.n_neighbors) # cell amount will influence the setti
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors)
        sc.tl.umap(adata)
    with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
        sc.pl.umap(adata, color='clusters', show=False)
        plt.savefig(args.save_folder+"clusters.png", bbox_inches="tight")  
        plt.close()

    gene_names = []
    for tmp in adata.var_names:
        gene_names.append(tmp.upper())
    adata.var_names = gene_names
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    print(adata)
        
    adata = get_TFs(adata, args.TF_databases)
    
    adata = select_gene(adata, args.n_selected_genes, n_neighbors=args.n_neighbors, n_jobs=args.n_jobs)
    
    selected_genes_list = adata.var.loc[adata.var['selected_genes'], :].index.tolist()
    
    print(adata)  
    adata.write(args.save_folder + 'pp.h5ad')
    print('Preprocessed', args.dataset_name)
    return 