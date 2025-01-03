import os
import numpy as np
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import time
import glob
import matplotlib.pyplot as plt
from .TSvelo_utils import scv_analysis


def load_branches(args):
    adata = ad.read_h5ad(args.save_folder +"/pp.h5ad")
    h5ad_file_path_all = glob.glob(os.path.join(args.save_folder, '*_TSvelo.h5ad'))
    adata_subs = []
    for li in range(len(h5ad_file_path_all)):
        adata_l = ad.read_h5ad(args.save_folder+'/l'+str(li)+'_TSvelo.h5ad')
        adata_subs.append(adata_l)
    #print(adata)
    #print(adata_subs[0])
    adata = adata[:, adata_subs[0].var_names]
    print(adata)
    return adata, adata_subs


def merge_layer(adata, adata_subs, layer_key, weighted=1):
    layer_merged = {}
    cell_weights = {}
    
    for i, adata_sub in enumerate(adata_subs):
        layer = adata_sub.layers[layer_key]
        cell_count = len(adata_sub.obs)  
        
        layer_merged_sub = pd.DataFrame(layer, index=adata_sub.obs.index)
        adata_sub.obs['t_combined'] = adata.obs['t']
        t0_idx = adata_sub.obs['t_combined'].argmin()
        t0_t = adata_sub[t0_idx].obs['t_steps'][0]
        adata_sub.obs['tmp'] = 1
        adata_sub.obs['tmp'][adata_sub.obs['t_steps'] < t0_t] = 0.01
        tmp = adata_sub.obs['tmp']
        
        for cell_id, layer_values in layer_merged_sub.iterrows():
            if cell_id not in layer_merged:
                layer_merged[cell_id] = []
                cell_weights[cell_id] = []
            layer_merged[cell_id].append(layer_values.values)
            cell_weights[cell_id].append(cell_count*tmp[cell_id]) 
            
    layer_mean = {}
    for cell_id, layer_values_list in layer_merged.items():
        layer_values_array = np.array(layer_values_list)
        weights_array = np.array(cell_weights[cell_id])
        
        if weighted:
            weighted_mean = np.average(layer_values_array, axis=0, weights=weights_array)
        else:
            weighted_mean = np.average(layer_values_array, axis=0)
        layer_mean[cell_id] = weighted_mean

    layer_mean_values = np.array([layer_mean[cell_id] for cell_id in adata.obs.index])
    adata.layers[layer_key] = layer_mean_values
    
    return adata


def merge_obs(adata, adata_subs, obs_key, obs_value_dtype=float, weighted=1):
    obs_merged = {}
    cell_weights = {}
    
    for i, adata_sub in enumerate(adata_subs):
        obs = adata_sub.obs[obs_key]
        cell_count = len(adata_sub.obs)  
        
        for cell_id, obs_value in obs.items():
            if cell_id not in obs_merged:
                obs_merged[cell_id] = []
                cell_weights[cell_id] = []
            if obs_value_dtype == int:
                obs_value = int(obs_value)
            obs_merged[cell_id].append(obs_value)
            cell_weights[cell_id].append(cell_count) 

    obs_mean = {}
    for cell_id, obs_values_list in obs_merged.items():
        obs_values_array = np.array(obs_values_list)
        weights_array = np.array(cell_weights[cell_id])
        
        if weighted:
            weighted_mean = np.average(obs_values_array, weights=weights_array)
        else:
            weighted_mean = np.average(obs_values_array)
        
        if obs_value_dtype == int:
            weighted_mean = int(weighted_mean)
        
        obs_mean[cell_id] = weighted_mean

    adata.obs[obs_key] = adata.obs.index.map(obs_mean).fillna(np.nan)
    
    return adata



def concate(args):
    adata, adata_subs = load_branches(args)
        
    for obs_key in ['t']:
        merge_obs(adata, adata_subs, obs_key=obs_key)
    with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
        sc.pl.umap(adata, color=['clusters', 't'], legend_loc='on data', show=False)
        plt.savefig(args.save_folder+"t_umap.png", bbox_inches="tight")  
        plt.close()
    
    for layer_key in ['du_dt', 'ds_dt', 'U', 'S', 'velocity']:
        adata = merge_layer(adata, adata_subs, layer_key, weighted=1)
        
    adata = scv_analysis(adata, figure_folder=args.save_folder, recompute_velocity=False)  
          
    print(adata)

    adata.write(args.save_folder+"TSvelo.h5ad")
    
    return 