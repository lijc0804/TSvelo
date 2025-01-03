import os
import numpy as np
import pandas as pd 
import anndata as ad
import numpy as np
import scanpy as sc
import time 
import matplotlib.pyplot as plt
from .TSvelo_utils import run_paga, run_palantir, get_colors, show_imgs, show_imgs_simple, show_imgs_alpha_s, sigmoid, relu, leaky_relu
from scipy.stats import pearsonr, spearmanr


def cluster_paga(args, adata, cluster_key):
    if args.dataset_name=='larry':
        resolution = 0.15
    else:
        resolution = 0.1
    if cluster_key=='leiden':   
        sc.tl.leiden(adata, resolution=resolution)  

    sc.tl.paga(adata, groups=cluster_key)
    print(adata)
    #sc.pl.paga(adata, color=cluster_key)
    print(adata.uns['paga']['connectivities'].todense())
    return adata

def detect_lineages_from_cluster(paga_graph, start_cluster, thres=0):
    import networkx as nx
    import scipy.sparse as sp
    paga_graph[paga_graph<thres] = 0
    paga_graph = sp.csr_matrix(paga_graph.todense())
    #print('PAGA graph:', paga_graph.todense())

    G = nx.from_scipy_sparse_array(paga_graph)
    #print('G.edges:', G.edges)
    #print(G.nodes)
    if start_cluster not in G.nodes:
        raise ValueError(f"Cluster {start_cluster} not found in PAGA graph")

    shortest_paths = {}
    
    paths = nx.single_source_shortest_path(G, start_cluster)

    for target_cluster, path in paths.items():
        if target_cluster != start_cluster: 
            shortest_paths[target_cluster] = path
    return shortest_paths


def remove_subsets(lst):
    to_remove = set() 
    for i, sublist in enumerate(lst):
        sublist_set = set(sublist)
        for j, other_sublist in enumerate(lst):
            if i != j and sublist_set.issubset(other_sublist):
                to_remove.add(i)
                break  
    return [lst[i] for i in range(len(lst)) if i not in to_remove]

def get_lineages(adata, cluster_key, start_cluster_annotation):
    paga_graph = adata.uns['paga']['connectivities'].copy()
    start_cluster = list(adata.obs[cluster_key].cat.categories).index(start_cluster_annotation) 
    lineages_dict = detect_lineages_from_cluster(paga_graph, start_cluster, thres=0.02)
    #print('Lineages dict:', lineages_dict)

    lineages = list(lineages_dict.values())
    lineages = remove_subsets(lineages)
    #print('Lineages:',lineages)

    def replace_index_with_strings(index_list_of_lists, string_list):
        return [[string_list[i] for i in sublist] for sublist in index_list_of_lists]

    lineages_cluster = replace_index_with_strings(lineages, list(adata.obs[cluster_key].cat.categories))
    print('Lineages cluster:', lineages_cluster)
    
    return lineages_cluster

def norm_init_t(t_ini):
    sorted_indices = np.argsort(t_ini)
    ranked_t_ini = np.zeros_like(sorted_indices)
    ranked_t_ini[sorted_indices] = np.arange(len(t_ini))
    t_ini_out = ranked_t_ini/ranked_t_ini.max()
    return t_ini_out

def init_t(args, adata, cluster_key, start_cluster_annotation):
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import cdist
    connectivities = adata.obsp['connectivities'].toarray()  
    X = adata.layers['Ms'][adata.obs[cluster_key] == str(start_cluster_annotation)].mean(0)
    distances = cdist(adata.layers['Ms'], X.reshape(1, -1), metric='euclidean')
    start_cell_indice = np.argmin(distances)
    if 1:
        adata_ = adata.copy()
        adata_.uns['iroot'] = start_cell_indice
        sc.tl.dpt(adata_)
        t_ini = adata_.obs['dpt_pseudotime']
    else:
        distances, predecessors = shortest_path(connectivities, directed=False, indices=[start_cell_indice], return_predecessors=True)
        t_ini = pd.Series(distances.mean(axis=0), index=adata.obs.index)

    adata.obs['t_ini'] = norm_init_t(t_ini)
    #sc.pl.umap(adata, color=['clusters', cluster_key, 't_ini'])
    with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
        sc.pl.umap(adata, color=['clusters', cluster_key, 't_ini'], legend_loc='on data', show=False)
        plt.savefig(args.save_folder+"/umap.png", bbox_inches="tight")  
        plt.close()
    return adata


def get_theta_diff(adata_l):
    adata_l = adata_l[:, adata_l.var['selected_genes']]
    all_result = []
    for g_id in range(adata_l.shape[1]):
        Mu = adata_l.layers['Mu'][np.argsort(adata_l.obs['t_ini'])][:, g_id] #[::-1]
        Ms = adata_l.layers['Ms'][np.argsort(adata_l.obs['t_ini'])][:, g_id]

        U, S = Mu.copy(), Ms.copy()
        n_bins = 100
        U = np.array([bin.mean() for bin in np.array_split(Mu, n_bins)])
        S = np.array([bin.mean() for bin in np.array_split(Ms, n_bins)])
        
        if (Mu.std() > Mu.mean()) or (Ms.std() > Ms.mean()):
            corr = 0
            result = 0
        
        else:
            dt_range = 20
            corr_max = -1e6
            for tmp in range(dt_range+1):
                U_tmp, S_tmp = U[:n_bins-tmp], S[tmp:]
                U_tmp, S_tmp = (U_tmp-U_tmp.mean())/(U_tmp.std()+1e-9), (S_tmp-S_tmp.mean())/(S_tmp.std()+1e-9)
                corr = spearmanr(U_tmp, S_tmp)[0] 
                if corr > corr_max:
                    corr_max = corr
                    result = tmp 
            for tmp in range(dt_range+1):
                U_tmp, S_tmp = U[tmp:], S[:n_bins-tmp]
                U_tmp, S_tmp = (U_tmp-U_tmp.mean())/(U_tmp.std()+1e-9), (S_tmp-S_tmp.mean())/(S_tmp.std()+1e-9)
                corr = spearmanr(U_tmp, S_tmp)[0] 
                if corr > corr_max:
                    corr_max = corr
                    result = - tmp 
        
        all_result.append(result)
   
        if 0:#g_id<5:
            print(result, corr)

            fig, ax1 = plt.subplots(figsize=(8, 6))
            t = np.arange(len(U))/len(U)   
            t_ = np.arange(len(Mu))/len(Mu)   
            ax1.scatter(t_, Mu, color='blue', label='U', alpha=0.3, s=1)
            ax1.scatter(t, U, color='blue', label='U')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('U', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax2 = ax1.twinx()  
            ax2.scatter(t_, Ms, color='red', label='S', alpha=0.3, s=1)
            ax2.scatter(t, S, color='red', label='S')
            ax2.set_ylabel('S', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            plt.title(adata_l.var_names[g_id])

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(Ms, Mu, c=t_, cmap='viridis', alpha=0.3, s=1)
            scatter = plt.scatter(S, U, c=t, cmap='viridis')
            plt.colorbar(scatter, label='t')
            plt.xlabel('S')
            plt.ylabel('U')
            plt.title(adata_l.var_names[g_id])
            plt.show()

    return np.array(all_result).mean()

def evaluate_lineages(args, adata, lineages_cluster, cluster_key, start_cluster_annotation):
    adata = init_t(args, adata, cluster_key, start_cluster_annotation)

    theta_diff_l = []
    for ii, l in enumerate(lineages_cluster):
        print('l', l)
        adata_l = adata[adata.obs[cluster_key].isin(l)]
        #sc.pl.umap(adata_l, color=['clusters', cluster_key, 't_ini'])
        print(adata_l.shape)
        theta_diff = get_theta_diff(adata_l)
        print(theta_diff)
        theta_diff_l.append(theta_diff)
        
    return np.array(theta_diff_l).mean()


def save_lineages(args, adata, lineages_cluster, cluster_key, to_save=1):
    for ii, l in enumerate(lineages_cluster):
        print('l', l)
        adata_l = adata[adata.obs[cluster_key].isin(l)]
        #sc.pl.umap(adata_l, color=['clusters', cluster_key, 't_ini'])
        print(adata_l)
        
        if to_save:
            with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
                sc.pl.umap(adata_l, color=['clusters', cluster_key, 't_ini'], legend_loc='on data', show=False)
                plt.savefig(args.save_folder+'/l'+str(ii)+"_umap.png", bbox_inches="tight")  
                plt.close()
            adata_l.write(args.save_folder+'/l'+str(ii)+"_pp.h5ad")
    return 


def init_branch(args, cluster_key='leiden'):
    adata = ad.read_h5ad(args.save_folder +"/pp.h5ad")
    print(adata)
    adata = cluster_paga(args, adata, cluster_key)
    adata_all, lineages_cluster_all, theta_diff_all = [], [], []
    for start_cluster_annotation in adata.obs[cluster_key].cat.categories:
        print('s', start_cluster_annotation)
        adata_s = adata.copy()
        lineages_cluster = get_lineages(adata_s, cluster_key, start_cluster_annotation)
        lineages_cluster_all.append(lineages_cluster)
        theta_diff_s = evaluate_lineages(args, adata_s, lineages_cluster, cluster_key, start_cluster_annotation)
        theta_diff_all.append(theta_diff_s)
        adata_all.append(adata_s)

    print(theta_diff_all)
    index = theta_diff_all.index(max(theta_diff_all))
    lineages_cluster = lineages_cluster_all[index]
    adata = adata_all[index]
    print('INIT:', lineages_cluster)
    save_lineages(args, adata, lineages_cluster, cluster_key)