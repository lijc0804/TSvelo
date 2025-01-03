import os
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import scanpy as sc
import scipy
import time

ZERO = 0.02

def geneid_symbol(adata):
    import mygene
    import anndata as ad
    mg = mygene.MyGeneInfo()
    ensg_ids = adata.var_names.tolist()
    results = mg.querymany(ensg_ids, scopes='ensembl.gene', fields='symbol', species='human')
    gene_symbols = {result['query']: result.get('symbol', result['query']) for result in results}
    adata.var_names = [gene_symbols[ensg_id] for ensg_id in ensg_ids]
    return adata


def get_TFs(adata, databases, layer2use='Ms'):
    print('Get TFs according to', databases)
    start_time = time.time()
    n_gene = adata.shape[1]
    adata.varm['TFs'] = np.full([n_gene, n_gene], 'blank', dtype='<U20')
    adata.varm['TFs_id'] = np.full([n_gene, n_gene], -1)
    adata.varm['TFs_times'] = np.full([n_gene, n_gene], 0)
    adata.varm['TFs_correlation'] = np.full([n_gene, n_gene], 0.0)
    adata.var['n_TFs'] = np.zeros(n_gene, dtype=int)
    gene_names = list(adata.var_names)
    all_TFs = []

    if databases == 'all_TFs':
        with open("data/TF_names_v_1.01.txt", "r") as f:  
            for line in f.readlines():
                TF_name = line.strip('\n') 
                if not TF_name in gene_names:
                    continue
                if not TF_name in all_TFs:
                    all_TFs.append(TF_name)
                TF_expression = np.ravel(adata[:, TF_name].layers[layer2use])
                for target in gene_names:
                    target_idx = gene_names.index(target)
                    if (target==TF_name):
                        continue
                    if (TF_name in adata.varm['TFs'][target_idx]):
                        ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
                        adata.varm['TFs_times'][target_idx, ii] += 1  
                        continue
                    target_expression = np.ravel(adata[:, target].layers[layer2use])
                    flag = (TF_expression>0.1) & (target_expression>0.1)
                    if flag.sum() < 2:
                        correlation = 0
                    else:
                        correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
                    tmp_n_TF = adata.var['n_TFs'][target_idx]
                    adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
                    adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
                    adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
                    adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
                    adata.var['n_TFs'][target_idx] += 1 
        f.close()

    else:
        if 'ENCODE' in databases:
            processd_ENCODE_path='ENCODE/processed/'
            TF_files = os.listdir(processd_ENCODE_path)
            for TF_file in TF_files:
                TF_name = TF_file.replace('.txt', '')
                if not TF_name in gene_names:
                    continue
                if not TF_name in all_TFs:
                    all_TFs.append(TF_name)
                TF_expression = np.ravel(adata[:, TF_name].layers[layer2use])
                with open(processd_ENCODE_path+TF_file, "r") as f:  
                    for line in f.readlines():
                        line = line.strip('\n')  
                        target = line.upper()
                        if target in gene_names:
                            target_idx = gene_names.index(target)
                            if (target==TF_name):
                                continue
                            if (TF_name in adata.varm['TFs'][target_idx]):
                                ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
                                adata.varm['TFs_times'][target_idx, ii] += 1  
                                continue
                            target_expression = np.ravel(adata[:, target].layers[layer2use])
                            flag = (TF_expression>0.1) & (target_expression>0.1)
                            if flag.sum() < 2:
                                correlation = 0
                            else:
                                correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
                            tmp_n_TF = adata.var['n_TFs'][target_idx]
                            adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
                            adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
                            adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
                            adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
                            adata.var['n_TFs'][target_idx] += 1 
                    f.close()

        if 'ChEA' in databases:
            with open('ChEA/ChEA_2016.txt', "r") as f:  
                for line in f.readlines():
                    line = line.strip('\n')  
                    line = line.split('\t')
                    TF_info = line[0]
                    TF_name = TF_info.split(' ')[0]
                    if not TF_name in gene_names:
                        continue
                    if not TF_name in all_TFs:
                        all_TFs.append(TF_name)
                    TF_expression = np.ravel(adata[:, TF_name].layers[layer2use])
                    targets = line[2:]
                    for target in targets:
                        if target in gene_names:
                            target_idx = gene_names.index(target)
                            if (target==TF_name):
                                continue
                            if (TF_name in adata.varm['TFs'][target_idx]):
                                ii = list(adata.varm['TFs'][target_idx]).index(TF_name)
                                adata.varm['TFs_times'][target_idx, ii] += 1  
                                continue
                            target_expression = np.ravel(adata[:, target].layers[layer2use])
                            flag = (TF_expression>0.1) & (target_expression>0.1)
                            if flag.sum() < 2:
                                correlation = 0
                            else:
                                correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
                            tmp_n_TF = adata.var['n_TFs'][target_idx]
                            adata.varm['TFs'][target_idx][tmp_n_TF] = TF_name
                            adata.varm['TFs_id'][target_idx][tmp_n_TF] = gene_names.index(TF_name)
                            adata.varm['TFs_times'][target_idx, tmp_n_TF] = 1 
                            adata.varm['TFs_correlation'][target_idx, tmp_n_TF] = correlation
                            adata.var['n_TFs'][target_idx] += 1 
                f.close()


    adata.uns['all_TFs'] = all_TFs
    max_n_TF = adata.var['n_TFs'].max()
    adata.varm['TFs'] = adata.varm['TFs'][:,:max_n_TF]
    adata.varm['TFs_id'] = adata.varm['TFs_id'][:,:max_n_TF]
    adata.varm['TFs_times'] = adata.varm['TFs_times'][:,:max_n_TF]
    adata.varm['TFs_correlation'] = adata.varm['TFs_correlation'][:,:max_n_TF]
    print('max_n_TF:', max_n_TF)
    print('mean_n_TF:', np.mean(adata.var['n_TFs']))
    print('gene num of 0 TF:', (adata.var['n_TFs']==0).sum())
    print('total num of TFs:', len(all_TFs))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Getting TFs in time: {elapsed_time} seconds")   
    return adata



def evaluate_single_gene(g_id, u_data, s_data, adj_csr, n_neighbors):
    data1, data2 = u_data[g_id], s_data[g_id]
    
    if (data1.std() == 0) or (data2.std() == 0):
        return g_id, 0.0, 0.0
    else:
        data1, data2 = data1/data1.std(), data2/data2.std()
        data = np.hstack([data1.reshape(-1, 1), data2.reshape(-1, 1)])
        adata_g = ad.AnnData(X=data)
        sc.pp.neighbors(adata_g, n_neighbors=n_neighbors)
        adj_g_csr = adata_g.obsp['connectivities']
        non_sparse_rate = ((data1 > ZERO * data1.max()) & (data2 > ZERO * data2.max())).sum() / data1.shape[0]
        similarity = adj_g_csr.multiply(adj_csr).todense()
        non_sparse_id = np.array((data1 > ZERO * data1.max()) & (data2 > ZERO * data2.max()))
        similarity_selected = similarity[non_sparse_id]
        similarity_selected = similarity_selected[:, non_sparse_id]
        g_select_metric = similarity_selected.sum()
        return g_id, g_select_metric, non_sparse_rate


def evaluate_g_KNN_CPUs(adata, n_neighbors, n_jobs):
    from joblib import Parallel, delayed
    start_time = time.time()
    adata.var['g_select_metric'] = 0.0
    adata.var['non_sparse_rate'] = 0.0
    adj_csr = adata.obsp['connectivities']
    u_data, s_data = adata.layers['Mu'].T, adata.layers['Ms'].T
    
    def evaluate_single_gene_parallel(g_id):
        return evaluate_single_gene(g_id, u_data=u_data, s_data=s_data, adj_csr=adj_csr, n_neighbors=n_neighbors)
    
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_single_gene_parallel)(g_id) for g_id in range(adata.shape[1]))
    
    for g_id, g_select_metric, non_sparse_rate in results:
        adata.var['g_select_metric'][g_id] = g_select_metric
        adata.var['non_sparse_rate'][g_id] = non_sparse_rate
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"evaluate_g_KNN_CPUs time: {elapsed_time} seconds")
    return adata


def select_gene(adata, n_selected_genes, n_neighbors, n_jobs=-1):
    adata = evaluate_g_KNN_CPUs(adata, n_neighbors, n_jobs)
    var_df = adata.var
    var_df = var_df[var_df['n_TFs']>0]
    var_df = var_df[var_df['non_sparse_rate']>0.5]
    var_df = var_df[var_df['g_select_metric']>0]
    sorted_var_df = var_df.sort_values(by='g_select_metric', ascending=False)
    sorted_genes = sorted_var_df.index.tolist()
    selected_genes = sorted_genes[:n_selected_genes]
    adata.var['selected_genes'] = False
    adata.var.loc[selected_genes, 'selected_genes'] = True
    print('Selected genes:', selected_genes)
    return adata
