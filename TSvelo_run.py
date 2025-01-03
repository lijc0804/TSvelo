import os
import glob
import numpy as np
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import time
import matplotlib.pyplot as plt
from TSvelo.TSvelo_utils import run_paga, run_palantir, get_colors, show_imgs, sigmoid, relu, scv_analysis
from TSvelo.TSvelo_pp import preprocess
from TSvelo.TSvelo_branch import init_branch
from TSvelo.TSvelo_concate import concate
from TSvelo.TSvelo_model import init_W, init_US, make_loss_mask, init_t, run, to_adata



def main(args, li):
    adata = ad.read_h5ad(args.save_folder+'/l'+str(li)+'_pp.h5ad')
    print(adata)
    
    adata, W_ini, W_0_mask, n_selected_genes = init_W(adata)
    Y, U, S = init_US(adata)
    loss_mask = make_loss_mask(adata, Y, U, S)

    adata, t_steps = init_t(args, adata, figure_folder=args.save_folder+'figures_l'+str(li)+'/')
    adata, best_results = run(args, adata, W_ini, W_0_mask, n_selected_genes, Y, U, S, loss_mask, t_steps, figure_folder=args.save_folder+'figures_l'+str(li)+'/')
    ii, best_EPOCH_loss, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre = best_results
    print("Best results at EPOCH", ii+1, best_EPOCH_loss)

    adata = to_adata(adata, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre, n_selected_genes, loss_mask)
    adata = scv_analysis(adata, figure_folder=args.save_folder+'figures_l'+str(li)+'/')        
    adata.write(args.save_folder+'/l'+str(li)+'_TSvelo.h5ad')
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="gastrulation_erythroid", help='gastrulation_erythroid pancreas 10x_mouse_brain pons dentategyrus larry') 
    parser.add_argument( '--preprocess', type=int, default=0, help='whether to perform preprocessing')
    parser.add_argument( '--n_jobs', type=int, default=-1, help='n_jobs')
    parser.add_argument( '--n_neighbors', type=int, default=30, help='Number of neighbors for KNN graph')
    parser.add_argument( '--n_top_genes', type=int, default=2000, help='n_top_genes')
    parser.add_argument( '--n_selected_genes', type=int, default=100, help='number of selected velocity genes')
    parser.add_argument( '--TF_databases', nargs='+', default='ENCODE ChEA', help='ChEA ENCODE')   
    parser.add_argument( '--N_steps', type=int, default=1000, help='Number of time steps')
    parser.add_argument( '--cuda', type=int, default=3, help='CUDA device id')
    parser.add_argument( '--N_EPOCH', type=int, default=30, help='Max number of epochs for EM')
    parser.add_argument( '--num_epochs', type=int, default=500, help='Max number of epochs for neural ODE')
    parser.add_argument( '--min_decrease', type=float, default=0, help='Minimum decrease for early stop')
    parser.add_argument( '--n_genes2show', type=int, default=0, help='number of genes to show during training')
    parser.add_argument( '--save_name', type=str, default='', help='save_name')

    args = parser.parse_args() 
    args.save_folder = 'TSvelo_'+ args.dataset_name + args.save_name + '/'
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    print('********************************************************************************************************')
    print(args)

    if args.preprocess or (not os.path.exists(args.save_folder+'l0_pp.h5ad')):
        preprocess(args) 
        init_branch(args)
    
    h5ad_file_path_all = glob.glob(os.path.join(args.save_folder, '*_pp.h5ad'))
    print(h5ad_file_path_all)
    for li in range(len(h5ad_file_path_all)):
        main(args, li) 

    if len(h5ad_file_path_all)>1:
        concate(args)
    else:
        os.rename(args.save_folder+'l0_TSvelo.h5ad', args.save_folder+'TSvelo.h5ad')
        os.rename(args.save_folder+'figures_l0', args.save_folder+'figures')

    print('Finished', args.dataset_name, args.save_name)
