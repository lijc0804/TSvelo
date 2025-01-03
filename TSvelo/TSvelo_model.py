import os
import glob
import numpy as np
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import time
import matplotlib.pyplot as plt
from .TSvelo_utils import run_paga, run_palantir, get_colors, show_imgs, sigmoid, relu, scv_analysis
from .TSvelo_pp import preprocess
from .TSvelo_branch import init_branch
from .TSvelo_concate import concate


def init_W(adata):
    adata.var['to_reserve'] = False
    all_TFs_used = []
    for g_id, g in enumerate(adata.var_names):
        if (g in adata.uns['all_TFs']):
            if (adata.layers['Ms'][:,g_id].mean()!=0):
                adata.var['to_reserve'][g_id] = True
                all_TFs_used.append(g)
            else:
                print('Delete TF', g) 
        if (adata.var['selected_genes'][g]):
            if (adata.layers['Mu'][:,g_id].mean()!=0) and (adata.layers['Ms'][:,g_id].mean()!=0):
                adata.var['to_reserve'][g_id] = True
            else:
                adata.var['selected_genes'][g_id] = False
                print('Delete target', g) 
    adata.uns['all_TFs_used'] = all_TFs_used
    adata = adata[:, adata.var['to_reserve']==True]
    N_cells, n_genes = adata.shape
    print(N_cells, n_genes)
    adata = adata[:, adata.var['selected_genes'].argsort()[::-1]]
    n_selected_genes = adata.var['selected_genes'].sum()
    n_non_selected_TFs = adata.var['to_reserve'].sum() - n_selected_genes

    TFs_id = - np.ones_like(adata.varm['TFs_id'])
    W_ini = np.zeros([n_genes, n_genes])
    for g_id, g in enumerate(adata.var_names):
        for ii, TF in enumerate(adata.varm['TFs'][g_id]):
            if TF == 'blank':
                break
            elif TF in adata.uns['all_TFs_used']:
                if TF == g:
                    continue
                TF_id = list(adata.var_names).index(TF)
                TFs_id[g_id, ii] = TF_id
                w = adata.varm['TFs_correlation'][g_id, ii]
                if w > 0:
                    W_ini[g_id, TF_id] = w/3
                else:
                    W_ini[g_id, TF_id] = w/6
            else:
                print(TF)
            
    adata.varm['TFs_id'] = TFs_id
    W_0_mask = (W_ini == 0)
    print(W_ini.shape)
    return adata, W_ini, W_0_mask, n_selected_genes
    
def init_US(adata):    
    U, S = adata.layers['Mu'], adata.layers['Ms']
    U, S = U/np.std(U, axis=0), S/np.std(S, axis=0)
    U, S = np.nan_to_num(U), np.nan_to_num(S)
    Y = np.concatenate([U, S], axis=1)
    print(Y.shape)
    return Y, U, S
        

def make_loss_mask(adata, Y, U, S):
    loss_mask = np.ones_like(Y, dtype=int)
    selected_genes_mask = np.array(adata.var['selected_genes'], dtype=int)
    selected_genes_mask_2 = np.concatenate([selected_genes_mask, np.ones_like(selected_genes_mask, dtype=int)], axis=0)
    loss_mask = loss_mask * selected_genes_mask_2
    print(loss_mask.shape)
    print(loss_mask)
    return loss_mask



def init_t(args, adata, figure_folder):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    t_ini = np.array(adata.obs['t_ini'])
    t_steps_ini = np.array(t_ini * (args.N_steps-1), dtype=int) 
    print(t_steps_ini)
    adata.obs['t_0'] = t_steps_ini.copy()
    with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
        sc.pl.umap(adata, color=['t_0'], legend_loc='on data', show=False)
        plt.savefig(figure_folder+"/umap_t_0.png", bbox_inches="tight")  
        plt.close()
    return adata, t_steps_ini


import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.optim as optim

class ODEFunc(nn.Module):
    def __init__(self, BETA, GAMMA, W, W_bias, W_0_mask, n_selected_genes):
        super(ODEFunc, self).__init__()
        self.n_genes = len(BETA)
        self.beta = nn.Parameter(torch.tensor(BETA, dtype=torch.float32), requires_grad=True)  #nn.Parameter(torch.ones(n_genes) * 1.0, requires_grad=True) 
        self.gamma = nn.Parameter(torch.tensor(GAMMA, dtype=torch.float32), requires_grad=True) #nn.Parameter(torch.ones(n_genes) * 1.0, requires_grad=True)
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=True)  
        self.W_bias = nn.Parameter(torch.tensor(W_bias, dtype=torch.float32), requires_grad=True) 
        self.W_0_mask = W_0_mask.to(torch.float32)
        self.n_selected_genes = n_selected_genes

    def forward(self, t, y):
        u, s = y[:self.n_genes], y[self.n_genes:]
        W_masked = self.W * (1 - self.W_0_mask)
        alpha = nn.ReLU()(W_masked @ s + self.W_bias) 
        beta, gamma = nn.ReLU()(self.beta), nn.ReLU()(self.gamma)
        du_dt = alpha - beta * u
        du_dt[self.n_selected_genes:] = 0
        ds_dt = beta * u - gamma * s
        ds_dt[self.n_selected_genes:] = alpha[self.n_selected_genes:] - gamma[self.n_selected_genes:] * s[self.n_selected_genes:]
        dy_dt = torch.cat((du_dt, ds_dt), dim=0)
        return dy_dt
    

def train_ODE(Y, t_steps, loss_mask, BETA, GAMMA, W, W_bias, W_0_mask, N_steps, n_selected_genes, device, num_epochs=200, patience=20, select_rate=1, min_decrease=2e-4): 
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device) 
    y0 = Y_t[np.argsort(t_steps)[:10]].mean(0)
    t = torch.linspace(0, 1, N_steps).to(device) 
    W_0_mask = torch.tensor(W_0_mask).to(device)
    ode_func = ODEFunc(BETA, GAMMA, W, W_bias, W_0_mask, n_selected_genes).to(device)
    optimizer = optim.Adam(ode_func.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # Learning rate scheduler
    loss_fn = nn.MSELoss() 

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        Y_pre = odeint(ode_func, y0, t)
        if select_rate<1:  
            num_rows = Y_t.size(0)
            num_selected = int(num_rows * select_rate)
            indices_selected = torch.randperm(num_rows)[:num_selected]
            loss_mask_t = torch.tensor(loss_mask[indices_selected], dtype=torch.bool).to(device)
            Y_t_masked = Y_t[indices_selected][loss_mask_t]
            Y_pre_masked = Y_pre[t_steps[indices_selected]][loss_mask_t]
        else:
            loss_mask_t = torch.tensor(loss_mask, dtype=torch.bool).to(device)
            Y_t_masked = Y_t[loss_mask_t]
            Y_pre_masked = Y_pre[t_steps][loss_mask_t]
        loss = loss_fn(Y_pre_masked, Y_t_masked) 
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # Update learning rate

        if (epoch + 1) % max(patience, 1) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

        # Early stopping logic
        if loss.item() < best_loss-min_decrease: #
            best_loss = loss.item()
            patience_counter = 0  # Reset counter if loss improves
        else:
            patience_counter += 1  # Increment counter if no improvement

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    if num_epochs > 0:
        EPOCH_loss = loss.item()
        print("EPOCH Loss:", EPOCH_loss)
    else:
        EPOCH_loss = 1e6
        
    Y_pre = odeint(ode_func, y0, t)
    
    return Y_t.detach().cpu().numpy(), Y_pre.detach().cpu().numpy(), \
        ode_func.beta.detach().cpu().numpy(), ode_func.gamma.detach().cpu().numpy(), \
        ode_func.W.detach().cpu().numpy(), ode_func.W_bias.detach().cpu().numpy(), EPOCH_loss  


from joblib import Parallel, delayed
def update_t(adata, ii, Y_t, Y_pre, loss_mask, t_steps, figure_folder, n_jobs=-1, save_fig=False):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    loss_mask_t = np.array(loss_mask[t_steps], dtype=bool)

    def compute_closest_index(i):
        dist = (Y_pre - Y_t[i])** 2  
        dist_masked = dist[:, loss_mask_t[i]] 
        mse_value = np.mean(dist_masked, axis=1)  
        closest_index = np.argmin(mse_value)  
        return closest_index

    t_steps_updated = Parallel(n_jobs=n_jobs)(delayed(compute_closest_index)(i) for i in range(Y_t.shape[0]))
    t_steps_updated = np.array(t_steps_updated)

    if save_fig:
        adata_t = adata.copy()
        adata_t.obs['ODE_t_'+str(ii+1)] = t_steps_updated#/t_steps_updated.max()
        with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
            sc.pl.umap(adata_t, color='ODE_t_'+str(ii+1), legend_loc='on data', show=False)
            plt.savefig(figure_folder+"/umap_t_"+str(ii+1)+'.png', bbox_inches="tight")  
            plt.close()
    
    return t_steps_updated


def analyze_g(adata, g_id, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre, colors, figure_folder):
    n_genes = adata.shape[1]
    u, s = U[:, g_id], S[:, g_id]
    alpha = relu((W[g_id] * S).sum(1) + W_bias[g_id]) 
    beta, gamma = relu(BETA[g_id]), relu(GAMMA[g_id])
    if adata.var['selected_genes'][g_id] == True:
        du_dt = alpha - beta*u
        ds_dt = beta*u - gamma*s
    else:
        du_dt = np.full(u.shape, np.nan) #np.zeros_like(u)
        ds_dt = alpha - gamma*s
    u_t, s_t = Y_pre[:, g_id], Y_pre[:, n_genes+g_id]
    show_imgs(adata.var_names[g_id], s, u, alpha, ds_dt, du_dt, beta, gamma, u_t, s_t, t_steps, colors, to_show=0, figure_folder=figure_folder)
    return         


def run(args, adata, W_ini, W_0_mask, n_selected_genes, Y, U, S, loss_mask, t_steps, figure_folder):
    n_genes=adata.shape[1]
    BETA, GAMMA, W, W_bias = 2*np.ones(n_genes), 2*np.ones(n_genes), W_ini, np.ones(n_genes)
    colors = get_colors(adata)

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    best_EPOCH_loss = 1e6
    wait_EPOCH = 0
    for ii in range(args.N_EPOCH):   
        print('EPOCH', ii+1)
        Y_t, Y_pre, BETA, GAMMA, W, W_bias, EPOCH_loss = train_ODE(Y, t_steps, loss_mask, BETA, GAMMA, W, W_bias, W_0_mask, args.N_steps, n_selected_genes, device, num_epochs=args.num_epochs, patience=20, min_decrease=args.min_decrease)
        print(W)
        t_steps = update_t(adata, ii, Y_t, Y_pre, loss_mask, t_steps, figure_folder=figure_folder, n_jobs=args.n_jobs)
        print(t_steps)
        adata = to_adata(adata, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre, n_selected_genes, loss_mask)
        #scv_analysis(adata, figure_folder=figure_folder+str(ii+1)+'/')        
        tmp = 0
        for g_id, g in enumerate(adata.var_names):
            if tmp >= args.n_genes2show:
                break
            tmp += 1
            analyze_g(adata, g_id, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre, colors, figure_folder=figure_folder+str(ii+1)+'/')

        if EPOCH_loss < best_EPOCH_loss:
            best_EPOCH_loss = EPOCH_loss
            best_results = [ii, EPOCH_loss, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre]
            wait_EPOCH = 0
        else:
            wait_EPOCH += 1
        if wait_EPOCH >= 3:
            break
        
    adata.obs['t'] = adata.obs['t_steps']/adata.obs['t_steps'].max()
    with plt.rc_context({"figure.dpi": (300)}): #"figure.figsize": (8, 8)
        sc.pl.umap(adata, color='t', show=False) 
        plt.savefig(args.save_folder+"/t.png", bbox_inches="tight")  
        plt.close()
            
    end_time = time.time()
    print('runing time(s):', end_time - start_time)    
    return adata, best_results



def to_adata(adata, U, S, W, W_bias, BETA, GAMMA, t_steps, Y_pre, n_selected_genes, loss_mask):
    adata.layers['U'], adata.layers['S'] = U, S
    adata.layers['alpha'], adata.layers['du_dt'], adata.layers['ds_dt'] = np.full(adata.shape, np.nan), np.full(adata.shape, np.nan), np.full(adata.shape, np.nan)
    adata.var['beta'], adata.var['gamma'], adata.var['W_bias'] = np.nan, np.nan, np.nan
    adata.varm['W'] = np.full(W.shape, np.nan)
    adata.obs['t_steps'] = t_steps
    adata.uns['U_t'] = Y_pre[:, :len(W)]
    adata.uns['U_t'][:,n_selected_genes:] = np.full(adata.uns['U_t'][:, n_selected_genes:].shape, np.nan)
    adata.uns['S_t'] = Y_pre[:, len(W):]
    adata.var['loss'] = 0
    adata.uns['loss_mask'] = loss_mask
    
    for g_id, g in enumerate(adata.var_names):
        u, s = U[:, g_id], S[:, g_id]
        alpha = relu((W[g_id] * S).sum(1) + W_bias[g_id])  
        beta, gamma = relu(BETA[g_id]), relu(GAMMA[g_id])
        if adata.var['selected_genes'][g_id]==True:
            du_dt = alpha - beta*u
            ds_dt = beta*u - gamma*s
        else:
            du_dt = np.full(u.shape, np.nan) #np.zeros_like(u)
            ds_dt = alpha - gamma*s
        adata.layers['alpha'][:,g_id] = alpha
        adata.var['beta'][g_id] = beta
        adata.var['gamma'][g_id] = gamma
        adata.layers['du_dt'][:,g_id] = du_dt
        adata.layers['ds_dt'][:,g_id] = ds_dt
        adata.varm['W'][g_id] = W[g_id]
        adata.var['W_bias'][g_id] = W_bias[g_id]
            
        u_t, s_t = adata.uns['U_t'][:, g_id], adata.uns['S_t'][:, g_id]
        u_pre, s_pre = u_t[t_steps], s_t[t_steps]
        if adata.var['selected_genes'][g_id]==True:
            loss = np.mean((u_pre-u)**2 + (s_pre-s)**2)
        else:
            loss = np.mean((s_pre-s)**2)
        adata.var['loss'][g_id] = loss
    return adata

