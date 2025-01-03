import os
import numpy as np
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import time
import matplotlib.pyplot as plt

def scv_analysis(adata, figure_folder=None, recompute_velocity=True):  
    if recompute_velocity:
        selected_genes_mask = adata.var['selected_genes']
        adata_v = adata[:, selected_genes_mask]
        U_t = adata_v.uns['U_t'][:, selected_genes_mask][adata_v.obs['t_steps']]
        S_t = adata_v.uns['S_t'][:, selected_genes_mask][adata_v.obs['t_steps']]
        velocity = np.full(adata.shape, np.nan)
        velocity[:, selected_genes_mask] = (U_t * np.tile(np.array(adata_v.var['beta']), (adata_v.shape[0], 1)) - 
                                            S_t * np.tile(np.array(adata_v.var['gamma']), (adata_v.shape[0], 1)))
        adata.layers['velocity'] = velocity

    import scvelo as scv
    scv.tl.velocity_graph(adata, vkey='velocity', xkey='S')
    if figure_folder is not None:
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        with plt.rc_context({"figure.dpi": (150)}): #"figure.figsize": (8, 8)
            scv.pl.velocity_embedding_stream(adata, vkey='velocity', cutoff_perc=0, show=False) 
            plt.savefig(figure_folder+"velocity_stream.png", bbox_inches="tight")  
            plt.close()
        with plt.rc_context({"figure.dpi": (150)}): #"figure.figsize": (8, 8)
            scv.pl.velocity_embedding_grid(adata, vkey='velocity', show=False) 
            plt.savefig(figure_folder+"velocity_grid.png", bbox_inches="tight")  
            plt.close()
    else:
        scv.pl.velocity_embedding_stream(adata, vkey='velocity')

    scv.tl.velocity_confidence(adata)
    adata.obs['velocity_consistency'] = adata.obs['velocity_confidence']
    del adata.obs['velocity_confidence']
    return adata
    
    
def run_paga(adata, iroot_tyre):
    # paga trajectory inference
    sc.tl.paga(adata, groups='clusters')
    #sc.pl.paga(adata, color=['clusters'], save='')
    adata.uns['iroot'] = np.flatnonzero(adata.obs['clusters']  == iroot_tyre)[0]
    sc.tl.dpt(adata)
    #sc.pl.umap(adata, color=['dpt_pseudotime'], legend_loc='on data', save='_dpt_pseudotime.png')
    #scv.pl.scatter(adata, color='dpt_pseudotime', color_map='gnuplot', size=20, save='_dpt_pseudotime.png')
    return adata

def run_palantir(adata, iroot_tyre):
    sc.external.tl.palantir(adata, n_components=5, knn=30)

    iroot = np.flatnonzero(adata.obs['clusters']  == iroot_tyre)[0]
    start_cell = adata.obs_names[iroot]

    pr_res = sc.external.tl.palantir_results(
        adata,
        early_cell=start_cell,
        ms_data='X_palantir_multiscale',
        num_waypoints=500,
    )
    adata.obs['pr_pseudotime'] = pr_res.pseudotime
    adata.obs['pr_entropy'] = pr_res.entropy
    #adata.obs['pr_branch_probs'] = pr_res.branch_probs
    #adata.uns['pr_waypoints'] = pr_res.waypoints
    
    #sc.pl.umap(adata, color=['pr_pseudotime'], legend_loc='on data', save='_pr_pseudotime.png')
    #scv.pl.scatter(adata, color='pr_pseudotime', color_map='gnuplot', size=20, save='pr_pseudotime.png')
    return adata



def get_colors(adata):
    import seaborn as sns
    if not 'clusters_colors' in adata.uns:
        clusters = adata.obs['clusters'].unique()
        n_clusters = len(clusters)
        palette = sns.color_palette("hsv", n_clusters)  
        adata.uns['clusters_colors'] = [sns.color_palette("hsv", n_clusters)[i] for i in range(n_clusters)]
        
    color_map = {}
    cluster_list = list(adata.obs['clusters'].cat.categories)
    for i,c in enumerate(cluster_list):
        color_map[c] = adata.uns['clusters_colors'][i]

    #print(color_map)
    
    colors = []
    for c in adata.obs['clusters']:
        colors.append(color_map[c])
    return colors


def show_imgs(g, s, u, alpha, ds_dt, du_dt, beta, gamma, u_t, s_t, t_steps, colors, plot_alpha=0.7, show_au=True, to_show=1, figure_folder='figures/'):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    norm = 0.1/np.max([np.max(ds_dt), np.max(du_dt)]) * np.min([np.max(s), np.max(u), np.max([np.max(alpha),2])])
    
    print('Drawing', g)

    N_steps, N_cells = len(u_t), len(u)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(221)
    ax1.scatter(t_steps/N_steps, u, label='u', color=colors, alpha=plot_alpha)
    ax1.scatter(np.arange(0,1,1/N_steps), u_t, label='u_t', color='black')
    ax1.set_xlabel('t', fontsize=32)
    ax1.set_ylabel('Unspliced', fontsize=32)
    #ax1.legend()

    ax2 = fig.add_subplot(222)
    ax2.scatter(t_steps/N_steps, s, label='s', color=colors, alpha=plot_alpha)
    ax2.scatter(np.arange(0,1,1/N_steps), s_t, label='s_t', color='black')
    ax2.set_xlabel('t', fontsize=32)
    ax2.set_ylabel('Spliced', fontsize=32)
    #ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.scatter(s, u, label='u-s', color=colors, alpha=plot_alpha)
    ax3.scatter(s_t, u_t, label='u_t-s_t', color='black')

    slopes = np.diff(u_t) / (np.diff(s_t)+1e-6)
    metric_c = np.abs(np.diff(slopes)) - ((np.diff(u_t)+1e-6) + (np.diff(s_t)+1e-6))[:-1]
    window_size = 100
    start = 100
    end = 800
    metric = np.convolve(metric_c, np.ones(window_size), mode='valid')
    t_index = np.argmin(metric[start:end]) 
    t_index = t_index + start
    s_p = s_t[t_index]
    u_p = u_t[t_index]
    delta = 5
    ds = s_t[t_index + delta] - s_t[t_index - delta] 
    du = u_t[t_index + delta] - u_t[t_index - delta] 
    tangent_vector = np.array([ds, du])
    tangent_vector /= np.linalg.norm(tangent_vector) 
    ax3.arrow(s_p, u_p, 0.0001*tangent_vector[0], 0.0001*tangent_vector[1], 
            head_width=0.4, head_length=0.8, fc='black', ec='white', lw=0.5)

    ax3.set_xlabel('Spliced', fontsize=32)
    ax3.set_ylabel('Unspliced', fontsize=32)
    #ax3.legend()
    
    ax4 = fig.add_subplot(224)
    if show_au:
        ax4.scatter(u, alpha, label='a-u', color=colors, alpha=plot_alpha)
        ax4.set_xlabel('Unspliced', fontsize=32)
        ax4.set_ylabel('Alpha', fontsize=32)
    else:
        ax4.scatter(s, alpha, label='a-s', color=colors, alpha=plot_alpha)
        ax4.set_xlabel('Spliced', fontsize=32)
        ax4.set_ylabel('Alpha', fontsize=32)      
        
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(g, fontsize=50)
    if to_show:
        plt.show()
    else:
        plt.savefig(figure_folder+g+'_compare.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    
    fig = plt.figure(figsize=(15, 15))  
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(s, u, alpha, c=colors, alpha=plot_alpha)
    ax1.quiver(s, u, alpha, norm*ds_dt, norm*du_dt, np.zeros_like(du_dt), color='black', alpha=0.5) 
    
    #t = t_steps/N_steps
    #t_predict = np.linspace(0, 1, len(s_t)) 
    #alpha_t = GAM_fit(alpha, t, t_predict)
    #ax1.scatter(s_t, u_t, alpha_t, c='black',s=10)
    
    if np.max(alpha) < 1:
        ax1.set_zlim(0, 1)
    #ax1.view_init(elev=0, azim=270)  # alpha-U:elev=0, azim=0     U-S: elev=90, azim=270   alpha-S: elev=0, azim=270
    ax1.set_xlabel('Spliced', fontsize=32)
    ax1.set_ylabel('Unspliced', fontsize=32)
    ax1.set_zlabel('alpha', fontsize=32)

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(s, u, alpha, c=colors, alpha=plot_alpha)
    ax2.quiver(s, u, alpha, norm*ds_dt, norm*du_dt, np.zeros_like(du_dt), color='black', alpha=0.5) 
    if np.max(alpha) < 1:
        ax2.set_zlim(0, 1)
    ax2.view_init(elev=90, azim=270)  # alpha-U:elev=0, azim=0     U-S: elev=90, azim=270   alpha-S: elev=0, azim=270
    ax2.set_xlabel('Spliced', fontsize=32)
    ax2.set_ylabel('Unspliced', fontsize=32)
    ax2.set_zlabel('alpha', fontsize=32)

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(s, u, alpha, c=colors, alpha=plot_alpha)
    ax3.quiver(s, u, alpha, norm*ds_dt, norm*du_dt, np.zeros_like(du_dt), color='black', alpha=0.5) 
    if np.max(alpha) < 1:
        ax3.set_zlim(0, 1)
    ax3.view_init(elev=0, azim=0)  # alpha-U:elev=0, azim=0     U-S: elev=90, azim=270   alpha-S: elev=0, azim=270
    ax3.set_xlabel('Spliced', fontsize=32)
    ax3.set_ylabel('Unspliced', fontsize=32)
    ax3.set_zlabel('alpha', fontsize=32)
    
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(s, u, alpha, c=colors, alpha=plot_alpha)
    ax4.quiver(s, u, alpha, norm*ds_dt, norm*du_dt, np.zeros_like(du_dt), color='black', alpha=0.5) 
    if np.max(alpha) < 1:
        ax4.set_zlim(0, 1)
    ax4.view_init(elev=0, azim=270)  # alpha-U:elev=0, azim=0     U-S: elev=90, azim=270   alpha-S: elev=0, azim=270
    ax4.set_xlabel('Spliced', fontsize=32)
    ax4.set_ylabel('Unspliced', fontsize=32)
    ax4.set_zlabel('alpha', fontsize=32)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])   
        ax.set_zticks([])            
    plt.suptitle(g, fontsize=50)
    if to_show:
        plt.show()
    else:
        plt.savefig(figure_folder+g+'_phase.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    return 



def show_imgs_simple(g, s, u, alpha, ds_dt, du_dt, beta, gamma, u_t, s_t, t_steps, colors, plot_alpha=0.7, text=None, show_au=True, to_show=1, figure_folder='figures/'):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    norm = 0.1/np.max([np.max(ds_dt), np.max(du_dt)]) * np.max([np.max(s), np.max(u)])
    
    print('Drawing', g)
    N_steps, N_cells = len(u_t), len(u)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(221)
    ax1.scatter(t_steps/N_steps, u, label='u', color=colors, alpha=plot_alpha)
    ax1.scatter(np.arange(0,1,1/N_steps), u_t, label='u_t', color='black')
    ax1.set_xlabel('t', fontsize=32)
    ax1.set_ylabel('Unspliced', fontsize=32)
    #ax1.legend()

    ax2 = fig.add_subplot(222)
    ax2.scatter(t_steps/N_steps, s, label='s', color=colors, alpha=plot_alpha)
    ax2.scatter(np.arange(0,1,1/N_steps), s_t, label='s_t', color='black')
    ax2.set_xlabel('t', fontsize=32)
    ax2.set_ylabel('Spliced', fontsize=32)
    #ax2.legend()

    ax3 = fig.add_subplot(223)
    ax3.scatter(s, u, label='u-s', color=colors, alpha=plot_alpha)
    ax3.scatter(s_t, u_t, label='u_t-s_t', color='black')
    ax3.set_xlabel('Spliced', fontsize=32)
    ax3.set_ylabel('Unspliced', fontsize=32)
    #ax3.legend()
    
    ax4 = fig.add_subplot(224)
    if show_au:
        ax4.scatter(u, alpha, label='a-u', color=colors, alpha=plot_alpha)
        ax4.set_xlabel('Unspliced', fontsize=32)
        ax4.set_ylabel('Alpha', fontsize=32)
    else:
        ax4.scatter(s, alpha, label='a-s', color=colors, alpha=plot_alpha)
        ax4.set_xlabel('Spliced', fontsize=32)
        ax4.set_ylabel('Alpha', fontsize=32)  
            
    if text is not None:
        plt.text(0.5, 0.5, text)
        
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])          
    plt.suptitle(g, fontsize=50)
        
    if to_show:
        plt.show()
    else:
        plt.savefig(figure_folder+g+'_compare.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return 


def show_imgs_alpha_s(g, s, u, alpha, beta, gamma, s_t, t_steps, colors, plot_alpha=0.7, text=None, to_show=1, figure_folder='figures/'):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
    print('Drawing', g)
    N_steps, N_cells = len(s_t), len(u)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.scatter(t_steps/N_steps, s, label='s', color=colors, alpha=plot_alpha)
    ax1.scatter(np.arange(0,1,1/N_steps), s_t, label='s_t', color='black')
    ax1.set_xlabel('t', fontsize=32)
    ax1.set_ylabel('Spliced', fontsize=32)
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(s, alpha, label='alpha-s', color=colors, alpha=plot_alpha)
    ax2.set_xlabel('Spliced', fontsize=32)
    ax2.set_ylabel('Alpha', fontsize=32)
    ax2.legend()

    if text is not None:
        plt.text(0.5, 0.5, text)
        
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])         
    plt.suptitle(g, fontsize=50)
        
    if to_show:
        plt.show()
    else:
        plt.savefig(figure_folder+g+'_alpha_s.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return 

def GAM_fit(data, t, t_predict):
    from pygam import LinearGAM, s
    gam = LinearGAM(s(0)).fit(np.array(t).reshape(-1, 1), data)
    data_predict = gam.predict(t_predict.reshape(-1, 1))
    return data_predict

def show_aus_t(g, s, u, alpha, u_t, s_t, t_steps, max_=None, use_model=True, to_show=1, figure_folder='figures/'):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder) 
    print('Drawing', g)
    s, u, u_t, s_t = s/s.max(), u/u.max(), u_t/u_t.max(), s_t/s_t.max()

    N_steps, N_cells = len(u_t), len(u)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    if use_model:
        t = t_steps/N_steps
        t_predict = np.linspace(0, 1, len(s_t)) 
        alpha_t = GAM_fit(alpha, t, t_predict)
        alpha_t = alpha_t/(alpha_t.max()+1e-6)
        scat1 = ax1.scatter(t_predict, alpha_t, label='alpha', color='g', alpha=1)

    else:
        scat1 = ax1.scatter(t_steps/N_steps, alpha, label='alpha', color='g', alpha=0.5)

    if use_model:  
        scat2 = ax1.scatter(np.arange(0,1,1/N_steps), u_t, label='U_t', color='b')
        scat3 = ax1.scatter(np.arange(0,1,1/N_steps), s_t, label='S_t', color='r')
    else:
        scat4 = ax1.scatter(t_steps/N_steps, u, label='U', color='b', alpha=0.5)
        scat5 = ax1.scatter(t_steps/N_steps, s, label='S', color='r', alpha=0.5)
        
    ax1.set_xlabel('t', fontsize=32)
    ax1.tick_params(axis='y')
    if max_ is not None:
        plt.ylim(0, 1.1/max_) 
    for ax in [ax1]:
        ax.set_xticks([])
        ax.set_yticks([])    
    
    handles = [scat1, scat2, scat3]
    labels = ['Alpha', 'U', 'S']
    plt.legend(handles=handles, labels=labels, fontsize=20)
    
    plt.suptitle(g, fontsize=20)
    if to_show:
        plt.show()
    else:
        plt.savefig(figure_folder+g+'_aus_t.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return

def show_as_t(g, s, alpha, s_t, t_steps, use_model=True, to_show=1, figure_folder='figures/'):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    print('Drawing', g)
    s, s_t = s/s.max(), s_t/s_t.max()

    N_steps, N_cells = len(s_t), len(s)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    if use_model:
        t = t_steps/N_steps
        t_predict = np.linspace(0, 1, len(s_t)) 
        alpha_t = GAM_fit(alpha, t, t_predict)
        alpha_t = alpha_t/alpha_t.max()
        scat1 = ax1.scatter(t_predict, alpha_t, label='alpha', color='g', alpha=1)
    else:
        scat1 = ax1.scatter(t_steps/N_steps, alpha, label='alpha', color='g', alpha=0.5)

    if use_model:  
        scat3 = ax1.scatter(np.arange(0,1,1/N_steps), s_t, label='S_t', color='r')
    else:
        scat3 = ax1.scatter(t_steps/N_steps, s, label='S', color='r', alpha=0.5)
        
    ax1.set_xlabel('t', fontsize=32)
    ax1.tick_params(axis='y')
    for ax in [ax1]:
        ax.set_xticks([])
        ax.set_yticks([])   
    handles = [scat1, scat3]
    labels = ['Alpha', 'S']
    plt.legend(handles=handles, labels=labels, fontsize=20)
    
    plt.suptitle(g, fontsize=20)
    if to_show:
        plt.show()
    else:
        plt.savefig(figure_folder+g+'_as_t.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return



def show_g_branch(g, s_subs, s_t_subs, t_steps_subs, colors_subs, colors_step_subs, show_raw=1, to_show=1, figure_folder='figures/', p_alpha=0.05):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    print('Drawing', g)

    fig, ax1 = plt.subplots(figsize=(6,6))
    for li, s in enumerate(s_subs):
        N_steps, N_cells = len(s_t_subs[li]), len(s_subs[li])
        t_steps = t_steps_subs[li]
        if show_raw:
            ax1.scatter(t_steps/t_steps.max(), s, label='l'+str(li), color=colors_subs[li], alpha=p_alpha)
        t_steps_list = np.arange(t_steps.min(), 1000, 1)
        t_steps_list = (t_steps_list-t_steps_list.min())/(t_steps_list.max()-t_steps_list.min())
        ax1.scatter(t_steps_list, s_t_subs[li], label='l'+str(li), color=colors_step_subs[li], alpha=0.9)  
        r = 0.015 * np.max(np.concatenate(s_subs)) - np.min(np.concatenate(s_subs))
        ax1.scatter(t_steps_list, s_t_subs[li]+r, color='black', alpha=1, s=0.1)     
        ax1.scatter(t_steps_list, s_t_subs[li]-r, color='black', alpha=1, s=0.1)     
        
    ax1.set_ylabel('Spliced', fontsize=32)
    ax1.set_xlabel('t', fontsize=32)
    for ax in [ax1]:
        ax.set_xticks([])
        ax.set_yticks([])   
    plt.suptitle(g, fontsize=20)
    if to_show:
        plt.show()
    else:   
        if show_raw:
            plt.savefig(figure_folder+g+'_branch.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.savefig(figure_folder+g+'_branch_noraw.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return


def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.02):
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def analyze_g(adata, g, max_=None, to_show=False, figure_folder='figures/'):
    g_id = list(adata.var_names).index(g)
    U, S = adata.layers['U'], adata.layers['S']
    U_t, S_t = adata.uns['U_t'], adata.uns['S_t']
    t_steps = adata.obs['t_steps']
    colors = get_colors(adata)
    
    u, s = U[:, g_id], S[:, g_id]
    u_t, s_t = U_t[:, g_id], S_t[:, g_id]
    alpha = adata.layers['alpha'][:,g_id] 
    beta = adata.var['beta'][g_id]
    gamma = adata.var['gamma'][g_id] 
    #du_dt = np.full(u.shape, np.nan)
    du_dt = adata.layers['du_dt'][:,g_id] 
    ds_dt = adata.layers['ds_dt'][:,g_id]
    W_g = adata.varm['W'][g_id]
    W_bias_g = adata.var['W_bias'][g_id] 

    u_pre, s_pre = u_t[t_steps], s_t[t_steps]

    if adata.var['selected_genes'][g_id]:
        show_imgs(g, s, u, alpha, ds_dt, du_dt, beta, gamma, u_t, s_t, t_steps, colors, show_au=True, to_show=to_show, figure_folder=figure_folder)
        show_aus_t(g, s, u, alpha, u_t, s_t, t_steps, max_, use_model=True, to_show=to_show, figure_folder=figure_folder)
    else:
        show_imgs_simple(g, s, u, alpha, ds_dt, du_dt, beta, gamma, u_t, s_t, t_steps, colors, show_au=False, to_show=to_show, figure_folder=figure_folder)
        show_as_t(g, s, alpha, s_t, t_steps, use_model=True, to_show=to_show, figure_folder=figure_folder)
    return

def project_t(adata, adata_subs):
    for li, adata_l in enumerate(adata_subs):
        adata_l.obs['t_combined'] = adata.obs['t']
        t0_idx = adata_l.obs['t_combined'].argmin()
        adata_l = adata_l[adata_l.obs['t_steps']>adata_l[t0_idx].obs['t_steps'][0]]
        adata_subs[li] = adata_l
    return adata_subs

def replace_black_with_closest(colors_step):
    non_black_indices = [i for i, color in enumerate(colors_step) if color != 'black']
    for i, color in enumerate(colors_step):
        if color == 'black':
            closest_index = min(non_black_indices, key=lambda idx: abs(i - idx))
            colors_step[i] = colors_step[closest_index]
    return colors_step
        
def get_colors_step_subs(adata_subs):
    colors_step_subs = []
    for adata in adata_subs:
        colors_step=[]
        for t in range(len(adata.uns['S_t'])):
            clusters_at_t = adata.obs['clusters'][adata.obs['t_steps'] == t]
            cluster_counts = clusters_at_t.value_counts()
            if cluster_counts.sum()>0:
                dominant_cluster = cluster_counts.idxmax()
                dominant_color = adata.uns['clusters_colors'][list(adata.obs['clusters'].cat.categories).index(dominant_cluster)]
            else:
                dominant_color = 'black'
            colors_step.append(dominant_color)   
        colors_step = replace_black_with_closest(colors_step)
        colors_step_subs.append(colors_step)
    return colors_step_subs

def analyze_g_subs(adata_subs, g, colors_step_subs_0, to_show=True, figure_folder='figures/'):
    s_subs, s_t_subs, t_steps_subs, colors_subs, colors_step_subs, s0_subs = [], [], [], [], [], []
    for li, adata in enumerate(adata_subs):
        g_id = list(adata_subs[0].var_names).index(g)
        U, S = adata.layers['U'], adata.layers['S']
        U_t, S_t = adata.uns['U_t'], adata.uns['S_t']
        t_steps = adata.obs['t_steps']
        colors = get_colors(adata)
        
        u, s = U[:, g_id], S[:, g_id]
        u_t, s_t = U_t[:, g_id], S_t[:, g_id]
        alpha = adata.layers['alpha'][:,g_id] 
        beta = adata.var['beta'][g_id]
        gamma = adata.var['gamma'][g_id] 
        #du_dt = np.full(u.shape, np.nan)
        du_dt = adata.layers['du_dt'][:,g_id] 
        ds_dt = adata.layers['ds_dt'][:,g_id]
        W_g = adata.varm['W'][g_id]
        W_bias_g = adata.var['W_bias'][g_id] 
        
        s_t = s_t[t_steps.min():]
        scale = (adata.layers['Ms'][:, g_id]).std()
        s0_subs.append(s_t[0]*scale)
        s_subs.append(s*scale)
        s_t_subs.append(s_t*scale)
        t_steps_subs.append(t_steps)
        colors_subs.append(colors)
        colors_step_subs.append(colors_step_subs_0[li][t_steps.min():])
    scales_subs = np.array(s0_subs).mean()/(np.array(s0_subs)+1e-6)
    max_subs = []    

    for ii in range(len(s_t_subs)):
        s_subs[ii] = s_subs[ii]*scales_subs[ii]
        s_t_subs[ii] = s_t_subs[ii]*scales_subs[ii]
        max_subs.append(s_t_subs[ii].max())
      
    if adata.var['selected_genes'][g_id]:
        show_g_branch(g, s_subs, s_t_subs, t_steps_subs, colors_subs, colors_step_subs, to_show=to_show, figure_folder=figure_folder, p_alpha=0.05)
    return np.array(max_subs)


def GO_KEGG(gene_list, background, figure_path='figures/', save_name=''):
    import gseapy as gp
    #print(len(gene_list), len(background))
    organism = 'mouse'
    gene_sets_KEGG='KEGG_2019_Mouse'
    # GO
    result_go = gp.enrichr(gene_list=gene_list, 
                            organism=organism, 
                            gene_sets='GO_Biological_Process_2021', 
                            background=background,
                            # description='test', 
                            outdir=figure_path+'Enrichr_GO_BP_'+save_name)

    # KEGG
    result_kegg = gp.enrichr(gene_list=gene_list, 
                            organism=organism, 
                            gene_sets=gene_sets_KEGG, 
                            background=background,
                            #   description='test', 
                            outdir=figure_path+'Enrichr_KEGG_'+save_name)
    #result_go.res2d
    #result_kegg.res2d
    return

def analyze_GO_KEGG(genes, background, save_folder='figures/', save_name=''):
    GO_KEGG(genes, background, figure_path=save_folder, save_name=save_name)
    return