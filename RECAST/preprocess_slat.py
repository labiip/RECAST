# uncomment this if you want to use interactive plot (only works in Jupyter not works in VScode)
# %matplotlib widget

import scanpy as sc 
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import vstack

import scSLAT
from scSLAT.model import Cal_Spatial_Net, load_anndatas, run_SLAT, spatial_match
from scSLAT.viz import match_3D_multi, hist, Sankey
from scSLAT.metrics import region_statistics

#adata1 = adata_list[33]#adata_list[1]
#adata2 = adata_list[0]

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def do_slat_pair(adata1, adata2, feature='pca', device=None):

    Cal_Spatial_Net(adata1, k_cutoff=20, model='KNN')
    Cal_Spatial_Net(adata2, k_cutoff=20, model='KNN')

    edges, features = load_anndatas([adata1, adata2], feature=feature, check_order=False)

    embd0, embd1, time = run_SLAT(features, edges, device=device)

    best, index, distance = spatial_match(features, adatas=[adata1, adata2], reorder=False)
    
    return [adata1, adata2], [best]




def get_the_feature(adata_list, matching_list):
    
    overlap = set(adata_list[0].var_names.tolist())
    
    for adata in adata_list[1:]:
        
        overlap &= set(adata.var_names.tolist())
        
    overlap = sorted(list(overlap))
    
    print(len(overlap))

    X1 = adata_list[0][:,overlap].X.toarray()

    spatial = adata_list[0][:,overlap].obsm['spatial']
        
    X = [X1]
    
    spatials = [spatial]
    
    y = [np.array([0] * adata_list[0].X.shape[0])]
    
    
    
    for i, adata in enumerate(adata_list[1:]):
        
        if len(matching_list[i].shape)>1:
        
            spatials.append(spatial[matching_list[i][1]])
            
            X.append(adata[matching_list[i][0],overlap].X.toarray())
            
            y.append(np.array([i+1] * adata[matching_list[i][0],overlap].X.shape[0]))
        
        else:

            
            spatials.append(spatial[matching_list[i]])
            
            X.append(adata[:,overlap].X.toarray())
            
            y.append(np.array([i+1] * adata[:,overlap].X.shape[0]))

    
    return np.concatenate(X, axis=0), np.concatenate(spatials, axis=0), np.concatenate(y, axis=0).astype(int), overlap 


def get_the_multi_feature(adata_list, matching_list):
    
    overlap = set(adata_list[0].var_names.tolist())
    
    for adata in adata_list[1:]:
        
        overlap &= set(adata.var_names.tolist())
        
    overlap = sorted(list(overlap))

    X1 = adata_list[0][:,overlap].X.toarray()

    y1 = adata_list[0][:,overlap].obsm['spatial']
    
    X1_addition = np.ones((y1.shape[0], 1)) * 0
    
    X = [np.concatenate((X1, X1_addition), axis=1)]
    
    y = [y1]
    
    y_1 = [np.array([0] * adata_list[0].X.shape[0])]
    
    
    for i, adata in enumerate(adata_list[1:]):
        
        if len(matching_list[i].shape)>1:
        
            tmp_y1 = y1[matching_list[i][1]]
        
            tmp_y1[...,-1] = np.ones((tmp_y1.shape[0])) * (i+1)
        
            y.append(tmp_y1)
            
            X.append(adata[matching_list[i][0],overlap].X.toarray())
            
            y_1.append(np.array([i+1] * adata[matching_list[i][0],overlap].X.shape[0]))
        
        else:
            
            tmp_y1 = y1[matching_list[i]]
            
            X1_addition = np.ones((tmp_y1.shape[0], 1)) * (i+1)
            
            y.append(tmp_y1)
            
            X.append(np.concatenate((adata[:,overlap].X.toarray(), X1_addition), axis=1))
            
            y_1.append(np.array([i+1] * adata[:,overlap].X.shape[0]))
  
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0), np.concatenate(y_1, axis=0).astype(int), adata_list[0].var_names.tolist()


def get_the_sim_feature(adata_list):
    
    
    X = [adata_list[0].X]
    
    y1 = adata_list[0].obsm['spatial']

    y = [y1]

    y_1 = [np.array([0] * adata_list[0].X.shape[0])]

    for i, adata in enumerate(adata_list[1:]):
        
        y.append(adata.obsm['spatial'])
        
        X.append(adata.X)
        
        y_1.append(np.array([i+1] * adata.X.shape[0]))

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0), np.concatenate(y_1, axis=0).astype(int), adata_list[0].var_names.tolist() 
def get_the_feature_new(adata_list, matching_list):
    """
    对齐基因 + 重排行 + 返回 numpy 数组 + 两个对齐后的 AnnData
    仅演示 2 张切片的情况；>2 张同理扩写循环即可。
    """
    # ---------- 1. 基因交集 ----------
    overlap = set(adata_list[0].var_names)
    for adata in adata_list[1:]:
        overlap &= set(adata.var_names)
    overlap = sorted(overlap)
    print('common genes:', len(overlap))

    # ---------- 2. 取出第一张（参考） ----------
    adata0 = adata_list[0][:, overlap].copy()
    adata0.obs['batch'] = '0'
    adata0.obs['global_row'] = np.arange(adata0.n_obs)  # 记录原始行号

    # ---------- 3. 第二张按匹配重排 ----------
    best = matching_list[0]          # 长度 = adata2.n_obs
    adata2 = adata_list[1][:, overlap].copy()
    adata2 = adata2[best, :]         # 关键：行顺序与 adata0 一一对应
    adata2.obs['batch'] = '1'
    adata2.obs['global_row'] = best  # 记录它在原 slice1 里的行号

    # ---------- 4. 构造“对齐后”的 AnnData ----------
    # 4.1 拼表达矩阵
    X_joint = vstack([adata0.X, adata2.X])
    # 4.2 拼 obs
    obs_joint = pd.concat([adata0.obs, adata2.obs], axis=0)
    # 4.3 新 AnnData
    adata_aligned = ad.AnnData(X=X_joint, var=adata0.var, obs=obs_joint)
    # 4.4 拼空间坐标
    adata_aligned.obsm['spatial'] = np.vstack([
        adata0.obsm['spatial'],
        adata2.obsm['spatial']
    ])
    # 4.5 把 SLAT 嵌入也塞进去（如果前面跑过）
    if 'X_SLAT' in adata0.obsm:
        adata_aligned.obsm['X_SLAT'] = np.vstack([
            adata0.obsm['X_SLAT'],
            adata2.obsm['X_SLAT']
        ])

    # ---------- 5. 也单独返回两张“已对齐”的 AnnData ----------
    adata0_aligned = adata0.copy()
    adata1_aligned = adata2.copy()
    # adata0.obs['global_row'] = np.arange(adata0.n_obs).astype(int)
    # adata2.obs['global_row'] = best.astype(int)

    # # ✅ 保留 layers（如果存在）
    # for key in adata_list[0].layers:
    #     adata0_aligned.layers[key] = adata_list[0].layers[key][adata0.obs['global_row'].values.astype(int), :][:, overlap]

    # for key in adata_list[1].layers:
    #     adata1_aligned.layers[key] = adata_list[1].layers[key][adata2.obs['global_row'].values.astype(int), :][:, overlap]

    # ---------- 6. 为 River 准备 numpy 数组 ----------
    gene_expression = adata_aligned.X.toarray()          # (n0+n1, n_genes)
    spatial         = adata_aligned.obsm['spatial']      # (n0+n1, 2)
    y               = adata_aligned.obs['batch'].values.astype(int)  # 0/1 标签

    return gene_expression, spatial, y, overlap, adata0_aligned, adata1_aligned
