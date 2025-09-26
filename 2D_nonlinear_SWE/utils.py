#Import the necessary standard set of libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#PyTorch NN essentials
import torch

#Parallelism
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

#DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

#PyTorch Geometric essentials
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import networkx as NX

#Set seed values for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


def create_edge_connectivity(nx=32, ny=32):
    edge_list = []

    def get_node_index(i, j):
        return i * ny + j
    
    #Standard edge connectivity
    for i in range(nx):
        for j in range(ny):
            
            node_idx = get_node_index(i, j)
            
            #---------Left neighbor----------#
            if j > 0:
                left_idx = get_node_index(i, j-1)
            elif j==0:
                left_idx = get_node_index(i, ny-1)
            edge_list.append([node_idx, left_idx])
            edge_list.append([left_idx, node_idx])
            #--------------------------------#
            
            
                
            #---------Right neighbor------------#
            if j < ny-1:
                right_idx = get_node_index(i, j+1)
            elif j==ny-1:
                right_idx = get_node_index(i, 0)
            edge_list.append([node_idx, right_idx])
            edge_list.append([right_idx, node_idx])
            #----------------------------------#
            
            
            
            #----------Top neighbor-------------#
            if i > 0:
                top_idx = get_node_index(i-1, j)
            elif i==0:
                top_idx = get_node_index(nx-1, j)
            edge_list.append([node_idx, top_idx])
            edge_list.append([top_idx, node_idx])
            #----------------------------------#
            
            
            
            #---------Bottom neighbor-----------#
            if i < nx-1:
                bottom_idx = get_node_index(i+1, j)
            elif i==nx-1:
                bottom_idx = get_node_index(0, j)
            edge_list.append([node_idx, bottom_idx])
            edge_list.append([bottom_idx, node_idx])
            #-----------------------------------#  
                
            #---------Top Left neighbor---------#
            ni = (i - 1) % nx
            nj = (j - 1) % ny
            top_left_idx = get_node_index(ni, nj)
            edge_list.append([node_idx, top_left_idx])
            edge_list.append([top_left_idx, node_idx])
            #-----------------------------------#

            #---------Top Right neighbor--------#
            ni = (i - 1) % nx
            nj = (j + 1) % ny
            top_right_idx = get_node_index(ni, nj)
            edge_list.append([node_idx, top_right_idx])
            edge_list.append([top_right_idx, node_idx])
            #-----------------------------------#

            #---------Bottom Left neighbor------#
            ni = (i + 1) % nx
            nj = (j - 1) % ny
            bottom_left_idx = get_node_index(ni, nj)
            edge_list.append([node_idx, bottom_left_idx])
            edge_list.append([bottom_left_idx, node_idx])
            #-----------------------------------#

            #---------Bottom Right neighbor-----#
            ni = (i + 1) % nx
            nj = (j + 1) % ny
            bottom_right_idx = get_node_index(ni, nj)
            edge_list.append([node_idx, bottom_right_idx])
            edge_list.append([bottom_right_idx, node_idx])
            #-----------------------------------#
                
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def visualize_grid_graph(edge_index, nx=6, ny=6):
    G = NX.Graph()
    pos = {}

    # Create grid positions
    for i in range(nx):
        for j in range(ny):
            node = i * ny + j
            pos[node] = (j, -i)  # (x, y) position for drawing
            G.add_node(node)

    # Add edges
    edge_index = edge_index.cpu().numpy()
    for src, dst in zip(edge_index[0], edge_index[1]):
        G.add_edge(src, dst)

    # Draw the graph
    plt.figure(figsize=(5, 5))
    NX.draw(G, pos, node_size=200, with_labels=True,
            node_color="lightblue", edge_color="gray", font_size=8)
    # plt.title("2D Periodic Grid with 8-Nearest Neighbors")
    plt.axis("equal")
    plt.axis("off")
    plt.show()

def get_trajectory_idx(data, num_trajectories, pca_components):
    
    trajectories_flat = np.array([data[i].flatten() for i in range(data.shape[0])])
    pca = PCA(n_components=pca_components)
    trajectories_pca = pca.fit_transform(trajectories_flat)
    
    kmeans = KMeans(n_clusters=num_trajectories, random_state=42, n_init=10)
    labels = kmeans.fit_predict(trajectories_pca)
    
    selected_indices = []
    for cluster_id in range(num_trajectories):
        members = np.where(labels == cluster_id)[0]  # indices in this cluster
        cluster_center = kmeans.cluster_centers_[cluster_id]

        dists = np.linalg.norm(trajectories_pca[members] - cluster_center, axis=1)
        closest_idx = members[np.argmin(dists)]  # closest to center
        selected_indices.append(closest_idx)
        
    return selected_indices

def reduce_loss(loss, world_size):
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss, dtype=torch.float, device=torch.device("cuda"))
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size
    return loss.item()