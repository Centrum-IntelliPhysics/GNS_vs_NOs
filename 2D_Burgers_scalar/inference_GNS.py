#Import the necessary standard set of libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import time

#PyTorch NN essentials
import torch
import torch.nn as nn
import torch.nn.functional as F

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


#Calling specific imports
from utils import create_edge_connectivity
from models_gns import EncodeProcessDecodeGNS

print("Program started...")

#Set seed values for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#Setting the device
torch.set_default_device('cuda')
device = torch.device("cuda")
print("Devices and seeds set.")

#Utility function for creating the node and edge feature matrices
def create_graph_data(X, edge_index):

    #X: states u(i) with shape (ns*nt, nx, ny, nv)
    Ns_Nt, Nx, Ny, Nv = X.shape
    
    #Reshape to (n_samples, n_nodes, nfeat) for compatibility with PyG data object
    X_reshaped = X.reshape(Ns_Nt, Nx*Ny, Nv)

    #Here the dynamic features are [u, v]. We will now start creating static features
    
    #Create the xy-coordinates for positional encoding
    x_coords = np.linspace(0, 1, Nx)
    y_coords = np.linspace(0, 1, Ny)
    
    [xx, yy] = np.meshgrid(x_coords, y_coords, indexing="ij")
    xx_ = xx.reshape(-1, 1)
    yy_ = yy.reshape(-1, 1)
    xy_coord = np.hstack((xx_, yy_))    #(Nx*Ny,2)
    xy_coord_tensor = torch.tensor(xy_coord, dtype = torch.float)

    #Create the fourier features for nodes
    sine_encoding = torch.sin(xy_coord_tensor * torch.pi)
    cosine_encoding = torch.cos(xy_coord_tensor * torch.pi)
 
    # Ensure edge_index is torch.long
    edge_index = edge_index.long()

    #Define sender and receiver nodes
    senders = edge_index[0, ...]
    receivers = edge_index[1, ...]

    #Get edge features: vector difference and euclidean distance between receiver and sender nodes
    #Relative positional encodings as edge features
    xy_diff = xy_coord_tensor[receivers] - xy_coord_tensor[senders]
    xy_dist = torch.linalg.norm(xy_diff, dim=1, keepdim=True)
    xy_dir = F.normalize(xy_diff, dim=-1)
    
    #Start creating a set of PyG graph data samples
    data_list = []

    #Iterate through all samples and create a graph data sample for each by assimilating:
    #(Input node features, Static node features, Static edge features, Target node features, edge connectivity)
    for i in range(X_reshaped.shape[0]):
        
        #(num_samples, num_nodes, node_dim)
        u_t = torch.tensor(X_reshaped[i, :, :], dtype = torch.float)
        
        #Create input node features
        node_features = torch.cat([u_t, 
                                   xy_coord_tensor,
                                  sine_encoding,
                                  cosine_encoding],
                                  dim=-1)
        
        #Get edge features: vector difference and euclidean distance between receiver and sender nodes
        
        #First for the actual solution vector field
        u_diff = u_t[receivers] - u_t[senders]
        u_dist = torch.linalg.norm(u_diff, dim=1, keepdim=True)

        # Create edge features
        edge_features = torch.cat([xy_diff, xy_dir, xy_dist, u_diff, u_dist], dim=-1)
        
        #Create a PyTorch Geometric Data object to start the Graph with all its characteristics
        data = Data(x = node_features, edge_index = edge_index, edge_attr = edge_features)
        data_list.append(data)
    return data_list

#Load the 2D Burgers coupled PDE dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))

#Read the u-velocity field
outputs_u = dataset['output_samples'].cpu().detach().numpy()
outputs_u = outputs_u[:1000, :, :, :, None]

#Inspecting shapes after concatenation
Ns, Nt, Nx, Ny, Nv = outputs_u.shape

#Create the edge connectivity list
edge_index = create_edge_connectivity(nx = Nx, ny = Ny)

#Declare the dimensions of node and edge feature vectors
input_node_feat_vector_dim = 7
input_edge_feat_vector_dim = 7
output_node_feat_vector_dim = 1

print("Input node feature vector dim: ",input_node_feat_vector_dim)
print("Input edge feature vector dim: ",input_edge_feat_vector_dim)
print("Output node feature vector dim: ",output_node_feat_vector_dim)


#Instantiate the GNS model
model = EncodeProcessDecodeGNS(input_dim=input_node_feat_vector_dim, 
                               hidden_dim=64, 
                               gnn_dim=64, 
                               edge_dim=input_edge_feat_vector_dim, 
                               output_dim=output_node_feat_vector_dim)

num_trajectories = 30
print(f"Inference initiated from training on {num_trajectories}")
result_dir = "GNS_explicit_Euler"
model_param_filename = f"/best_model_params_{num_trajectories}.pth"

#Load the model corresponding to the best params saved during training
state_dict = torch.load(result_dir + model_param_filename)
model.load_state_dict(state_dict)

#Send model to device
model.to(device)

#Set model to eval mode
model.eval()
print("Best model loaded and set to eval mode")

#Define loss function
loss_fn = nn.MSELoss()

#Define the initial starting state conditions
num_timesteps = Nt
dt = (1-0)/(Nt-1)
predicted_states = np.zeros_like(outputs_u)

predicted_states[:, 0, :, :, :] = outputs_u[:, 0, :, :, :]
u_curr = predicted_states[:, 0, :, :, :]

#Peform autoregressive inference
print("Beginning autoregressive inference...")
start_time = time.time()
with torch.no_grad():
    for t in range(1, num_timesteps):
        u_curr = torch.tensor(u_curr, dtype = torch.float)
        inference_dataset = create_graph_data(u_curr, edge_index)
        
        #Create inference dataloader
        inference_dataloader = DataLoader(inference_dataset, batch_size=u_curr.shape[0], shuffle=False)
        
        for batch in inference_dataloader:
            batch = batch.to(device)
            
            #Predict the time derivative at state i
            grad_pred = model(batch.x, batch.edge_index, batch.edge_attr)
            
            #Roll one-step forward using an Euler integrator
            u_t = batch.x[:, 0:1]
            
            #Euler step
            u_t_plus_1 = u_t + grad_pred * dt
            u_t_plus_1_reshaped = u_t_plus_1.reshape(len(batch), Nx, Ny, Nv)
        predicted_states[:, t, :, :, :] = u_t_plus_1_reshaped.detach().cpu().numpy()
        u_curr = u_t_plus_1_reshaped
end_time = time.time()
print(f"Inference complete for {predicted_states.shape[0]} samples in {end_time-start_time} secs")

print("Post inference...")
print(f"Predictions: {predicted_states.shape}, output: {outputs_u.shape}")

overall_rel_L2_err = np.linalg.norm(predicted_states - outputs_u)/np.linalg.norm(outputs_u)
print(f"Overall relative L2 error: {overall_rel_L2_err}")

#Computing overall relative L2 errors for u and v
rel_L2_err_u = np.linalg.norm(predicted_states[...,0] - outputs_u[...,0])/np.linalg.norm(outputs_u[...,0])

print(f"Relative L2 error in u = {rel_L2_err_u}")

#Save predictions and outputs
save = False

if save:
    np.save(result_dir + "/u_pred.npy", predicted_states[...,0])
    np.save(result_dir + "/u_actual.npy", outputs_u[...,0])