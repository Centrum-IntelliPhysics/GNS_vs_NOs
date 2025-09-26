#Import the necessary standard set of libraries
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

#PyTorch NN essentials
import torch
import torch.nn as nn
import torch.nn.functional as F

#Parallelism and DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

#PyTorch Geometric essentials
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn

#Calling specific imports
from utils import *
from models_gns import EncodeProcessDecodeGNS

#Lines for ensuring strict reproducibility in the results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#DDP utils
def setup_ddp(local_rank, rank, world_size):
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

def cleanup_ddp():
    destroy_process_group()


#Utility function for creating the node and edge feature matrices
def create_graph_data(X, Y, edge_index):

    #X: states u(i) with shape (ns*nt, nx, ny, nv)
    #Y: states u(i+1) with shape (ns*nt, nx, ny, nv)
    Ns_Nt, Nx, Ny, Nv = X.shape
    
    #Reshape to (n_samples, n_nodes, nfeat) for compatibility with PyG data object
    X_reshaped = X.reshape(Ns_Nt, Nx*Ny, Nv)
    Y_reshaped = Y.reshape(Ns_Nt, Nx*Ny, Nv)

    #Here the dynamic features are [u, v]. We will now start creating static features
    print("Creating static features such as positional encoding, fourier features, Euclidean distances, etc.")
    print("Building node features")
    
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

    print("Building edge features independent of solution field")
 
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
        u_t_plus_1 = torch.tensor(Y_reshaped[i, :, :], dtype = torch.float)
        
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
        data = Data(x = node_features, edge_index = edge_index, y = u_t_plus_1, edge_attr = edge_features)
        data_list.append(data)
    return data_list


def train(model, dataloader, optimizer, loss_fn, device, dt = 1e-2):
    model.train()
    total_loss = 0
    
    # for batch in dataloader:
    for batch in tqdm(dataloader, desc="Training Progress", dynamic_ncols=True):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        #Predict the time derivative at state i - [du/dt, dv/dt, dh/dt]
        grad_pred = model(batch.x, batch.edge_index, batch.edge_attr)
        
        #Roll one-step forward using an Euler integrator
        u_t = batch.x[:, 0:3]        #Extracting - [u(t), v(t), h(t)]
        
        # #Euler step
        u_t_plus_1 = u_t + grad_pred * dt      #Compute [u(t+1), v(t+1), h(t+1)]
        
        loss = loss_fn(u_t_plus_1, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(dataloader)
    return avg_loss


def test(model, dataloader, loss_fn, device, dt = 1e-2):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            
            #Predict the time derivative at state i - [du/dt, dv/dt, dh/dt]
            grad_pred = model(batch.x, batch.edge_index, batch.edge_attr)
        
            #Roll one-step forward using an Euler integrator
            u_t = batch.x[:, 0:3]        #Extracting - [u(t), v(t), h(t)]
        
            # #Euler step
            u_t_plus_1 = u_t + grad_pred * dt      #Compute [u(t+1), v(t+1), h(t+1)]
        
            loss = loss_fn(u_t_plus_1, batch.y)
            total_loss += loss.item()
    avg_loss = total_loss/len(dataloader)
    return avg_loss


#--------------MAIN function---------------#
def main(log_loss, total_epochs, batch_size):
    print("Program started...", flush=True)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # DDP setup
    setup_ddp(local_rank, rank, world_size)

    # Properly define the CUDA device
    device = torch.device(f'cuda:{local_rank}')

    print("DDP setup done!", flush=True)
    print(f"[Rank {rank}] Running on device {device}, total GPUs: {torch.cuda.device_count()}", flush=True)

    #Set seed values for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    print("Seed set")

    #Load the 2D nonlinear SWE coupled PDE dataset
    base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_nonlinear_SWE_coupled"
    dataset = torch.load(os.path.join(base_path, "2D_nonlinear_SWE.pth"))

    #For h-field
    initial_h = dataset['initial_h'].cpu().detach().numpy()
    outputs_h = dataset['output_h'].cpu().detach().numpy()
    
    #For u-velocity field
    outputs_u = dataset["output_u"].cpu().detach().numpy()

    #For v-velocity field
    outputs_v = dataset["output_v"].cpu().detach().numpy()

    #Check if the IVP can be cast into a Cauchy problem in the context of height perturbation
    #This means inputs = outputs[t=0]
    max_err_h = np.max(np.abs(outputs_h[:, 0, :, :] - initial_h))

    print("Checking if its a Cauchy Problem in the context of height perturbation")
    print(f"Max error in initial_h and outputs_h[t=0] for h-field: {max_err_h}")

    #Concatenate the velocity fields and height perturbations - [u, v, h]
    outputs_uvh = np.concatenate([outputs_u[..., None], outputs_v[..., None], outputs_h[..., None]], axis=-1)

    #Inspecting shapes before subselection of training trajectories
    print(f"Inspecting shape of the outputs_uvh dataset before subselection: {outputs_uvh.shape}")

    #Selecting training trajectories
    num_trajectories = 30

    if os.path.exists(f"subselected_train_idx_GNS_{num_trajectories}.npy"):
        print("Training trajectories exist already. Loading them.")
        selected_trajectories = np.load(f"subselected_train_idx_GNS_{num_trajectories}.npy")
    else:
        print("Performing trajectory selection")
        selected_trajectories = get_trajectory_idx(outputs_uvh, num_trajectories, pca_components = 50)
        #Save trajectories for future use
        np.save(f"subselected_train_idx_GNS_{num_trajectories}.npy", selected_trajectories)
    
    outputs_uvh_subselected = outputs_uvh[selected_trajectories, :, :, :]

    print("Indices selected: ",selected_trajectories)
    print(f"After subselection, outputs_uvh dataset shape: {outputs_uvh_subselected.shape}")

    Ns, Nt, Nx, Ny, Nv = outputs_uvh_subselected.shape

    #Define the dt_val to be used for autoregressive time-stepping
    dt_val = (1-0)/(Nt-1)
    print(f"Computed dt value from simulation: {dt_val}")

    #Design the dataset for creating input-output mapping from state i to i+1
    print("Arranging the dataset for one-step prediction.")
    
    #Creating the input and output training data
    init_timestep = 0
    end_timestep = Nt-1

    input_data_NN = outputs_uvh_subselected[:,init_timestep,:,:]
    output_data_NN = outputs_uvh_subselected[:,init_timestep+1,:,:]

    for i in range(init_timestep+1, end_timestep):
        input_data_NN = np.vstack((input_data_NN, outputs_uvh_subselected[:,i,:,:]))
        output_data_NN = np.vstack((output_data_NN, outputs_uvh_subselected[:,i+1,:,:]))

    print(f"After data arrangement shapes => input: {input_data_NN.shape}, output: {output_data_NN.shape}")

    #Create the edge connectivity list - basically define the 2D grid emulating the graph structure
    edge_index = create_edge_connectivity(nx = Nx, ny = Ny)
    print("Edge connectivity list created. Now moving on to generating the train-test splits")

    #Define train_test split
    X_train, X_test, Y_train, Y_test = train_test_split(input_data_NN, output_data_NN, test_size=0.2, 
                                                        random_state=42, shuffle=True)
    print("After train-test split shapes")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")

    #Create the graph datasets - basically define the node and edge feature vectors
    train_dataset = create_graph_data(X_train, Y_train, edge_index)
    test_dataset = create_graph_data(X_test, Y_test, edge_index)
    print("Train and test datasets created in PyG format. Proper node and edge features defined.")

    # Sampler for DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    print("Train-test DDP samplers created")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                sampler=train_sampler, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                sampler=test_sampler, pin_memory=True)
    print(f"Train-test dataloaders created with batch size = {batch_size}")

    #Declare the dimensions of node and edge feature vectors
    input_node_feat_vector_dim = train_dataset[0].x.shape[-1]
    input_edge_feat_vector_dim = train_dataset[0].edge_attr.shape[-1]
    output_node_feat_vector_dim = 3      #Predict du/dt, dv/dt, dh/dt

    print("Input node feature vector dim: ",input_node_feat_vector_dim)
    print("Input edge feature vector dim: ",input_edge_feat_vector_dim)
    print("Output node feature vector dim: ",output_node_feat_vector_dim)

    #Instantiate the GNS model
    model = EncodeProcessDecodeGNS(input_dim=input_node_feat_vector_dim, 
                                hidden_dim=64, 
                                gnn_dim=64, 
                                edge_dim=input_edge_feat_vector_dim, 
                                output_dim=output_node_feat_vector_dim)

    #Send model to GPU
    model.to(device)
    
    #Wrap model with DDP
    model = DDP(model, device_ids=[device], output_device=device)

    #Use torch.compile
    model = torch.compile(model)
    print("Model instantiated properly with DDP and torch.compile on top of it..", flush=True)

    init_lr = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr = init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    print("Training settings done")

    train_loss_lst = []
    test_loss_lst = []
    min_test_loss = np.inf
    result_dir = "GNS_explicit_Euler"
    file_name_params = f"best_model_params_{num_trajectories}_bs1.pth"

    try:
        os.mkdir(result_dir)
        print("Result directory created successfully")
    except FileExistsError:
        print("Result directory already exists. Moving on to training.")


    print("Starting training....", flush=True)

    if device == torch.device("cuda"):
        torch.distributed.barrier()

    start_time = time.time()
    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)
        train_loss = train(model, train_dataloader, optimizer, loss_fn, device, dt=dt_val)
        test_loss = test(model, test_dataloader, loss_fn, device, dt=dt_val)

        train_loss = torch.tensor([train_loss]).to(device)
        test_loss = torch.tensor([test_loss]).to(device)

        if device == torch.device("cuda"):
            torch.distributed.reduce(train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            train_loss /= world_size

            torch.distributed.reduce(test_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            test_loss /= world_size
        
        lr_scheduler.step()

        if rank == 0:
            if epoch % log_loss==0:
                print(f"Epoch: {epoch}, Training loss: {train_loss.item()}, Testing loss: {test_loss.item()}")
            
            # if test_loss.item() < min_test_loss:
            #     torch.save(model._orig_mod.module.state_dict(), os.path.join(result_dir, file_name_params))
            #     min_test_loss = test_loss.item()
            #     print(f"Saving params at epoch: {epoch}")
        
            train_loss_lst.append(train_loss.item())
            test_loss_lst.append(test_loss.item())
    end_time = time.time()
    print(f"Training completed successfully with {world_size} GPUs in {end_time-start_time} secs", flush=True)

    if rank==0:
        plt.plot(train_loss_lst, label="Training Loss")
        plt.plot(test_loss_lst, label="Testing Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")

        plt.tick_params(which = "major", axis = "both", direction = "in", length = 6)
        plt.tick_params(which = "minor", axis = "both", direction = "in", length = 3.5)
        plt.minorticks_on()

        plt.legend(loc = "best")
        plt.grid()
        # plt.savefig(result_dir + f"/loss_curves_{num_trajectories}_bs1.jpeg", dpi=200)
        print("Loss curves saved!")

    #DDP cleanup
    cleanup_ddp()
#--------------------------------------#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed training for GNS')
    parser.add_argument('--total_epochs', type=int, required=True, help='Total epochs to train the model')
    parser.add_argument('--log_loss', type=int, required=True, help='How often to print the loss')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 64)')
    args = parser.parse_args()

    main(args.log_loss, args.total_epochs, args.batch_size)