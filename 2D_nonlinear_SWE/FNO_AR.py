#!/usr/bin/env python
# coding: utf-8

#Importing all the necessary libraries
import os
import sys
import pickle
import time
from tqdm import tqdm

import torch
import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

import flax
from flax import linen as nn

import optax
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, List
import scipy

from models_fno import FNO2d
from utils_jax import *
from utils import *

seed = 42
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

#Load the 2D nonlinear SWE coupled PDE dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_nonlinear_SWE_coupled"
dataset = torch.load(os.path.join(base_path, "2D_nonlinear_SWE.pth"))

#For h-field
initial_h = jnp.array(dataset['initial_h'])
outputs_h = jnp.array(dataset['output_h'])

#For u-velocity field
outputs_u = jnp.array(dataset["output_u"])

#For v-velocity field
outputs_v = jnp.array(dataset["output_v"])

#Concatenating outputs
outputs_uvh = jnp.stack([outputs_u, outputs_v, outputs_h], axis=-1)
print(outputs_uvh.shape)


#Selecting training trajectories
num_trajectories = 30
if os.path.exists(f"subselected_train_idx_GNS_{num_trajectories}.npy"):
    print("Reading selected trajectories from file...")
    selected_trajectories = np.load(f"subselected_train_idx_GNS_{num_trajectories}.npy")
else:
    print("Selecting trajectories using PCA + KMeans")
    selected_trajectories = get_trajectory_idx(outputs_uvh, num_trajectories, pca_components = 50)
print(selected_trajectories)

#Selecting a subset of the data
outputs_uvh_subselected = outputs_uvh[selected_trajectories, :, :, :, :]
print(f"After subselection, shape: {outputs_uvh_subselected.shape}")

#Free up memory
del outputs_uvh, dataset

Ns, Nt, Nx, Ny, Nv = outputs_uvh_subselected.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}, Nv: {Nv}")

#Creating the input and output training data
init_timestep = 0
end_timestep = Nt-1

input_data_NN = outputs_uvh_subselected[:, init_timestep, :, :, :]    
output_data_NN = outputs_uvh_subselected[:, init_timestep+1, :, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs_uvh_subselected[:,i,:,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs_uvh_subselected[:,i+1,:,:,:]))
print(input_data_NN.shape, output_data_NN.shape)


#Add channel dimension to the input-output data pairs

#Create the meshes along X and Y axes
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

[X, Y] = jnp.meshgrid(xspan, yspan, indexing = "ij") 

#X and Y have shapes: (Nx, Ny). Next we tile it across all samples: (Ns*(Nt-1))
X_tiled = jnp.tile(X[None, :, :], (input_data_NN.shape[0], 1, 1))
Y_tiled = jnp.tile(Y[None, :, :], (input_data_NN.shape[0], 1, 1))

#Concatenate mesh to the input_data_NN_tensor
inputs_to_FNO = jnp.concatenate([input_data_NN, X_tiled[:, :, :, None], Y_tiled[:, :, :, None]], 
                                axis=-1)
output_FNO = output_data_NN

print("After merging meshes")
print(f"Inputs: {inputs_to_FNO.shape}, Outputs: {output_FNO.shape}")


#Free up some memory
del input_data_NN, output_data_NN, X_tiled, Y_tiled


#Split into training and testing datasets
Ntrain = int(0.8*inputs_to_FNO.shape[0])
perm = jax.random.permutation(jax.random.PRNGKey(0), inputs_to_FNO.shape[0])

train_idx = perm[:Ntrain]
test_idx = perm[Ntrain:]

train_x = jnp.take(inputs_to_FNO, train_idx, axis=0)
test_x = jnp.take(inputs_to_FNO, test_idx, axis=0)

train_y = jnp.take(output_FNO, train_idx, axis=0)
test_y = jnp.take(output_FNO, test_idx, axis=0)

print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")


#Stop gradients for test_x and test_y
test_x = jax.lax.stop_gradient(test_x)
test_y = jax.lax.stop_gradient(test_y)


#Create the FNO-2D model object
modes1 = 16
modes2 = 16

#Create the FNO-2D model object
fno = FNO2d(in_channels = train_x.shape[-1],
            out_channels = train_y.shape[-1],
            modes1 = modes1,
            modes2 = modes2,
            width = 64,
            n_blocks = 6,
            activation = nn.activation.gelu,  
)


model_fn = jax.jit(fno.apply)

#Instantiate the model params
params = fno.init(key, train_x[0:1])

@jax.jit
def loss_fn(params, x, y):
    y_pred = model_fn(params, x)
    loss = jnp.mean((y_pred - y) ** 2)
    return loss

@jax.jit
def make_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    val_loss = loss_fn(params, test_x, test_y)
    return params, opt_state, loss, val_loss


lr = 1e-3
lr_scheduler = optax.schedules.exponential_decay(init_value=lr, transition_steps=2000, decay_rate=0.96)
optimizer = optax.adam(lr_scheduler)
opt_state = optimizer.init(params)

loss_history = []
val_loss_history = []
batch_size = 16
shuffle_key = jax.random.PRNGKey(80)
epochs = int(2e3)
min_val_loss = jnp.inf

result_dir = f"./FNO_AR"

print("Starting training...")
for epoch in range(epochs):
    shuffle_key, subkey = jax.random.split(shuffle_key)
    total_loss = 0
    total_val_loss = 0
    nbatches = 0
    for (batch_x, batch_y) in tqdm(dataloader(subkey, train_x, train_y, batch_size)):
        
        batch_x = jax.lax.stop_gradient(batch_x)
        batch_y = jax.lax.stop_gradient(batch_y)

        params, opt_state, loss, val_loss = make_step(params, opt_state, batch_x, batch_y)
        
        total_loss += loss
        total_val_loss += val_loss
        nbatches += 1
    
    loss = total_loss/nbatches
    val_loss = total_val_loss/nbatches
    
    #Save the best model
    # if val_loss < min_val_loss:
    #     best_params = params
    #     min_val_loss = val_loss
    #     save_model_params(best_params, result_dir, filename = f"best_model_params_{num_trajectories}.pkl")
    
    loss_history.append(total_loss/nbatches)
    val_loss_history.append(total_val_loss/nbatches)
    
    if epoch % 50==0:
        print(f"Epoch: {epoch}, Train loss: {total_loss/nbatches}, Val loss: {total_val_loss/nbatches}") 
print("Training completed successfully.")

plt.figure(dpi = 130)
plt.semilogy(np.arange(epoch+1), loss_history, label = "Train loss")
plt.semilogy(np.arange(epoch+1), val_loss_history, label = "Test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tick_params(which = 'major', axis = 'both', direction = 'in', length = 6)
plt.tick_params(which = 'minor', axis = 'both', direction = 'in', length = 3.5)
plt.minorticks_on()

plt.grid(alpha = 0.3)
plt.legend(loc = 'best')

save = False
if save:
    plt.savefig(result_dir + f"/loss_plot_{num_trajectories}.jpeg", dpi = 300)
plt.show()


#Save the loss arrays
save = False
if save:
    np.save(result_dir + "/Train_loss.npy",loss_history)
    np.save(result_dir + "/Test_loss.npy",val_loss_history)


# def create_input(x):
    
#     Ns_, Nx, Ny, Nv = x.shape
    
#     xspan = jnp.linspace(0, 1, Nx)
#     yspan = jnp.linspace(0, 1, Ny)
    
#     [X, Y] = jnp.meshgrid(xspan, yspan, indexing = "ij")
#     X_tiled = jnp.tile(X[None, :, :], (Ns_, 1, 1))
#     Y_tiled = jnp.tile(Y[None, :, :], (Ns_, 1, 1))
    
#     x_with_mesh = jnp.concatenate([x, X_tiled[:,:,:,None], Y_tiled[:,:,:,None]], axis=-1)
#     return x_with_mesh


# def run_inference(initial_uvh, n_steps):
#     uvh_states = np.zeros_like(outputs_uvh)  # List to store the states over time
#     uvh_states[:,0,:,:,:] = initial_uvh
    
#     # Initialize the previous state (this could be your u_0 and u_1, etc.)
#     uvh_curr = initial_uvh  # Set the current state to the initial state
    
#     for i in range(1, n_steps):
#         uvh_curr_in_FNO = create_input(uvh_curr)    #(Ns, Nx, Ny, in_channels)
#         uvh_next_out_FNO = model_fn(best_params, uvh_curr_in_FNO)   #(Ns, Nx, Ny, out_channels)
#         uvh_next = uvh_next_out_FNO[:, :, :, :]
        
#         # Append the predicted state to the list
#         uvh_states[:, i, :, :, :] = uvh_next
        
#         # Update previous and current states for the next step
#         uvh_curr = uvh_next
    
#     return uvh_states


# # Load the best model parameters
# best_params = load_model_params(result_dir, filename = f"best_model_params_{num_trajectories}.pkl")
# print("Using best_params with filename")
# print(f"best_model_params_{num_trajectories}.pkl")

# #Load the 2D nonlinear SWE coupled PDE dataset
# base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_nonlinear_SWE_coupled"
# dataset = torch.load(os.path.join(base_path, "2D_nonlinear_SWE.pth"))

# #For h-field
# initial_h = jnp.array(dataset['initial_h'])
# outputs_h = jnp.array(dataset['output_h'])

# #For u-velocity field
# outputs_u = jnp.array(dataset["output_u"])

# #For v-velocity field
# outputs_v = jnp.array(dataset["output_v"])

# #Concatenating outputs
# outputs_uvh = jnp.stack([outputs_u, outputs_v, outputs_h], axis=-1)
# print(outputs_uvh.shape)

# Ns, Nt, Nx, Ny, Nv = outputs_uvh.shape
# print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}, Nv: {Nv}")

# print("Running inference...")
# uvh_start = outputs_uvh[:, 0, :, :, :]
# start_time = time.time()
# uvh_pred = run_inference(uvh_start, n_steps=Nt)
# end_time = time.time()
# print(f"Inference complete for {uvh_pred.shape[0]} samples: {end_time-start_time} secs")
# print("Post inference, shapes...")
# print(f"Predicted: {uvh_pred.shape}, Outputs: {outputs_uvh.shape}")

# #Compute relative L2 error between predicted and outputs
# rel_l2_err = np.linalg.norm(uvh_pred - outputs_uvh)/np.linalg.norm(outputs_uvh)
# print(f"Overall relative L2 error: {rel_l2_err}")

# #Compute relative L2 errors separately for u and v velocity fields
# rel_l2_err_u = np.linalg.norm(uvh_pred[...,0] - outputs_uvh[...,0])/np.linalg.norm(outputs_uvh[...,0])
# rel_l2_err_v = np.linalg.norm(uvh_pred[...,1] - outputs_uvh[...,1])/np.linalg.norm(outputs_uvh[...,1])
# rel_l2_err_h = np.linalg.norm(uvh_pred[...,2] - outputs_uvh[...,2])/np.linalg.norm(outputs_uvh[...,2])

# print(f"Relative L2 error in u: {rel_l2_err_u}")
# print(f"Relative L2 error in v: {rel_l2_err_v}")
# print(f"Relative L2 error in h: {rel_l2_err_h}")

# #Save all the relevant arrays
# save = True
# if save:
#     np.save(result_dir + f"/uvh_pred_{num_trajectories}.npy", uvh_pred)

# print("Program executed successfully!")