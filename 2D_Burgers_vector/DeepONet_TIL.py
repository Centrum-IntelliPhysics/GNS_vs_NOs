#!/usr/bin/env python
# coding: utf-8

import os, sys, pickle
import jax, jaxlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import numpy as np

import torch
import flax
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from typing import Callable, Sequence
from functools import partial

from tqdm import tqdm
import time

from models_jax import DeepONet, LearnableRK4
from utils_jax import *
from utils import *


seed = 42

np.random.seed(seed)
key = jax.random.PRNGKey(seed)

#Load the 2D Burgers' coupled dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers_coupled"
dataset = torch.load(os.path.join(base_path, "2D_coupled_burgers.pth"))

#For u-velocity field
outputs_u = jnp.array(dataset["output_field_u"])

#For v-velocity field
outputs_v = jnp.array(dataset["output_field_v"])
print(f"Output_u shape: {outputs_u.shape}, output_v shape: {outputs_v.shape}")


#Concatenating outputs
outputs_uv = jnp.stack([outputs_u, outputs_v], axis=-1)
print("After concatenation...")
print(f"Shape of outputs_uv: {outputs_uv.shape}")


#Selecting training trajectories
num_trajectories = 30

if os.path.exists(f"subselected_train_idx_GNS_{num_trajectories}.npy"):
    print("Reading selected trajectories from file...")
    selected_trajectories = np.load(f"subselected_train_idx_GNS_{num_trajectories}.npy")
else:
    print("Selecting trajectories using PCA + KMeans")
    selected_trajectories = get_trajectory_idx(outputs_uv, num_trajectories, pca_components = 50)


#Selecting a subset of the data
outputs_uv_subselected = outputs_uv[selected_trajectories, :, :, :, :]
print(f"After subselection, shape of outputs_uv: {outputs_uv_subselected.shape}")

#Free up memory
del outputs_u, outputs_v, outputs_uv, dataset

Ns, Nt, Nx, Ny, Nv = outputs_uv_subselected.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}, Nv: {Nv}")

#Creating the input and output training data
init_timestep = 0
end_timestep = Nt-1

input_data_NN = outputs_uv_subselected[:, init_timestep, :, :, :]    
output_data_NN = outputs_uv_subselected[:, init_timestep+1, :, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs_uv_subselected[:,i,:,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs_uv_subselected[:,i+1,:,:,:]))
print(input_data_NN.shape, output_data_NN.shape)

#Reshape the output data for shape compatibility
output_data_NN = output_data_NN.reshape((output_data_NN.shape[0], Nx*Ny, Nv))

#Do train-test split
input_data_NN_train, input_data_NN_test, output_data_NN_train, output_data_NN_test = \
                    train_test_split(input_data_NN, output_data_NN, test_size = 0.2, random_state = 42)
print(input_data_NN_train.shape, input_data_NN_test.shape, output_data_NN_train.shape, output_data_NN_test.shape)


#Form branch and trunk inputs train
xspan = jnp.linspace(0, 1, Nx)
yspan = jnp.linspace(0, 1, Ny)

#Create for trunk network
[x,y] = jnp.meshgrid(xspan, yspan, indexing = 'ij')
grid = jnp.transpose(jnp.array([x.flatten(), y.flatten()]))
print(grid.shape)
print(grid)


#Creating the training data for branch and trunk inputs
branch_inputs_train = input_data_NN_train
trunk_inputs_train = grid
outputs_train = output_data_NN_train

print("Shape of branch inputs train: ",branch_inputs_train.shape)
print("Shape of trunk inputs train: ",trunk_inputs_train.shape)
print("Shape of outputs train: ",outputs_train.shape)
print("Shape of grid: ",grid.shape)


#For branch and trunk inputs test

branch_inputs_test = input_data_NN_test
trunk_inputs_test = grid
outputs_test = output_data_NN_test

print("Shape of branch inputs test: ",branch_inputs_test.shape)
print("Shape of trunk inputs test: ",trunk_inputs_test.shape)
print("Shape of outputs test: ",outputs_test.shape)
print("Shape of grid: ",grid.shape)


#DeepONet settings
num_sensor_locations = branch_inputs_train.shape[1]
num_query_locations = 2
latent_vector_size = 100

branch_network_layer_sizes = [256, 128, 128] + [latent_vector_size]
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]   #Earlier it was [128]*3

model_u = DeepONet(branch_net_config = branch_network_layer_sizes, 
                 trunk_net_config = trunk_network_layer_sizes,
                branch_activation = nn.activation.tanh,
                trunk_activation = nn.activation.tanh)
model_v = DeepONet(branch_net_config = branch_network_layer_sizes, 
                 trunk_net_config = trunk_network_layer_sizes,
                branch_activation = nn.activation.tanh,
                trunk_activation = nn.activation.tanh)

model_fn_u = jax.jit(model_u.apply)
model_fn_v = jax.jit(model_v.apply)

model_rk_u = LearnableRK4()
model_rk_fn_u = jax.jit(model_rk_u.apply)

model_rk_v = LearnableRK4()
model_rk_fn_v = jax.jit(model_rk_v.apply)


def dynamic_rk4_step(u_curr, model_fn, params, model_rk_fn, rk_params, trunk_inputs, dt):
    
    #u_curr is basically (ns*nt, nx, ny)
    #To pass through MLP, reshape to (ns*nt, nx*ny)
    # u_curr_reshaped = u_curr.reshape(u_curr.shape[0], nx*ny)
    
    alpha = jax.vmap(model_rk_fn, in_axes = (None, 0))(rk_params, u_curr)         #(Shape: (batch_size, 4)
    
    #Extract the coefficients  - each with shape (batch_size, 1)
    alpha1 = alpha[:,0:1,None]
    alpha2 = alpha[:,1:2,None]
    alpha3 = alpha[:,2:3,None]
    alpha4 = alpha[:,3:,None]
    
    k1 = model_fn(params, u_curr, trunk_inputs)
    k1 = k1.reshape(k1.shape[0], Nx, Ny)
    
    k2 = model_fn(params, u_curr + 0.5 * dt * k1, trunk_inputs)
    k2 = k2.reshape(k2.shape[0], Nx, Ny)
    
    k3 = model_fn(params, u_curr + 0.5 * dt * k2, trunk_inputs)
    k3 = k3.reshape(k3.shape[0], Nx, Ny)
    
    k4 = model_fn(params, u_curr + dt * k3, trunk_inputs)
    k4 = k4.reshape(k4.shape[0], Nx, Ny)

    #Adaptive RK4 update
    u_pred_next = u_curr + dt * (alpha1 * k1 + alpha2 * k2 + alpha3 * k3 + alpha4 * k4)

    return u_pred_next    #(ns*nt, nx, ny)


@jax.jit
def loss_fn(params_u, params_v, rk_params_u, rk_params_v, branch_inputs, trunk_inputs, gt_outputs, dt):
    
    #U-velocity field
    u_curr = branch_inputs[...,0]
    u_next = gt_outputs[...,0]
    u_pred_next = dynamic_rk4_step(u_curr, model_fn_u, params_u, model_rk_fn_u, rk_params_u, trunk_inputs, dt)
    
    #Reshape u_pred_next to match compatibility of u_next
    u_pred_next = u_pred_next.reshape(u_pred_next.shape[0], Nx*Ny)
    
    #V-velocity field
    v_curr = branch_inputs[...,1]
    v_next = gt_outputs[...,1]
    v_pred_next = dynamic_rk4_step(v_curr, model_fn_v, params_v, model_rk_fn_v, rk_params_v, trunk_inputs, dt)

    #Reshape u_pred_next to match compatibility of u_next
    v_pred_next = v_pred_next.reshape(v_pred_next.shape[0], Nx*Ny)
                             
    loss_u = jnp.mean(jnp.square(u_pred_next - u_next))
    loss_v = jnp.mean(jnp.square(v_pred_next - v_next))

    mse_loss = loss_u + loss_v
    
    return mse_loss


@jax.jit
def update(params_u, params_v, rk_params_u, rk_params_v,
           branch_inputs, trunk_inputs, gt_outputs, 
           opt_state_u, opt_state_v, opt_state_rk_u, opt_state_rk_v, dt):
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0,1,2,3))(params_u, params_v, rk_params_u, rk_params_v,
                                              branch_inputs, trunk_inputs, gt_outputs, dt)
    
    grads_u, grads_v, grads_rk_u, grads_rk_v = grads

    updates_u, opt_state_u = optimizer_u.update(grads_u, opt_state_u)
    updates_v, opt_state_v = optimizer_v.update(grads_v, opt_state_v)
    updates_rk_u, opt_state_rk_u = optimizer_rk_u.update(grads_rk_u, opt_state_rk_u)
    updates_rk_v, opt_state_rk_v = optimizer_rk_v.update(grads_rk_v, opt_state_rk_v)

    params_u = optax.apply_updates(params_u, updates_u)
    params_v = optax.apply_updates(params_v, updates_v)
    rk_params_u = optax.apply_updates(rk_params_u, updates_rk_u)
    rk_params_v = optax.apply_updates(rk_params_v, updates_rk_v)
    
    return params_u, params_v, rk_params_u, rk_params_v, \
                opt_state_u, opt_state_v, opt_state_rk_u, opt_state_rk_v, loss


# Initialize model parameters
key, init_u_key, init_v_key, init_rk_u_key, init_rk_v_key = jax.random.split(key, num=5)

#Instantiating for the TI-DON module
params_u = model_u.init(init_u_key, branch_inputs_train[0:1, :, :, 0], trunk_inputs_train[0:1, ...])
params_v = model_v.init(init_v_key, branch_inputs_train[0:1, :, :, 1], trunk_inputs_train[0:1, ...])

#Instantiating for the auxiliary learnable RK4 network
rk_params_u = model_rk_u.init(init_rk_u_key, branch_inputs_train[0:1, :, :, 0])
rk_params_v = model_rk_v.init(init_rk_v_key, branch_inputs_train[0:1, :, :, 1])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
rk_lr_scheduler = optax.schedules.exponential_decay(init_value=2e-3, transition_steps=5000, decay_rate=0.95)

optimizer_u = optax.adam(learning_rate=lr_scheduler)
optimizer_v = optax.adam(learning_rate=lr_scheduler)
optimizer_rk_u = optax.adam(learning_rate=rk_lr_scheduler)
optimizer_rk_v = optax.adam(learning_rate=rk_lr_scheduler)

opt_state_u = optimizer_u.init(params_u)
opt_state_v = optimizer_v.init(params_v)
opt_state_rk_u = optimizer_rk_u.init(rk_params_u)
opt_state_rk_v = optimizer_rk_v.init(rk_params_v)

training_loss_history = []
test_loss_history = []
# num_epochs = int(1.75e5)
num_epochs = int(2e3)
batch_size = 32
dt_val = (1-0)/(Nt-1)

min_test_loss = jnp.inf

filepath = 'DON_TIL'
filename = f"model_params_best_{num_trajectories}.pkl"

print("Starting training...")
for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]
    
    # Update the parameters and optimizer state
    params_u, params_v, rk_params_u, rk_params_v, \
                opt_state_u, opt_state_v, opt_state_rk_u, opt_state_rk_v, loss = update(
        params_u=params_u,
        params_v=params_v,
        rk_params_u=rk_params_u,
        rk_params_v=rk_params_v,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state_u=opt_state_u,
        opt_state_v=opt_state_v,
        opt_state_rk_u=opt_state_rk_u,
        opt_state_rk_v=opt_state_rk_v,
        dt=dt_val
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss = loss_fn(params_u = params_u,
                            params_v = params_v,
                            rk_params_u = rk_params_u,
                            rk_params_v = rk_params_v,
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test,
                           dt = dt_val)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    # if test_mse_loss < min_test_loss:
    #     best_params = {"params_u":params_u, "params_v":params_v, "rk_params_u":rk_params_u, "rk_params_v":rk_params_v}
    #     save_model_params(best_params, path = filepath, filename = filename)
    #     min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, best_test_loss: {min_test_loss}")
print("Training completed..")

plt.figure(dpi = 130)
plt.semilogy(np.arange(epoch+1), training_loss_history, label = "Train loss")
plt.semilogy(np.arange(epoch+1), test_loss_history, label = "Test loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tick_params(which = 'major', axis = 'both', direction = 'in', length = 6)
plt.tick_params(which = 'minor', axis = 'both', direction = 'in', length = 3.5)
plt.minorticks_on()

plt.grid(alpha = 0.3)
plt.legend(loc = 'best')

save = False

if save:
    plt.savefig(filepath + f"/loss_curves_{num_trajectories}.jpeg", dpi=200)
plt.show()


#Save the loss arrays

save = False
if save:
    np.save(filepath + "/train_loss.npy", training_loss_history)
    np.save(filepath + "/test_loss.npy", test_loss_history)

def inference(model_fn, model_rk_fn, best_params, best_rk_params, u_curr, trunk_inputs_test, dt):
    u_next = dynamic_rk4_step(u_curr, model_fn, best_params, model_rk_fn, 
                              best_rk_params, trunk_inputs_test, dt)
    return u_next


def run_inference(model_fn_inf, model_rk_fn_inf, best_params_inf, best_rk_params_inf,
                    initial_u, trunk_inputs_test, n_steps, dt):
    u_states = np.zeros(shape = (initial_u.shape[0], Nt, Nx, Ny))  # List to store the states over time
    u_states[:,0,:,:] = initial_u
    
    # Initialize the current state (this could be your u_0 and u_1, etc.)
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        # Perform one inference step using the multistep method
        u_next = inference(model_fn = model_fn_inf, model_rk_fn = model_rk_fn_inf, 
                           best_params = best_params_inf, 
                           best_rk_params = best_rk_params_inf,
                           u_curr = u_curr,
                           trunk_inputs_test = trunk_inputs_test, dt = dt)
        
        # Append the predicted state to the list
        u_states[:, i, :, :] = u_next
        
        # Update current state for the next step
        u_curr = u_next
    
    return u_states


#Do inference
best_params = load_model_params(path = filepath, filename = filename)
best_params_u = best_params['params_u']
best_params_v = best_params['params_v']
best_rk_params_u = best_params['rk_params_u']
best_rk_params_v = best_params['rk_params_v']

dataset = torch.load(os.path.join(base_path, "2D_coupled_burgers.pth"))

#For u-velocity field
outputs_u = jnp.array(dataset["output_field_u"])

#For v-velocity field
outputs_v = jnp.array(dataset["output_field_v"])

Ns, Nt, Nx, Ny = outputs_u.shape
print("At inference...")
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

u_start = outputs_u[:,0,:,:]
v_start = outputs_v[:,0,:,:]

start_time = time.time()
u_pred = run_inference(model_fn_inf = model_fn_u, 
                       model_rk_fn_inf = model_rk_fn_u, 
                       best_params_inf = best_params_u, 
                       best_rk_params_inf = best_rk_params_u, 
                       initial_u = u_start, 
                       trunk_inputs_test = trunk_inputs_test, 
                       n_steps=Nt, dt=dt_val)
print("Inference complete for u-velocity field")

v_pred = run_inference(model_fn_inf = model_fn_v, 
                       model_rk_fn_inf = model_rk_fn_v, 
                       best_params_inf = best_params_v, 
                       best_rk_params_inf = best_rk_params_v, 
                       initial_u = v_start, 
                       trunk_inputs_test = trunk_inputs_test, 
                       n_steps=Nt, dt=dt_val)
print("Inference complete for v-velocity field")
end_time = time.time()

print(f"Overall inference time for u and v velocity fields: {end_time-start_time} secs")


#Overall relative L2 error computation
rel_l2_err_u = np.linalg.norm(u_pred - outputs_u)/np.linalg.norm(outputs_u)
rel_l2_err_v = np.linalg.norm(v_pred - outputs_v)/np.linalg.norm(outputs_v)

print("Overall relative L2 error")
print(f"For u-velocity field, rel. L2 error = {rel_l2_err_u}")
print(f"For v-velocity field, rel. L2 error = {rel_l2_err_v}")

save = False
if save:
    np.save(filepath + "/u_pred.npy", u_pred)
    np.save(filepath + "/v_pred.npy", v_pred)