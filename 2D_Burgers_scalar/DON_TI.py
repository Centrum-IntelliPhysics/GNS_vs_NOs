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

from models_jax import DeepONet
from utils_jax import *
from utils import *


seed = 42
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

#Load the 2D Burgers' scalar dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))

#For u-velocity field
outputs_u = (jnp.array(dataset["output_samples"]))[:1000]
print(outputs_u.shape)

#Selecting training trajectories
num_trajectories = 30
selected_trajectories = np.load(f"Selected_indices_GNS_2D_Euler{num_trajectories}.npy")
print(selected_trajectories)

#Selecting a subset of the data
outputs_u_subselected = outputs_u[selected_trajectories, :, :, :]
print(f"After subselection, shape: {outputs_u_subselected.shape}")

#Free up memory
del outputs_u, dataset

Ns, Nt, Nx, Ny = outputs_u_subselected.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

#Creating the input and output training data
init_timestep = 0
end_timestep = Nt-1

input_data_NN = outputs_u_subselected[:, init_timestep, :, :]    
output_data_NN = outputs_u_subselected[:, init_timestep+1, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs_u_subselected[:,i,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs_u_subselected[:,i+1,:,:]))
print(input_data_NN.shape, output_data_NN.shape)

output_data_NN = output_data_NN.reshape((output_data_NN.shape[0], Nx*Ny))

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
trunk_network_layer_sizes = [128]*7 + [latent_vector_size]

model = DeepONet(branch_net_config = branch_network_layer_sizes, 
                 trunk_net_config = trunk_network_layer_sizes,
                branch_activation = nn.activation.tanh,
                trunk_activation = nn.activation.tanh)

model_fn = jax.jit(model.apply)


def rk4_update(model_fn, params, u_curr, trunk_inputs, dt):
    
    #Implement one-step of 4th order Runge-Kutta (RK4) time-stepping method
    u_dot = model_fn(params, u_curr, trunk_inputs)    #(bs, nx*ny)
    
    k1 = u_dot   #(ns*nt, nx*ny)
    k1 = k1.reshape(k1.shape[0], Nx, Ny)   #(ns*nt, nx, ny)
    
    k2 = model_fn(params, u_curr + 0.5 * dt * k1, trunk_inputs)   #(ns*nt, nx*ny)
    k2 = k2.reshape(k2.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k3 = model_fn(params, u_curr + 0.5 * dt * k2, trunk_inputs)     #(ns*nt, nx*ny)
    k3 = k3.reshape(k3.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k4 = model_fn(params, u_curr + dt * k3, trunk_inputs)    #(ns*nt, nx*ny)
    k4 = k4.reshape(k4.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    # Calculate the next state using RK4
    u_pred_next = u_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  #(ns*nt, nx, ny)
    
    #Reshape u_pred_next to match compatibility of u_next
    u_pred_next = u_pred_next.reshape(u_pred_next.shape[0], Nx*Ny)
    
    return u_pred_next


@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs, dt):
    
    #U-velocity field
    u_curr = branch_inputs
    u_next = gt_outputs
    u_pred_next = rk4_update(model_fn, params, u_curr, trunk_inputs, dt)
    loss_u = jnp.mean(jnp.square(u_pred_next - u_next))
    mse_loss = loss_u
    return mse_loss


@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state, dt):
    loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params, branch_inputs, trunk_inputs, gt_outputs, dt)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Initialize model parameters
params = model.init(key, branch_inputs_train[0:1], trunk_inputs_train[0:1])

# # Optimizer setup
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
# num_epochs = int(2e5)
num_epochs = int(2e3)
batch_size = 32
dt_val = (1-0)/(Nt-1)

min_test_loss = jnp.inf

filepath = 'DON_TI'
filename = f"model_params_best_{num_trajectories}.pkl"

print("Starting training...")
for epoch in tqdm(range(num_epochs), desc="Training Progress"):

    #Perform mini-batching
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(epoch), branch_inputs_train.shape[0])
    batch_indices = shuffled_indices[:batch_size]

    branch_inputs_train_batch = branch_inputs_train[batch_indices]
    outputs_train_batch = outputs_train[batch_indices]
    
    # Update the parameters and optimizer state
    params, opt_state, loss = update(
        params=params,
        branch_inputs=branch_inputs_train_batch,
        trunk_inputs=trunk_inputs_train,
        gt_outputs=outputs_train_batch,
        opt_state=opt_state,
        dt=dt_val
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss = loss_fn(params = params,
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test,
                           dt=dt_val)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    # if test_mse_loss < min_test_loss:
    #     best_params = params
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


def inference_ab(model_fn_inf, params_inf, u_curr, u_prev, trunk_inputs_test, dt):
    
    # Step 1: Apply the predictor (Adams-Bashforth) using u_curr and u_prev
    
    # Predict the rate of change at u_curr
    u_dot_curr = model_fn_inf(params_inf, u_curr, trunk_inputs_test)  
    
    # Predict the rate of change at u_prev
    u_dot_prev = model_fn_inf(params_inf, u_prev, trunk_inputs_test)  
    
    #Reshaping u_dot_curr and u_dot_prev to broadcast compatible with u_curr
    u_dot_curr = u_dot_curr.reshape(u_dot_curr.shape[0], Nx, Ny)
    u_dot_prev = u_dot_prev.reshape(u_dot_prev.shape[0], Nx, Ny)
    
    # Adams-Bashforth predictor (using previous two points)
    u_pred = u_curr + dt * (1.5 * u_dot_curr - 0.5 * u_dot_prev)
    
    
    # Step 2: Apply the corrector (Adams-Moulton) using the predicted u_pred
    
    # Predict the rate of change at u_pred
    u_dot_pred = model_fn_inf(params_inf, u_pred, trunk_inputs_test)
    
    #Reshaping u_dot_pred to broadcast compatible with u_curr, u_dot_curr, u_dot_prev
    u_dot_pred = u_dot_pred.reshape(u_dot_pred.shape[0], Nx, Ny)
    
    # Adams-Moulton corrector (refine the prediction using u_pred)
    u_next = u_curr + dt * (5/12 * u_dot_pred + 8/12 * u_dot_curr - 1/12 * u_dot_prev)
    
    return u_next


def inference_rk(model_fn_inf, params_inf, u_curr, trunk_inputs_test, dt):
    
    # Predict the system dynamics (u_dot) at the current state using the model
    u_dot = model_fn_inf(params_inf, u_curr, trunk_inputs_test)  # Model's predicted rate of change

    # Implementing the 4th-order Runge-Kutta (RK4) time-stepping method
    k1 = u_dot   #(ns*nt, nx*ny)
    k1 = k1.reshape(k1.shape[0], Nx, Ny)   #(ns*nt, nx, ny)
    
    k2 = model_fn_inf(params_inf, u_curr + 0.5 * dt * k1, trunk_inputs_test)   #(ns*nt, nx*ny)
    k2 = k2.reshape(k2.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k3 = model_fn_inf(params_inf, u_curr + 0.5 * dt * k2, trunk_inputs_test)     #(ns*nt, nx*ny)
    k3 = k3.reshape(k3.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k4 = model_fn_inf(params_inf, u_curr + dt * k3, trunk_inputs_test)    #(ns*nt, nx*ny)
    k4 = k4.reshape(k4.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    # Calculate the next state using RK4
    u_pred_next = u_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  #(ns*nt, nx, ny)
    
    return u_pred_next


def run_inference(model_fn_inf, params_inf, initial_u, trunk_inputs_test, n_steps, method, dt):
    u_states = np.zeros(shape = (Ns, Nt, Nx, Ny))  # List to store the states over time
    u_states[:,0,:,:] = initial_u
    
    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_prev = initial_u  # Set the previous state to the initial state
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        
        if method == "AB":
            # Perform one inference step using the multistep method
            u_next = inference_ab(model_fn_inf, params_inf, u_curr, u_prev, trunk_inputs_test, dt)

            # Append the predicted state to the list
            u_states[:, i, :, :] = u_next

            # Update previous and current states for the next step
            u_prev = u_curr
            u_curr = u_next
        
        elif method == "RK":
            #Perform one inference step using the rk-4 method
            u_next = inference_rk(model_fn_inf, params_inf, u_curr, trunk_inputs_test, dt)
            
            # Append the predicted state to the list
            u_states[:, i, :, :] = u_next
            
            #Update the current state for the next step
            u_curr = u_next
    
    return u_states


#Do inference
best_params = load_model_params(path = filepath, filename = filename)

#Load the 2D Burgers' scalar dataset
base_path = "/home/dnayak2/data_sgoswam4/Dibya/Datasets/2D_Burgers"
dataset = torch.load(os.path.join(base_path, "Burgers_equation_2D_scalar.pt"))

#For u-velocity field
outputs_u = (jnp.array(dataset["output_samples"]))[:1000]

Ns, Nt, Nx, Ny = outputs_u.shape
print("At inference...")
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")

u_start = outputs_u[:,0,:,:]
start_time = time.time()
u_pred = run_inference(model_fn, best_params, u_start, trunk_inputs_test, 
                       n_steps=Nt, method="AB", dt=dt_val)
end_time = time.time()
print(f"Inference complete for u-velocity field in {end_time-start_time} secs")

#Overall relative L2 error computation
rel_l2_err_u = np.linalg.norm(u_pred - outputs_u)/np.linalg.norm(outputs_u)
print("Overall relative L2 error")
print(f"For u-velocity field, rel. L2 error = {rel_l2_err_u}")

save = False
if save:
    np.save(filepath + "/u_pred.npy", u_pred)