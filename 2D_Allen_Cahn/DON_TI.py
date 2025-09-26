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

from tqdm import tqdm
import time

from models_jax import DeepONet
from utils_jax import *


seed = 42
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

#Load the 2D Allen Cahn dataset
base_path = "/data/sgoswam4/Dibya/Datasets/2D_Allen_Cahn/AllenCahn2D_32.mat"
dataset_mat = scipy.io.loadmat(base_path)
print(dataset_mat.keys())

#Consider 1st 1000 samples
outputs = jnp.array(dataset_mat['u_train'])[:1000, ...]

#Select only a subset of the samples
num_trajectories = 30
selected_idx = np.load(f"Selected_indices_GNS_2D_Euler{num_trajectories}.npy")
print(selected_idx)

outputs = outputs[selected_idx, ...]
print("After subselection")

Ns, Nt, Nx, Ny = outputs.shape
print(f"Ns: {Ns}, Nt: {Nt}, Nx: {Nx}, Ny: {Ny}")
inputs = outputs[:, 0, :, :]

del dataset_mat

#Compute dt value
dt = (1-0)/(Nt-1)
print(f"Computed dt value: {dt}")

#Creating the input and output training data
init_timestep = 0
end_timestep = Nt-1

input_data_NN = outputs[:, init_timestep, :, :]    
output_data_NN = outputs[:, init_timestep+1, :, :]

for i in range(init_timestep+1, end_timestep):
    input_data_NN = jnp.vstack((input_data_NN, outputs[:,i,:,:]))
    output_data_NN = jnp.vstack((output_data_NN, outputs[:,i+1,:,:]))
print(input_data_NN.shape, output_data_NN.shape)

#Reshaping the output_data_NN from (ns*nt//2, nx, ny) to (ns*nt//2, nx*ny)
#Input_data_NN remains as it is, i.e., (ns*nt//2, nx, ny)
output_data_NN = output_data_NN.reshape(output_data_NN.shape[0], 
                                        output_data_NN.shape[1]*output_data_NN.shape[2])
print(input_data_NN.shape, output_data_NN.shape)


input_data_NN_train, input_data_NN_test, output_data_NN_train, output_data_NN_test = \
                        train_test_split(input_data_NN, output_data_NN, test_size = 0.2, random_state = 42)
print(input_data_NN_train.shape, input_data_NN_test.shape, 
      output_data_NN_train.shape, output_data_NN_test.shape)

#Freeing memory by deleting input_data_NN and output_data_NN
del input_data_NN, output_data_NN


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
                branch_activation=nn.activation.tanh,
                trunk_activation=nn.activation.tanh)
model_fn = jax.jit(model.apply)

@jax.jit
def loss_fn(params, branch_inputs, trunk_inputs, gt_outputs, dt):
    
    u_curr = branch_inputs  # Current state input (e.g., u(t))
    u_next = gt_outputs     # Ground truth next state (e.g., u(t+1))

    # Predict the system dynamics (u_dot) at the current state using the model
    u_dot = model_fn(params, u_curr, trunk_inputs)  # Model's predicted rate of change

    # Implementing the 4th-order Runge-Kutta (RK4) time-stepping method
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

    # Compute the Mean Squared Error loss between the predicted and ground truth next states
    mse_loss = jnp.mean(jnp.square(u_pred_next - u_next))
    return mse_loss


@jax.jit
def update(params, branch_inputs, trunk_inputs, gt_outputs, opt_state, dt):
    loss, grads = jax.value_and_grad(loss_fn)(params, branch_inputs, trunk_inputs, gt_outputs, dt)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

#Freeing memory by deleting inputs and outputs
del outputs

# Initialize model parameters
key = jax.random.PRNGKey(seed)   #42
params = model.init(key, branch_inputs_train[0:1, ...], trunk_inputs_train[0:1, ...])

#Initialize optimizer for DeepONet
lr_scheduler = optax.schedules.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.95)
optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

training_loss_history = []
test_loss_history = []
# num_epochs = int(2e5)
num_epochs = int(2e3)
batch_size = 128

min_test_loss = jnp.inf

filepath = 'DeepONet_NODE'
filename = f"model_params_best_{num_trajectories}.pkl"

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
        dt=dt
    )

    training_loss_history.append(loss)
    
    #Do predictions on the test data simultaneously
    test_mse_loss = loss_fn(params = params, 
                            branch_inputs = branch_inputs_test, 
                            trunk_inputs = trunk_inputs_test, 
                            gt_outputs = outputs_test,
                            dt=dt)
    test_loss_history.append(test_mse_loss)
    
    #Save the params of the best model encountered till now
    # if test_mse_loss < min_test_loss:
    #     best_params = params
    #     save_model_params(best_params, path = filepath, filename = filename)
    #     min_test_loss = test_mse_loss
    
    #Print the train and test loss history every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, train_loss: {loss}, test_loss: {test_mse_loss}, best_test_loss: {min_test_loss}")


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
# plt.savefig(filepath + f"/loss_plot_{num_trajectories}.jpeg", dpi = 800)
plt.close()


#Save the loss arrays
save = False

if save:
    np.save(filepath + "/Train_loss.npy",training_loss_history)
    np.save(filepath + "/Test_loss.npy",test_loss_history)


@jax.jit
def inference_ab(u_curr, u_prev, trunk_inputs_test, dt):
    
    # Step 1: Apply the predictor (Adams-Bashforth) using u_curr and u_prev
    u_dot_curr = model_fn(best_params, u_curr, trunk_inputs_test)  # Predict the rate of change at u_curr
    u_dot_prev = model_fn(best_params, u_prev, trunk_inputs_test)  # Predict the rate of change at u_prev
    
    #Reshaping u_dot_curr and u_dot_prev to broadcast compatible with u_curr
    u_dot_curr = u_dot_curr.reshape(u_dot_curr.shape[0], Nx, Ny)
    u_dot_prev = u_dot_prev.reshape(u_dot_prev.shape[0], Nx, Ny)
    
    
    # Adams-Bashforth predictor (using previous two points)
    u_pred = u_curr + dt * (1.5 * u_dot_curr - 0.5 * u_dot_prev)
    
    # Step 2: Apply the corrector (Adams-Moulton) using the predicted u_pred
    u_dot_pred = model_fn(best_params, u_pred, trunk_inputs_test)  # Predict the rate of change at u_pred
    
    #Reshaping u_dot_pred to broadcast compatible with u_curr, u_dot_curr, u_dot_prev
    u_dot_pred = u_dot_pred.reshape(u_dot_pred.shape[0], Nx, Ny)
    
    # Adams-Moulton corrector (refine the prediction using u_pred)
    u_next = u_curr + dt * (5/12 * u_dot_pred + 8/12 * u_dot_curr - 1/12 * u_dot_prev)
    
    return u_next


@jax.jit
def inference_rk(u_curr, trunk_inputs_test, dt):
    
    # Predict the system dynamics (u_dot) at the current state using the model
    u_dot = model_fn(params, u_curr, trunk_inputs_test)  # Model's predicted rate of change

    # Implementing the 4th-order Runge-Kutta (RK4) time-stepping method
    k1 = u_dot   #(ns*nt, nx*ny)
    k1 = k1.reshape(k1.shape[0], Nx, Ny)   #(ns*nt, nx, ny)
    
    k2 = model_fn(params, u_curr + 0.5 * dt * k1, trunk_inputs_test)   #(ns*nt, nx*ny)
    k2 = k2.reshape(k2.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k3 = model_fn(params, u_curr + 0.5 * dt * k2, trunk_inputs_test)     #(ns*nt, nx*ny)
    k3 = k3.reshape(k3.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    k4 = model_fn(params, u_curr + dt * k3, trunk_inputs_test)    #(ns*nt, nx*ny)
    k4 = k4.reshape(k4.shape[0], Nx, Ny)    #(ns*nt, nx, ny)
    
    # Calculate the next state using RK4
    u_pred_next = u_curr + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  #(ns*nt, nx, ny)
    
    return u_pred_next


def run_inference(initial_u, trunk_inputs_test, n_steps, method, dt):
    u_states = np.zeros(shape = (initial_u.shape[0], Nt, Nx, Ny))  # List to store the states over time
    u_states[:,0,:,:] = initial_u
    
    # Initialize the previous state (this could be your u_0 and u_1, etc.)
    u_prev = initial_u  # Set the previous state to the initial state
    u_curr = initial_u  # Set the current state to the initial state
    
    for i in range(1, n_steps):
        
        if method == "AB":
            # Perform one inference step using the multistep method
            u_next = inference_ab(u_curr, u_prev, trunk_inputs_test, dt)

            # Append the predicted state to the list
            u_states[:, i, :, :] = u_next

            # Update previous and current states for the next step
            u_prev = u_curr
            u_curr = u_next
        
        elif method == "RK":
            #Perform one inference step using the rk-4 method
            u_next = inference_rk(u_curr, trunk_inputs_test, dt)
            
            # Append the predicted state to the list
            u_states[:, i, :, :] = u_next
            
            #Update the current state for the next step
            u_curr = u_next
    
    return u_states


# Load the best model parameters
import time
best_params = load_model_params(path=filepath, filename=filename)
dataset_mat = loadmat(base_path)

#Consider 1st 1000 samples
outputs = jnp.array(dataset_mat['u_train'])[:1000, ...]
Ns, Nt, Nx, Ny = outputs.shape
print(outputs.shape)

del dataset_mat

method = "AB"
u_curr = outputs[:, 0, :, :]
start_time = time.time()
u_pred = run_inference(u_curr, trunk_inputs_test, n_steps=Nt, method = method, dt=dt)
end_time = time.time()

print(f"Total time of inference for {u_pred.shape[0]} samples: ",end_time-start_time)
print(u_pred.shape, outputs.shape)

overall_rel_l2_err = jnp.linalg.norm(u_pred - outputs)/jnp.linalg.norm(outputs)
print(f"Overall relative L2 error: {overall_rel_l2_err}")


#Randomly selecting "size" number of samples out of the test dataset
indices = np.random.choice(np.arange(u_pred.shape[0]), size = 3, replace = 'False')

x_test = jnp.linspace(0, 1, Nx)
y_test = jnp.linspace(0, 1, Ny)
t_test = jnp.linspace(0, 1, Nt)

t_query = [25, 50, 75, -1]

for idx in indices:
    
    for t in t_query:
        plt.figure(figsize = (12,3))
        plt.subplot(1, 3, 1)
        contour1 = plt.contourf(x_test, y_test, u_pred[idx, t, :, :].T, levels = 20, cmap = 'viridis')
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("t", fontsize = 14)
        plt.yticks(fontsize = 12)
        plt.xticks(fontsize = 12)
        cbar1 = plt.colorbar()
        cbar1.ax.tick_params(labelsize=12)
        plt.title("Predicted", fontsize = 16)

        plt.subplot(1, 3, 2)
        contour2 = plt.contourf(x_test, y_test, outputs[idx, t, :, :].T, levels = 20, cmap = 'viridis')
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("t", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        cbar2 = plt.colorbar()
        cbar2.ax.tick_params(labelsize=12)
        plt.title("Actual", fontsize=16)

        plt.subplot(1,3,3)
        contour3 = plt.contourf(x_test, y_test, jnp.abs(u_pred[idx,t, :, :].T - 
                                                        outputs[idx,t, :, :].T), cmap = 'Wistia')
        plt.xlabel("x", fontsize = 14)
        plt.ylabel("t", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        cbar3 = plt.colorbar()
        cbar3.ax.tick_params(labelsize=12)
        plt.title("Error", fontsize = 16)
        
        plt.suptitle(f"Sample Idx: {idx}, Timestep: {t}")
        
        plt.tight_layout()

        # plt.savefig(filepath + f"/Contour_plots_{idx}_{method}_{num_trajectories}.jpeg", dpi = 800)
        plt.show()

save = False
if save:
    np.save(filepath + f"/u_pred_{method}.npy", u_pred)