#Import the necessary standard set of libraries

#JAX essentials
import jax, jaxlib
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax
from typing import Callable, Sequence

class branch_net(nn.Module):

    layer_sizes: Sequence[int] 
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.glorot_normal()
        
        # #x has shape (ns, nx, ny) - so add channel dimension: (ns, nx, ny, nc)
        if x.ndim != 4:
            x = x[..., jnp.newaxis]
        
        #Convolutional layers
        x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")
        
        x = nn.Conv(features = 64, kernel_size = (2, 2), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape = (2,2), strides = (2,2), padding = "SAME")
        
        x = x.flatten()   #flatten
        
        #MLP layers
        for i, layer in enumerate(self.layer_sizes[:-1]):
            x = nn.Dense(layer, kernel_init = init)(x)
            x = self.activation(x)
        x = nn.Dense(self.layer_sizes[-1], kernel_init = init)(x)
        return x
    
class trunk_net(nn.Module):
    trunk_layer_config: Sequence[int]
    activation: Callable
    
    @nn.compact
    def __call__(self, x):
        
        init = nn.initializers.glorot_normal()
        
        #Branch network forward pass
        for i, layer_size in enumerate(self.trunk_layer_config):
            x = nn.Dense(layer_size, kernel_init = init)(x)
            x = self.activation(x)
        
        return x
    
class DeepONet(nn.Module):

    branch_net_config: Sequence[int]
    trunk_net_config: Sequence[int]
    branch_activation: Callable
    trunk_activation: Callable

    def setup(self):

        self.branch_net = branch_net(self.branch_net_config, self.branch_activation)
        self.trunk_net = trunk_net(self.trunk_net_config, self.trunk_activation)


    def __call__(self, x_branch, x_trunk):
        
        #Vectorize over multiple samples of input functions
        branch_outputs = jax.vmap(self.branch_net, in_axes = 0)(x_branch)
        
        #Vectorize over multiple query points
        trunk_outputs = jax.vmap(self.trunk_net, in_axes = 0)(x_trunk)       
        
        inner_product = jnp.einsum('ik,jk->ij', branch_outputs, trunk_outputs)

        return inner_product
    
class LearnableRK4(nn.Module):
    hidden_dim: int = 32
    
    @nn.compact
    def __call__(self, u_curr):
        
        init = nn.initializers.glorot_normal()
        
        x = u_curr
        
        #u_curr is [bs, nx, ny]. So, add a channel dimension to make it [bs, nx, ny, 1]
        x = x[..., jnp.newaxis]
        
        #Convolutional layers
        x = nn.Conv(features = self.hidden_dim, kernel_size = (3, 3), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")
        
        x = nn.Conv(features = self.hidden_dim, kernel_size = (2, 2), strides = 1, padding = "SAME")(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides = (2, 2), padding = "SAME")
        
        x = x.flatten()
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.activation.tanh(x)
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.activation.tanh(x)
        
        x = nn.Dense(4)(x)
        x = nn.activation.softmax(x)
        
        
        return x