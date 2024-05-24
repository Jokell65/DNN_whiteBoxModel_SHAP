import os
import pickle

import jax.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as initializers

class Cifar10CNN(nn.Module):
    init_func : any
    activation_func : any

    @nn.compact
    def __call__(self, x):
        const_initalizer = initializers.constant(0)
        
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = self.activation_func(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = self.activation_func(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256, kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = self.activation_func(x)
        x = nn.Dense(features=10, kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = nn.log_softmax(x)
        return x


class WineQualityNetwork(nn.Module):
    init_func : any
    activation_func : any

    @nn.compact
    def __call__(self, x):
        const_initalizer = initializers.constant(0)
        x = nn.Dense(features=128, kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = self.activation_func(x)
        x = nn.Dense(features=32, kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = self.activation_func(x)
        x = nn.Dense(features=1, kernel_init=self.init_func, bias_init=const_initalizer)(x)
        x = nn.sigmoid(x)
        return x
    
def create_model(model_class, rng, input_shape=(1, 32, 32, 3),
                 init_func=initializers.xavier_uniform(), activation_func=nn.tanh):
    """Create a model and intilize it's weights.

    :param model_class: model architecture class
    :type model_class:
    :param rng: random number generator state
    :type rng: jax.random.PRNGKey
    :param input_shape: input shape to the model, defaults to (1, 32, 32, 3)
    :type input_shape: tuple, optional
    :param init_method: choose between 'xavier' and 'kaiming' weight-init,
        defaults to 'xavier'
    :type init_method: string, optional TODO
    :param activation_func: activation layer to be used,
        defaults to flax.linen.nn.tanh
    :type activation_func: flax.linen.nn
    :returns: model and weights
    :rtype: tuple
    """
    # Create Model Object
    model = model_class(init_func, activation_func)
    # Initilize weights
    weights = model.init(rng, jnp.ones(input_shape, jnp.float32))
    return model, weights 


def save_weights(weights, path: str, name: str):
    """Save weights.

    :param weights: weights to be saved
    :type weights: 
    :param path: path to the folder the weights should be saved in
    :type folder_path: str
    :param name: name of the file to be created (without .pkl ending)
    :type name: str
    """
    filepath = os.path.join(path, name) + '.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)


def load_weights(path: str):
    """Load weights from path.
    
    :param path: path that werights are saved at
    :type path: str
    :return: models weights
    :rtype:
    """
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
