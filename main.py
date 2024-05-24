import os
import argparse

import numpy as np

import jax
from flax.linen import relu
from flax.linen.initializers import kaiming_uniform

from model_checkpoints.data_loaders import load_cifar10, load_wine_quality
from models import Cifar10CNN, WineQualityNetwork, create_model, load_weights
from utils import validate_cifar, validate_wine


def main(args):
    #random number generator key (seed)
    rng = jax.random.PRNGKey(0)
    
    #check if the datapath is a proper path
    if not os.path.exists(args.datapath):
        raise ValueError(f'The datapath {args.datapath} is not viable, please pick another.')

    #create data folder
    datapath = os.path.join(args.datapath, 'data')
    os.makedirs(datapath, exist_ok=True)

    #download datasets and store the splits
    cif_train, cif_val, cif_test = load_cifar10(datapath)
    wine_train, wine_val, wine_test = load_wine_quality(datapath)

    #load models
    cifar_model, _ = create_model(Cifar10CNN, rng, init_func=kaiming_uniform(), activation_func=relu)
    wine_model, _ = create_model(WineQualityNetwork, rng, input_shape=(1, 11), init_func=kaiming_uniform(), activation_func=relu)

    #create weight dictionary
    weights = {
        'cifar' : {
            0 : load_weights(f'./model_checkpoints/cifar10/kaiming_uniform/relu/initial_weights.pkl'),
            1 : load_weights(f'./model_checkpoints/cifar10/kaiming_uniform/relu/best_weights.pkl'),
            2 : load_weights(f'./model_checkpoints/cifar10/kaiming_uniform/relu/overtrained_model.pkl')
                  },
        'wine' : {
            0 : load_weights(f'./model_checkpoints/wine_quality/kaiming_uniform/relu/initial_weights.pkl'),
            1 : load_weights(f'./model_checkpoints/wine_quality/kaiming_uniform/relu/best_weights.pkl'),
            2 : load_weights(f'./model_checkpoints/wine_quality/kaiming_uniform/relu/overtrained_model.pkl')
                 }
    }
    
    #validates if models work properly
    if args.validate:
        try:
            val_losses = np.zeros((2,3), dtype=np.float32)
            for i in range(3):
                val_losses[0][i] = validate_cifar(cif_val, cifar_model, weights['cifar'][i])
                val_losses[1][i] = validate_wine(wine_val, wine_model, weights['wine'][i])
            
            print('CIFAR10 Validation Loss (Start):', val_losses[0][0], f'Expected: {2.633335}.')
            print('CIFAR10 Validation Loss (Best):', val_losses[0][1], f'Expected: {0.8622213}.')
            print('CIFAR10 Validation Loss (Overfitted):', val_losses[0][2], f'Expected: {2.1891441}.')
        
            print('Wine Quality Validation Loss (Start):', val_losses[1][0], f'Expected: {0.03249127}.')
            print('Wine Quality Validation Loss (Best):', val_losses[1][1], f'Expected: {0.0047187214}.')
            print('Wine Quality Validation Loss (Overfitted):', val_losses[1][2], f'Expected: {0.006644476}.')
            
            assert np.allclose(val_losses, np.array([[2.633,0.8622,2.1891],[0.03249,0.0047187,0.006644]],dtype=np.float32), rtol=1e-3, atol=1e-4)
        
        except AssertionError as e:
            print(e)
            raise ImportError(f'The loaded model checkpoints are incorrect, validation error does not match expected value.')
        

    print('Successfully set up the data and models, now SHAP and Grad-CAM may be used as in the Jupyter notebooks.')

if __name__ == "__main__":
    #parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-datapath",
        default='./',
        help="full or relative path where datasets should be saved on this computer",
        type=str
    )
    parser.add_argument(
        "-exportdir",
        default='./',
        help="full or relative path where images should be saved on this computer",
        type=str
    )
    parser.add_argument(
        "-validate",
        default=True,
        help="Boolean flag whether to check model checkpoints and data to be correct",
        type=bool
    )
    args = parser.parse_args()
    
    #run main
    main(args)