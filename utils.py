import jax
import jax.numpy as jnp

def validate_cifar(data, model, weights):
    """validate cifar10"""
    loss = 0
    for image, label in data:
        image = jnp.expand_dims(image, axis=0) #make batch size 1
        pred = model.apply(weights, image)
        loss += jnp.mean(jax.vmap(cross_entropy_loss)(pred, label)).item()
    
    return loss

def validate_wine(data, model, weights):
    """validate wine quality"""
    loss = 0
    for features, label in data:
        features = jnp.expand_dims(features, axis=0) #make batch size 1
        pred = model.apply(weights, features)
        loss += jnp.mean(jax.vmap(mse_loss)(pred, label)).item()
    
    return loss

def prediction(weights, model, data):
    """Returns the prediction for given data and weights using model."""
    return model.apply(weights, data)

def cross_entropy_loss(logits, labels):
    """Calculates the cross entropy loss between logits and labels."""
    return -jnp.sum(labels * jax.nn.log_softmax(logits))

def mse_loss(predictions, targets):
    """Calculates the mean squared error loss between predictions and targets."""
    return jnp.mean(jnp.square(predictions - targets))


def get_grad_cam(model, weights, input_data, target_class=None):
    import jax.numpy as jnp
    import jax
    
    # Define a function to get model logits
    def model_with_logits(weights, x):
        logits = model.apply(weights, x)
        return logits
    
    # Get the target class if not provided
    if target_class is None:
        target_class = jnp.argmax(model.apply(weights, input_data))
    
    # Calculate gradients with respect to the target class
    grad_fn = jax.grad(model_with_logits)
    grads = grad_fn(weights, input_data)
    
    # Calculate the forward pass
    logits = model.apply(weights, input_data)
    
    # Calculate Grad-CAM
    weights = jnp.mean(grads, axis=(1, 2))
    cam = jnp.zeros(logits.shape[1:3])
    
    for i, w in enumerate(weights):
        cam += w * logits[..., target_class]
    
    cam = jnp.maximum(cam, 0)
    cam = cam / jnp.max(cam)
    
    return cam


