import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import numpy as np

def init_layer_params(input_size, output_size, key):
    W = jax.random.normal(key, shape=(output_size, input_size))
    b = jnp.zeros((output_size,))
    return {"W": W, "b": b}

def init_mlp_params(key, layer_sizes):
    params = []
    # new key for each layer
    keys = random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        layer_params = init_layer_params(layer_sizes[i], layer_sizes[i+1], keys[i])
        params.append(layer_params)
    return params

def linear_layer(params, x):
    W = params["W"]
    b = params["b"]
    return jnp.dot(W, x) + b

def relu(x):
    return jnp.maximum(x, 0)

def mlp_forward(params, x):
    for i, layer in enumerate(params):
        x = linear_layer(layer, x)
        if i < len(params) - 1:
            x = relu(x)
    return x

def mse_loss(params, x, y):
    y_pred = mlp_forward(params, x)
    return jnp.mean((y_pred - y)**2)

def batch_mse_loss(params, x, y):
    loss = vmap(lambda x1, y1:  mse_loss(params, x1, y1))(x, y)
    return jnp.mean(loss)

@jit
def train_step(params, x, y, lr):
    loss, grad = jax.value_and_grad(batch_mse_loss)(params, x, y)
    new_paramters = []
    for p, g in zip(params, grad):
        new_W = p["W"] - lr*g["W"]
        new_b = p["b"] - lr*g["b"]
        new_paramters.append({"W": new_W, "b": new_b})
    return new_paramters, loss

def main():
    key = random.PRNGKey(42)
    data_key, param_key = random.split(key)
    n_samples = 1000
    x_data = random.uniform(data_key, (n_samples, 2), minval = -1, maxval = 1)
    y_data = jnp.sum(x_data**2, axis = 1, keepdims = True)
    layer_sizes = [2, 32, 16, 1]
    params = init_mlp_params(param_key, layer_sizes)

    for i, layer_params in enumerate(params):
        print(f"  Layer {i}: W={layer_params['W'].shape}, b={layer_params['b'].shape}")

    lr = 0.01
    batch_size = 32
    epochs = 50

    for epoch in range(epochs):
        per_epoch_loss = 0.0
        n_batches = n_samples // batch_size

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]

            params, loss = train_step(params, x_batch, y_batch, lr)
            per_epoch_loss += loss

        if epoch % 10 == 0:
            avg_loss = per_epoch_loss / n_batches
            print(f"Epoch {epoch}, Loss: {avg_loss}")

    test_input = jnp.array([[0.5, 0.5], [-0.5, 0.5], [1.0, -1.0]])
    for x in test_input:
        pred = mlp_forward(params, x)
        true = jnp.sum(x**2)
        print(f"Input: {x}, Pred: {pred}, True: {true}")

if __name__ == "__main__":
    main()
