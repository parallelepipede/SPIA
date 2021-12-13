import numpy as np
import jax.numpy as jnp
from jax import grad, jit,vmap
from jax import random, device_put

key = random.PRNGKey(0)
x = random.normal(key,(10,))
print(x)

size= 3_000
x = random.normal(key, (size,size), dtype=jnp.float32)

import time

tim = time.time()
jnp.dot(x,x.T).block_until_ready()
print(time.time() - tim)
x  = np.random.normal(size = (size,size)).astype(np.float32)
x = device_put(x)
x = jnp.dot(x,x.T).block_until_ready()

def selu(x, alpha = 1.67, lmbda = 1.05):
    return lambda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (10_000, ))
selu(x).block_until_ready()
print("hello world")