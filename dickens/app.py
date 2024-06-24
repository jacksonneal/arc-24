import jax
import jax.numpy as jnp

from loguru import logger


logger.debug(f"Using jax {jax.__version__}")

a = jnp.zeros((2, 5), dtype=jnp.float32)

logger.debug(a)

b = jnp.arange(6)

logger.debug(b)

logger.debug(b.__class__)

logger.debug(b.devices())

b_cpu = jax.device_get(b)
logger.debug(b_cpu.__class__)

b_gpu = jax.device_put(b_cpu)
logger.debug(f"Device put: {b_gpu.__class__} on {b_gpu.devices()}")

logger.debug(f"{(b_cpu + b_gpu).__class__}")

logger.debug(jax.devices())

b_new = b.at[0].set(1)
logger.debug(f"Original array: {b}")
logger.debug(f"Changed array: {b_new}")
