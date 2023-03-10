import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np


def rgb2y(img: jnp.ndarray) -> jnp.ndarray:
    input_dtype = img.dtype
    if input_dtype == jnp.uint8:
        img = (img / jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)
    elif input_dtype == jnp.float32:
        pass
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {input_dtype}')

    out_img = jnp.dot(img, jnp.array([65.481, 128.553, 24.966], dtype=jnp.float64)) + 16.0

    if input_dtype == jnp.uint8:
        out_img = out_img.round()
    else:
        out_img = out_img / jnp.array(255., dtype=jnp.float64)
    return out_img.astype(input_dtype)[..., jnp.newaxis]


def preprocess(img: jnp.ndarray, crop_border: int, to_y: bool) -> jnp.ndarray:
    if img.ndim == 3:
        img = img[jnp.newaxis, ...]

    input_type = img.dtype
    if input_type == jnp.uint8:
        img = (img / jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)
    elif input_type == jnp.float32:
        pass
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {input_type}')

    if to_y:
        img = rgb2y(img)

    if input_type == jnp.uint8:
        img = img.round().astype(jnp.float64)
    else:
        img = img / jnp.array(255., dtype=jnp.float64)

    if crop_border > 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border, :]

    return img.astype(input_type)
