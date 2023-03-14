import jax.numpy as jnp

from typing import Optional


def rgb2y(img: jnp.ndarray, return_dtype: Optional = None) -> jnp.ndarray:
    input_dtype = img.dtype
    return_dtype = return_dtype or input_dtype

    if input_dtype == jnp.uint8:
        img = (img.astype(jnp.float64) / jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)
    elif input_dtype == jnp.float32:
        pass
    else:
        raise TypeError(f'The img type should be jnp.float32 or jnp.uint8, but got {input_dtype}')

    out_img = jnp.dot(img, jnp.array([65.481, 128.553, 24.966], dtype=jnp.float64)) + 16.0

    if return_dtype == jnp.uint8:
        out_img = out_img.round()
    else:
        out_img = out_img / jnp.array(255., dtype=jnp.float64)
    return out_img.astype(return_dtype)[..., jnp.newaxis]


def preprocess(img: jnp.ndarray, crop_border: int, to_y: bool) -> jnp.ndarray:
    if img.ndim == 3:
        img = img[jnp.newaxis, ...]

    if to_y:
        img = rgb2y(img, return_dtype=jnp.uint8)

    if img.dtype == jnp.uint8:
        img = img.astype(jnp.float64)
    elif img.dtype == jnp.float32:
        img = img.astype(jnp.float64) * jnp.array(255., dtype=jnp.float64)
    else:
        raise TypeError(f'The img type should be jnp.float32 or jnp.uint8, but got {img.dtype}')

    if crop_border > 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border, :]

    return img
