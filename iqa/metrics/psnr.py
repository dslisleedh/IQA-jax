import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np

from iqa.utils.convert_img import preprocess


def psnr(
        img1: jnp.ndarray, img2: jnp.ndarray, crop_border: int, test_y: bool
) -> jnp.ndarray:
    img1 = preprocess(img1, crop_border, test_y)
    img2 = preprocess(img2, crop_border, test_y)

    mse = jnp.mean(jnp.square(img1 - img2), axis=(1, 2, 3), dtype=jnp.float64)
    mask = mse == 0
    val = 10 * jnp.log10(jnp.square(jnp.array(255., dtype=jnp.float64)) / mse)
    return jnp.where(mask, jnp.inf, val)


class PSNR:
    def __init__(self, crop_border: int, test_y: bool):
        self.crop_border = crop_border
        self.test_y = test_y

    def __call__(self, img1: jnp.ndarray, img2: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return psnr(img1, img2, self.crop_border, self.test_y)
