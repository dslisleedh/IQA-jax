import jax.numpy as jnp

from iqa.utils.convert_img import preprocess


def psnr(
        img1: jnp.ndarray, img2: jnp.ndarray, crop_border: int, test_y: bool
) -> jnp.ndarray:
    img1 = preprocess(img1, crop_border, test_y)
    img2 = preprocess(img2, crop_border, test_y)

    mse = jnp.mean((img1 - img2) ** 2, axis=(1, 2, 3))
    mask = mse == 0
    return jnp.where(mask, 1.0, 10.0 * jnp.log10(255 ** 2 / mse))


class PSNR:
    def __init__(self, crop_border: int, test_y: bool):
        self.crop_border = crop_border
        self.test_y = test_y

    def __call__(self, img1: jnp.ndarray, img2: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return psnr(img1, img2, self.crop_border, self.test_y)
