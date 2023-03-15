import jax.numpy as jnp

from iqa.utils.convert_img import preprocess


def psnr(
        img1: jnp.ndarray, img2: jnp.ndarray, crop_border: int, test_y: bool
) -> jnp.ndarray:
    """
    Calculate PSNR between two images.

    Args:
        img1: 0 ~ 255 RGB image
        img2: 0 ~ 255 RGB image
        crop_border: Crop border size.
        test_y: Whether to use Y channel for PSNR calculation.

    Returns:
        PSNR value.
    """
    img1 = preprocess(img1, crop_border, test_y)
    img2 = preprocess(img2, crop_border, test_y)

    mse = jnp.mean((img1 - img2) ** 2, axis=(1, 2, 3))
    mask = mse == 0
    return jnp.where(mask, jnp.inf, 10.0 * jnp.log10(255 ** 2 / mse))


class PSNR:
    def __init__(self, crop_border: int, test_y: bool):
        """
        Class for PSNR calculation.
        Args:
            crop_border: Crop border size.
            test_y: Whether to use Y channel for PSNR calculation.
        """
        self.crop_border = crop_border
        self.test_y = test_y

    def __call__(self, img1: jnp.ndarray, img2: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return psnr(img1, img2, self.crop_border, self.test_y)
