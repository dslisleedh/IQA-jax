import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import einops

from iqa.utils.convert_img import preprocess


def _get_2d_gaussian_kernel(kernel_size: int, sigma: float) -> jnp.ndarray:
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)[..., np.newaxis, np.newaxis]
    return jnp.array(kernel / np.sum(kernel), dtype=jnp.float64)


def _calculate_ssim(
        img1: jnp.ndarray, img2: jnp.ndarray, kernel: jnp.ndarray,
        c1: float, c2: float
) -> jnp.ndarray:
    n_channels = img1.shape[-1] if img1.shape[-1] > 0 else None

    mu1 = lax.conv_general_dilated(
        img1, kernel, window_strides=(1, 1), padding='VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    mu2 = lax.conv_general_dilated(
        img2, kernel, window_strides=(1, 1), padding='VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = lax.conv_general_dilated(
        img1 ** 2, kernel, window_strides=(1, 1), padding='VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) - mu1_sq
    sigma2_sq = lax.conv_general_dilated(
        img2 ** 2, kernel, window_strides=(1, 1), padding='VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) - mu2_sq
    sigma12 = lax.conv_general_dilated(
        img1 * img2, kernel, window_strides=(1, 1), padding='VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return jnp.mean(ssim_map, axis=(1, 2, 3))


def ssim(
        img1: jnp.ndarray, img2: jnp.ndarray, kernel_size: int = 11, sigma: float = 1.5,
        k1: float = 0.01, k2: float = 0.03, crop_border: int = 0, test_y: bool = False
) -> jnp.ndarray:
    img1 = preprocess(img1, crop_border=crop_border, to_y=test_y)
    img2 = preprocess(img2, crop_border=crop_border, to_y=test_y)

    img1 = img1.astype(jnp.float64)
    img2 = img2.astype(jnp.float64)

    kernel = _get_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    if img1.shape[-1] != 1:
        kernel = jnp.repeat(kernel, 3, axis=-1)
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2

    return _calculate_ssim(img1, img2, kernel, c1, c2)


class SSIM:
    def __init__(
            self, kernel_size: int = 11, sigma: float = 1.5, k1: float = 0.01,
            k2: float = 0.03, crop_border: int = 0, test_y: bool = False
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = _get_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
        self.kernel = jnp.repeat(kernel, 3, axis=-1) if not test_y else kernel
        self.k1 = k1
        self.k2 = k2
        self.c1 = (k1 * 255.) ** 2
        self.c2 = (k2 * 255.) ** 2
        self.crop_border = crop_border
        self.test_y = test_y

    def __call__(self, img1: jnp.ndarray, img2: jnp.ndarray) -> jnp.ndarray:
        img1 = preprocess(img1, crop_border=self.crop_border, to_y=self.test_y)
        img2 = preprocess(img2, crop_border=self.crop_border, to_y=self.test_y)
        return _calculate_ssim(img1, img2, self.kernel, self.c1, self.c2)
