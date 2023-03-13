import jax
import jax.numpy as jnp
import numpy as np

from iqa.utils.convert_img import preprocess


def _get_2d_gaussian_kernel(kernel_size: int, sigma: float) -> jnp.ndarray:
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)[..., np.newaxis, np.newaxis]
    return jnp.array(kernel / np.sum(kernel), dtype=jnp.float64)


def _calculate_ssim(
        img1: jnp.ndarray, img2: jnp.ndarray, kernel: jnp.ndarray,
        c1: jnp.ndarray, c2: jnp.ndarray
) -> jnp.ndarray:
    n_channels = img1.shape[-1]
    mu1 = jax.lax.conv_general_dilated(
        img1, kernel, (1, 1), 'VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=jax.lax.Precision.HIGHEST)
    mu2 = jax.lax.conv_general_dilated(
        img2, kernel, (1, 1), 'VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=jax.lax.Precision.HIGHEST)

    mu1_sq = jnp.square(mu1)
    mu2_sq = jnp.square(mu2)
    mu12 = mu1 * mu2

    sigma1_sq = jax.lax.conv_general_dilated(
        jnp.square(img1), kernel, (1, 1), 'VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=jax.lax.Precision.HIGHEST
    ) - mu1_sq
    sigma2_sq = jax.lax.conv_general_dilated(
        jnp.square(img2), kernel, (1, 1), 'VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=jax.lax.Precision.HIGHEST
    ) - mu2_sq
    sigma12 = jax.lax.conv_general_dilated(
        img1 * img2, kernel, (1, 1), 'VALID', feature_group_count=n_channels,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=jax.lax.Precision.HIGHEST
    ) - mu12

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu12 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return jnp.mean(ssim_map, axis=(1, 2, 3))


def ssim(
        img1: jnp.ndarray, img2: jnp.ndarray, kernel_size: int = 11, sigma: float = 1.5,
        k1: float = 0.01, k2: float = 0.03, crop_border: int = 0, calculate_y: bool = False
) -> jnp.ndarray:
    img1 = preprocess(img1, crop_border, calculate_y)
    img2 = preprocess(img2, crop_border, calculate_y)

    kernel = _get_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    if not calculate_y:
        kernel = jnp.repeat(kernel, 3, axis=-1)
    c1 = (jnp.array(k1 * 255., dtype=jnp.float64)) ** 2
    c2 = (jnp.array(k2 * 255., dtype=jnp.float64)) ** 2

    return _calculate_ssim(img1, img2, kernel, c1, c2)


class SSIM:
    def __init__(
            self, kernel_size: int = 11, sigma: float = 1.5, k1: float = 0.01,
            k2: float = 0.03, crop_border: int = 0, calculate_y: bool = False
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        kernel = _get_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
        self.kernel = jnp.repeat(kernel, 3, axis=-1) if not calculate_y else kernel
        self.k1 = k1
        self.k2 = k2
        self.c1 = (jnp.array(k1 * 255., dtype=jnp.float64)) ** 2
        self.c2 = (jnp.array(k2 * 255., dtype=jnp.float64)) ** 2
        self.crop_border = crop_border
        self.calculate_y = calculate_y

    def __call__(self, img1: jnp.ndarray, img2: jnp.ndarray) -> jnp.ndarray:
        img1 = preprocess(img1, self.crop_border, self.calculate_y)
        img2 = preprocess(img2, self.crop_border, self.calculate_y)
        return _calculate_ssim(img1, img2, self.kernel, self.c1, self.c2)
