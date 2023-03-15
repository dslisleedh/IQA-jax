import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from iqa.utils.convert_img import preprocess


def _get_2d_gaussian_kernel(kernel_size: int, sigma: float) -> jnp.ndarray:
    ax = jnp.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size, dtype=jnp.float64)
    gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(sigma))[..., jnp.newaxis]
    kernel = jnp.outer(gauss, gauss.T)
    return (kernel / jnp.sum(kernel))[..., jnp.newaxis, jnp.newaxis]


def _calculate_ssim(
        img1: jnp.ndarray, img2: jnp.ndarray, kernel: jnp.ndarray, c1: float, c2: float
) -> jnp.ndarray:
    n_channels = img1.shape[-1]

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
        img1 ** 2, kernel, window_strides=(1, 1),  padding='VALID', feature_group_count=n_channels,
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
    return ssim_map.mean((1, 2, 3))


def ssim(
        img1: jnp.ndarray, img2: jnp.ndarray, crop_border: int, test_y: bool,
        kernel_size: int = 11, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
) -> jnp.ndarray:
    """
    Calculate SSIM between two images.

    Args:
        img1(jnp.ndarray[int, float]): 0 ~ 255 RGB image
        img2(jnp.ndarray[int, float]): 0 ~ 255 RGB image
        crop_border(int): Crop border size.
        test_y(bool): Whether to use Y channel for PSNR calculation.
        kernel_size(int): Gaussian kernel size.
        sigma(float): Gaussian kernel standard deviation.
        k1(float): constant in the SSIM index formula (0.01 in the original paper)
        k2(float): constant in the SSIM index formula (0.03 in the original paper)

    Returns:
        jnp.ndarray: calculated SSIM value
    """
    img1 = preprocess(img1, crop_border=crop_border, to_y=test_y)
    img2 = preprocess(img2, crop_border=crop_border, to_y=test_y)

    kernel = _get_2d_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    if img1.shape[-1] != 1:
        kernel = jnp.repeat(kernel, 3, axis=-1)
    c1 = (k1 * 255.) ** 2
    c2 = (k2 * 255.) ** 2

    return _calculate_ssim(img1, img2, kernel, c1, c2)


class SSIM:
    def __init__(
            self, crop_border: int = 0, test_y: bool = False,
            kernel_size: int = 11, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03
    ):
        """
        Class for calculating SSIM between two images.
        Maybe faster than ssim function when calling multiple times.

        Args:
            crop_border(int): Crop border size.
            test_y(bool): Whether to use Y channel for PSNR calculation.
            kernel_size(int): Gaussian kernel size.
            sigma(float): Gaussian kernel standard deviation.
            k1(float): constant in the SSIM index formula (0.01 in the original paper)
            k2(float): constant in the SSIM index formula (0.03 in the original paper)
        """
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
