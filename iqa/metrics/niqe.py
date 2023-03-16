import os.path

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy as jsp
import numpy as np

from iqa.utils.convert_img import rgb2y, rgb2gray

from typing import Literal, Sequence


"""
Incomplete file.
Currently working on this but there's so many functions to implement this.
"""


def _gamma(x):
    """
    There's no gamma function in JAX, so we use the log abs gamma function and exp function instead.
    """
    return jnp.exp(jsp.special.gammaln(x))


def _estimate_aggd_param(block) -> Sequence[jnp.ndarray]:
    block = block.flatten()
    gam = jnp.arange(0.2, 10.001, 0.001)
    gam_reciprocal = jnp.reciprocal(gam)
    r_gam = jnp.square(_gamma(gam_reciprocal * 2)) / (_gamma(gam_reciprocal) * _gamma(gam_reciprocal * 3))

    left_std = jnp.sqrt(
        jnp.nanmean(jnp.where(block < 0, jnp.square(block), jnp.nan))
    )
    right_std = jnp.sqrt(
        jnp.nanmean(jnp.where(block > 0, jnp.square(block), jnp.nan))
    )
    gamma_hat = left_std / right_std
    rhat = (jnp.mean(jnp.abs(block))) ** 2 / jnp.mean(jnp.square(block))
    rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)
    array_position = jnp.argmin(jnp.square(r_gam - rhat_norm))

    alpha = gam[array_position]
    beta_l = left_std * jnp.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    beta_r = right_std * jnp.sqrt(_gamma(1 / alpha) / _gamma(3 / alpha))
    return alpha, beta_l, beta_r


def _compute_feature(block):
    feats = []

    # Using vmap instead of a for loop to speed up the computation


def _calculate_niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size=96):
    _, h, w, _ = img.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    img = img[:, :n_blocks_h * block_size, :n_blocks_w * block_size, :]
    k_h, k_w = gaussian_window.shape
    gaussian_window = gaussian_window[..., jnp.newaxis, jnp.newaxis]

    distparam = []
    img_pad = jnp.pad(img, ((0, 0), (k_h//2, k_h//2), (k_w//2, k_w//2), (0, 0)), mode='edge')
    mu = lax.conv_general_dilated(
        img_pad, gaussian_window, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    sigma = lax.conv_general_dilated(
        jnp.square(img_pad), gaussian_window, window_strides=(1, 1), padding='VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    ) - jnp.square(mu)
    sigma = jnp.sqrt(jnp.abs(sigma))
    img_norm = (img - mu) / (sigma + 1)

    for i in (1, 2):
        a = 1



def niqe(img: jnp.ndarray, crop_border: int, convert_to: Literal['y', 'gray']):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    f = np.load(os.path.join(ROOT_DIR, 'niqe_model.npz'))
    mu_pris_param = f['mu_pris_param']
    cov_pris_param = f['cov_pris_param']
    gaussian_window = f['gaussian_window']

    if convert_to == 'y':
        img = rgb2y(img)
    elif convert_to == 'gray':
        img = rgb2gray(img)
    else:
        raise ValueError(f'Unknown convert_to value: {convert_to}')

    if crop_border > 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border, :]

    img = img.round()
    return _calculate_niqe(img, mu_pris_param, cov_pris_param, gaussian_window)