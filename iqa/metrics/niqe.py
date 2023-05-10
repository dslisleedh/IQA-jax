import os.path

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy as jsp
import numpy as np
import einops

from iqa.utils.convert_img import rgb2y, rgb2gray, imresize_half

from typing import Literal, Sequence
from functools import partial
import pickle


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
    alpha, beta_l, beta_r = _estimate_aggd_param(block)
    feats = jnp.array([alpha, (beta_l + beta_r) / 2])

    shifts = jnp.array([[0, 1], [1, 0], [1, 1], [1, -1]])

    def _compute_feature_shift(shift):
        shifted_block = jnp.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = _estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (_gamma(2 / alpha) / _gamma(1 / alpha))
        return jnp.array([alpha, mean, beta_l, beta_r])
    feats = jnp.concatenate([feats, jax.vmap(_compute_feature_shift)(shifts).flatten()])
    return feats


def _nancov(x):
    """
    Exclude whole rows that contain NaNs.
    """
    nan_cond = ~jnp.any(jnp.isnan(x), axis=1, keepdims=True)
    n = jnp.sum(nan_cond)

    x_filtered = jnp.where(nan_cond, x, jnp.zeros_like(x))
    x_mean = jnp.sum(x_filtered, axis=0) / n
    x_centered = jnp.where(nan_cond, x - x_mean, jnp.zeros_like(x))
    cov = jnp.matmul(x_centered.T, x_centered) / (n - 1)
    return cov


def _calculate_niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size=96):
    h, w, _ = img.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size
    img = img[jnp.newaxis, :n_blocks_h * block_size, :n_blocks_w * block_size, :].astype(jnp.float32)  # TODO: Fix later
    k_h, k_w = gaussian_window.shape
    gaussian_window = gaussian_window[..., jnp.newaxis, jnp.newaxis].astype(jnp.float32)  # Only 1 channel ( Y, gray )

    distparams = []
    for scale in (1, 2):
        img_pad = jnp.pad(img, ((0, 0), (k_h // 2, k_h // 2), (k_w // 2, k_w // 2), (0, 0)), mode='edge')
        mu = lax.conv_general_dilated(
            img_pad.astype(jnp.float32), gaussian_window, window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        ).astype(jnp.float32)
        sigma = lax.conv_general_dilated(
            jnp.square(img_pad).astype(jnp.float32), gaussian_window, window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        ).astype(jnp.float32) - jnp.square(mu)
        sigma = jnp.sqrt(jnp.abs(sigma))
        img_norm = ((img.astype(jnp.float32) - mu) / (sigma + jnp.ones((1,), dtype=jnp.float32)))[0]

        img_norm = einops.rearrange(  # blocks are arranged from w to h. (w h) b1 b2 c
            img_norm, '(h b1) (w b2) c -> (w h) b1 b2 c', b1=block_size // scale, b2=block_size // scale)
        feats = jax.vmap(_compute_feature)(img_norm)

        distparams.append(jnp.array(feats))

        if scale == 1:
            img = imresize_half(img / 255., antialiasing=True) * 255.

    distparams = jnp.concatenate(distparams, axis=-1)

    mu_dist_param = jnp.nanmean(distparams, axis=0, keepdims=True, dtype=jnp.float64)
    cov_dist_param = _nancov(distparams)

    invcov_dist_params = jnp.linalg.pinv((cov_pris_param + cov_dist_param) / 2, rcond=1e-15)
    val = jnp.matmul(
        jnp.matmul((mu_pris_param - mu_dist_param), invcov_dist_params),
        jnp.transpose((mu_pris_param - mu_dist_param))
    )
    quality = jnp.sqrt(val).squeeze()
    return quality


def niqe(img: jnp.ndarray, crop_border: int, convert_to: Literal['y', 'gray']):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(ROOT_DIR + '/niqe_pris_params.pkl', 'rb') as f:
        loaded = pickle.load(f)
    mu_pris_param = loaded['mu_pris_param']
    cov_pris_param = loaded['cov_pris_param']
    gaussian_window = loaded['gaussian_window']

    if convert_to == 'y':
        img = rgb2y(img)
    elif convert_to == 'gray':
        img = rgb2gray(img)
    else:
        raise ValueError(f'Unknown convert_to value: {convert_to}')

    if crop_border > 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border, :]

    img = img.round().astype(jnp.float64)

    calc_func = partial(
        _calculate_niqe, mu_pris_param=mu_pris_param, cov_pris_param=cov_pris_param, gaussian_window=gaussian_window)
    quality = jax.vmap(calc_func)(img)
    return quality


class NIQE:
    def __init__(self, crop_border: int, convert_to: Literal['y', 'gray']):
        self.crop_border = crop_border
        self.convert_to = convert_to
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        with open(ROOT_DIR + '/niqe_pris_params.pkl', 'rb') as f:
            loaded = pickle.load(f)
        self.mu_pris_param = loaded['mu_pris_param']
        self.cov_pris_param = loaded['cov_pris_param']
        self.gaussian_window = loaded['gaussian_window']

    def __call__(self, img: jnp.ndarray):
        if self.convert_to == 'y':
            img = rgb2y(img)
        elif self.convert_to == 'gray':
            img = rgb2gray(img)
        else:
            raise ValueError(f'Unknown convert_to value: {self.convert_to}')

        if self.crop_border > 0:
            img = img[:, self.crop_border:-self.crop_border, self.crop_border:-self.crop_border, :]

        img = img.round().astype(jnp.float64)

        calc_func = partial(
            _calculate_niqe, mu_pris_param=self.mu_pris_param, cov_pris_param=self.cov_pris_param,
            gaussian_window=self.gaussian_window)
        quality = jax.vmap(calc_func)(img)
        return quality
