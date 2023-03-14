import warnings

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg


def fid(feats_1: np.array, feats_2: np.array, eps: float = 1e-6) -> np.array:
    """
    Current Jax's sqrtm implementation is different from scipy's sqrtm.
    So this implementation is just copy of BasicSR's FID calculation.
    It will be fixed in the future.
    Not directly use BasicSR's FID calculation to prevent issues.
    https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/fid.py

    Not incorporate feature extraction in this function because original Numpy does not support Jit-compile
    while Flax's module does.
    """
    warnings.warn("Current fid does not support Jit-compile. It will be fixed in the future.")

    mu1 = np.mean(feats_1, axis=0)
    mu2 = np.mean(feats_2, axis=0)

    sigma1 = np.cov(feats_1, rowvar=False)
    sigma2 = np.cov(feats_2, rowvar=False)

    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Product might be almost singular
    if not np.isfinite(cov_sqrt).all():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)
    fid_score = mean_norm + trace

    return fid_score


def fid_jax(
        imgs_1: jnp.ndarray, imgs_2: jnp.ndarray, feats_extractor: nn.Module, eps: float = 1e-6
) -> jnp.ndarray:
    """
    This implementation results different from fid() because of Jax's sqrtm implementation.
    Use this only when Jit-compile must be used.
    """
    warnings.warn("fid_jax function results different from original FID calculation."
                  "Use fid() unless You must use jit-compile.")

    feats_1 = feats_extractor(imgs_1)
    feats_2 = feats_extractor(imgs_2)

    mu1 = jnp.mean(feats_1, axis=0)
    mu2 = jnp.mean(feats_2, axis=0)

    sigma1 = jnp.cov(feats_1, rowvar=False)
    sigma2 = jnp.cov(feats_2, rowvar=False)

    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Two covariances have different dimensions'

    cov_sqrt = jax.scipy.linalg.sqrtm(sigma1 @ sigma2)

    # Product might be almost singular
    if jnp.isfinite(cov_sqrt).any():
        print('Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = jnp.eye(sigma1.shape[0]) * eps
        cov_sqrt = jax.scipy.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # Numerical error might give slight imaginary component
    if jnp.iscomplexobj(cov_sqrt):
        if not jnp.allclose(jnp.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = jnp.max(jnp.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real

    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff
    trace = jnp.trace(sigma1) + jnp.trace(sigma2) - 2 * jnp.trace(cov_sqrt)
    fid_score = mean_norm + trace

    return fid_score
