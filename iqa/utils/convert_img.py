import jax
import jax.numpy as jnp
import jax.lax as lax

from typing import Sequence
from functools import partial


def rgb2y(img: jnp.ndarray) -> jnp.ndarray:
    """
    Convert RGB image to Y channel.
    https://github.com/XPixelGroup/BasicSR/wiki/Color-conversion-in-SR

    Args:
        img(jnp.ndarray[int, float]): 0 ~ 255 RGB image (N, H, W, C)

    Returns:
        jnp.ndarray[float32]: Y channel image

    """
    img = (img.astype(jnp.float64) / jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)

    out_img = jnp.dot(img, jnp.array([65.481, 128.553, 24.966], dtype=jnp.float64)) \
              + jnp.array(16.0, dtype=jnp.float64)

    return out_img[..., jnp.newaxis].astype(jnp.float32)


def rgb2gray(img: jnp.ndarray) -> jnp.ndarray:
    """
    Convert RGB image to gray-scale image.

    Args:
        img(jnp.ndarray[int, float]): 0 ~ 255 RGB image (N, H, W, C) or (H, W, C)

    Returns:
        jnp.ndarray[float32]: 0 ~ 255 Gray-scale image
    """
    # m = (jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32) * 1000).astype(jnp.int64) << 32
    # return ((jnp.dot(img.astype(jnp.int64), m) >> 32).astype(jnp.float64) / jnp.array(1000., dtype=jnp.float64))[..., jnp.newaxis]

    img = (img.astype(jnp.float64) / jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)
    img_gray = jnp.dot(
        img.astype(jnp.float64), jnp.array([0.299, 0.587, 0.114], dtype=jnp.float64))[..., jnp.newaxis]
    return (img_gray.astype(jnp.float64) * jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)


def preprocess(img: jnp.ndarray, crop_border: int, to_y: bool) -> jnp.ndarray:
    """
    Preprocessing images for calculate metrics.

    Args:
        img(jnp.ndarray[int, float]): 0 ~ 255 RGB image
        crop_border(int): Crop border size.
        to_y(bool): Whether to only return Y channel

    Returns:
        jnp.ndarray[float64]: Preprocessed image

    """
    if img.ndim == 3:
        img = img[jnp.newaxis, ...]

    if to_y:
        img = rgb2y(img)

    img = img.astype(jnp.float64)

    if crop_border > 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border, :]

    return img


def cubic(x: jnp.ndarray) -> jnp.ndarray:
    absx = jnp.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    cond_1 = (absx <= 1).astype(jnp.float64)
    cond_2 = ((1 < absx) & (absx <= 2)).astype(jnp.float64)
    return (1.5 * absx3 - 2.5 * absx2 + 1) * cond_1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * cond_2


def calculate_weights_indices(
        in_length, out_length, scale, kernel, kernel_size, antialias
) -> Sequence:
    if scale < 1 and antialias:
        kernel_size = kernel_size / scale

    p = int(kernel_size + 2)

    x = jnp.linspace(1, out_length, out_length, dtype=jnp.float64)
    u = x / scale + 0.5 * (1. - 1. / scale)
    left = jnp.floor(u - kernel_size / 2.).astype(jnp.int32)

    indices = left[..., jnp.newaxis].repeat(p, axis=-1) \
              + jnp.linspace(0, p - 1, p, dtype=jnp.int32)[jnp.newaxis, ...].repeat(out_length, axis=0)

    distance_to_center = u[..., jnp.newaxis].repeat(p, axis=-1) - indices

    if scale < 1 and antialias:
        weights = scale * kernel(distance_to_center * scale)
    else:
        weights = kernel(distance_to_center)

    weighted_sum = jnp.sum(weights, axis=1, keepdims=True)
    weights /= weighted_sum

    weights = lax.slice(weights, (0, 1), (out_length, p - 1))
    indices = lax.slice(indices, (0, 1), (out_length, p - 1))

    sym_len_s = kernel_size // 2 - 1
    sym_len_e = kernel_size // 2 - 1
    indices = indices + sym_len_s - 1
    return weights, indices.astype(jnp.int32), int(sym_len_s), int(sym_len_e)


def imresize_half(img: jnp.ndarray, antialiasing: bool = True) -> jnp.ndarray:
    """
    It is hard to implement interpolation function for every scale.
    So, I implemented interpolation for half scale that needed for calculate NIQE.

    Args:
        img:
        antialiasing:

    Returns:

    """
    if img.ndim == 3:
        img = img[jnp.newaxis, ...]
    _, h, w, c = img.shape
    scale = .5
    h_new = int(h * scale)
    w_new = int(w * scale)
    kernel = cubic
    kernel_size = 4
    img = img.round(decimals=4)

    weights_h, indices_h, sym_len_s_h, sym_len_e_h = calculate_weights_indices(
        h, h_new, scale, kernel, kernel_size, antialiasing
    )
    weights_w, indices_w, sym_len_s_w, sym_len_e_w = calculate_weights_indices(
        w, w_new, scale, kernel, kernel_size, antialiasing
    )
    weights_w = weights_w.round(decimals=4)
    weights_h = weights_h.round(decimals=4)

    weights_h_conv = weights_h[:1, :, jnp.newaxis, jnp.newaxis].repeat(c, axis=-1).transpose((1, 0, 2, 3))

    # I want to use conv, but results differ from matlab.
    # H-wise First
    img_aug = jnp.pad(img, ((0, 0), (sym_len_s_h, sym_len_e_h), (0, 0), (0, 0)), mode='symmetric')
    def calc_h(x, indices, weights):
        stacked = jnp.take_along_axis(x, indices[jnp.newaxis, :, jnp.newaxis, jnp.newaxis], axis=1)
        weighted_product = stacked * weights[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
        return jnp.sum(weighted_product, axis=1, keepdims=False)
    img = jax.vmap(
        calc_h, in_axes=(None, 0, 0), out_axes=1)(img_aug, indices_h, weights_h)
    img = jnp.round(img, decimals=4)

    img_conv = lax.conv_general_dilated(
        img_aug, weights_h_conv, (2, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGH
    )
    img_conv = jnp.round(img_conv, decimals=4)

    # W-wise Second
    img = jnp.pad(img, ((0, 0), (0, 0), (sym_len_s_w, sym_len_e_w), (0, 0)), mode='symmetric')
    def calc_w(x, indices, weights):
        stacked = jnp.take_along_axis(x, indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis], axis=2)
        weighted_product = stacked * weights[jnp.newaxis, jnp.newaxis, ..., jnp.newaxis]
        return jnp.sum(weighted_product, axis=2, keepdims=False)
    img = jax.vmap(
        calc_w, in_axes=(None, 0, 0), out_axes=2)(img, indices_w, weights_w)
    return img
