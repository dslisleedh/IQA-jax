import jax
import jax.numpy as jnp
import jax.lax as lax

from typing import Sequence


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
    https://github.com/XPixelGroup/BasicSR/wiki/Color-conversion-in-SR

    Args:
        img(jnp.ndarray[int, float]): 0 ~ 255 RGB image (N, H, W, C)

    Returns:
        jnp.ndarray[float32]: Gray-scale image( 0 ~ 255 )
    """
    img = img.astype(jnp.float64)
    img_gray = jnp.dot(img, jnp.array([0.299, 0.587, 0.114], dtype=jnp.float64))
    return img_gray


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
        in_length, out_length, scale, kernel, kernel_width, antialias: bool, p
) -> Sequence:
    if scale < 1 and antialias:
        kernel_width = kernel_width / scale

    x = jnp.linspace(1, out_length, out_length, dtype=jnp.float64)
    u = x / scale + 0.5 * (1. - 1. / scale)
    left = jnp.floor(u - kernel_width / 2.).astype(jnp.int32)

    indices = left[..., jnp.newaxis].repeat(p, axis=-1) \
              + jnp.linspace(0, p - 1, p, dtype=jnp.int32)[jnp.newaxis, ...].repeat(out_length, axis=0)

    distance_to_center = u[..., jnp.newaxis].repeat(p, axis=-1) - indices

    if scale < 1 and antialias:
        weights = scale * kernel(distance_to_center * scale)
    else:
        weights = kernel(distance_to_center)

    weighted_sum = jnp.sum(weights, axis=1, keepdims=True)
    weights /= weighted_sum

    weights_zero_tmp = jnp.sum((weights == 0), axis=0)
    """
    Orginal code Cut off the weights of the boundary pixels.
    But jax.lax.cond can't deal with different shape per path. So I just set the weights to 0.
    indices might cause problem, But at least in test code, it works.
    """
    weights = lax.cond(
        weights_zero_tmp[0] > 0, lambda x: x.at[:, :1].set(0.).at[:, p-1:].set(0.), lambda x: x, weights
    )
    weights = lax.cond(
        weights_zero_tmp[-1] > 0, lambda x: x.at[:, p - 2:].set(0.), lambda x: x, weights
    )
    sym_len_s = -jnp.min(indices) + 1
    sym_len_e = jnp.max(indices) - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, sym_len_s.astype(jnp.int32), sym_len_e.astype(jnp.int32)


def imresize(img: jnp.ndarray, scale: float | int, antialiasing: bool = True) -> jnp.ndarray:
    if img.ndim == 3:
        img = img[jnp.newaxis, ...]
    _, h, w, c = img.shape
    h_new = int(h * scale)
    w_new = int(h * scale)
    kernel_size = 4
    kernel = cubic
    p = kernel_size + 2

    weights_h, indices_h, sym_len_s_h, sym_len_e_h = calculate_weights_indices(
        h, h_new, scale, kernel, kernel_size, antialiasing, p
    )
    weights_w, indices_w, sym_len_s_w, sym_len_e_w = calculate_weights_indices(
        w, w_new, scale, kernel, kernel_size, antialiasing, p
    )

    # H-wise First
    img = jnp.pad(img, ((0, 0), (sym_len_s_h, sym_len_e_h), (0, 0), (0, 0)), mode='symmetric')
    def calc_h(x, indices, weights):
        stacked = jnp.take_along_axis(x, indices[jnp.newaxis, :, jnp.newaxis, jnp.newaxis], axis=1)
        weighted_product = stacked * weights[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
        return jnp.sum(weighted_product, axis=1, keepdims=False)
    img = jax.vmap(calc_h, in_axes=(None, 0, 0), out_axes=1)(img, indices_h, weights_h)

    # W-wise Second
    img = jnp.pad(img, ((0, 0), (0, 0), (sym_len_s_w, sym_len_e_w), (0, 0)), mode='symmetric')
    def calc_w(x, indices, weights):
        stacked = jnp.take_along_axis(x, indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis], axis=2)
        weighted_product = stacked * weights[jnp.newaxis, jnp.newaxis, ..., jnp.newaxis]
        return jnp.sum(weighted_product, axis=2, keepdims=False)
    img = jax.vmap(calc_w, in_axes=(None, 0, 0), out_axes=2)(img, indices_w, weights_w)

    return img
