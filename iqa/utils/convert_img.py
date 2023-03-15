import jax.numpy as jnp


def rgb2y(img: jnp.ndarray) -> jnp.ndarray:
    """
    Convert RGB image to Y channel.

    Args:
        img(jnp.ndarray[int, float]): 0 ~ 255 RGB image

    Returns:
        jnp.ndarray[float32]: Y channel image

    """
    img = (img.astype(jnp.float64) / jnp.array(255., dtype=jnp.float64)).astype(jnp.float32)

    out_img = jnp.dot(img, jnp.array([65.481, 128.553, 24.966], dtype=jnp.float64)) \
              + jnp.array(16.0, dtype=jnp.float64)

    return out_img[..., jnp.newaxis].astype(jnp.float32)


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
