import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np

import cv2
import torch
from basicsr.metrics.metric_util import to_y_channel
from basicsr.utils.matlab_functions import imresize, cubic
from iqa.utils.convert_img import preprocess, rgb2gray
from iqa.utils.convert_img import (
    cubic as cubic_jax,
    calculate_weights_indices as calculate_weights_indices_jax,
    imresize_half as imresize_jax,
)

from absl.testing import absltest, parameterized

from functools import partial
import itertools
from tqdm import tqdm


jax.config.parse_flags_with_absl()


search_space = {
    'is_single': [True, False],
    'use_cpu': [True, False],
}
search_space_list = list(itertools.product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


resize_search_space = {
    'in_length': [96 * 1, 96 * 2, 96 * 3, 96 * 4],
    'antialiasing': [True, False],
}
resize_search_space_list = list(itertools.product(*resize_search_space.values()))
resize_search_space = [dict(zip(resize_search_space.keys(), v)) for v in resize_search_space_list]


class PreprocessingTest(parameterized.TestCase):
    @parameterized.parameters(*search_space)
    def test_y_conversion(self, is_single, use_cpu):
        if is_single:
            inputs = np.random.randint(0., 256., size=(1024, 1024, 3))
        else:
            inputs = np.random.randint(0., 256., size=(32, 1024, 1024, 3))

        inputs_jax = jnp.array(inputs, dtype=jnp.uint8)
        inputs_bsr = inputs.astype(np.uint8)

        if is_single:  # BasicSR uses BGR2YCbCr to get Y channel. So I reversed the channel.
            y_bsr = to_y_channel(inputs_bsr[..., ::-1])
        else:
            y_bsr = []
            for i in range(inputs_bsr.shape[0]):
                y_bsr.append(to_y_channel(inputs_bsr[i][..., ::-1]))
            y_bsr = np.stack(y_bsr)

        device = jax.devices('cpu' if use_cpu else 'gpu')[0]
        inputs_jax = jax.device_put(inputs_jax, device=device)
        func = jax.jit(partial(preprocess, crop_border=0, to_y=True))
        y_iqa = func(inputs_jax)

        np.testing.assert_allclose(y_bsr.squeeze(), y_iqa.squeeze(), atol=1e-4, rtol=1e-4)

    @parameterized.parameters(*search_space)
    def test_gray_conversion(self, is_single, use_cpu):
        if is_single:
            inputs = np.random.randint(0., 256., size=(1024, 1024, 3))
        else:
            inputs = np.random.randint(0., 256., size=(32, 1024, 1024, 3))

        inputs_jax = jnp.array(inputs, dtype=jnp.uint8)
        inputs_bsr = inputs.astype(np.float32)

        if is_single:
            y_bsr = cv2.cvtColor(inputs_bsr[..., ::-1] / 255., cv2.COLOR_BGR2GRAY) * 255.
        else:
            y_bsr = []
            for i in range(inputs_bsr.shape[0]):
                y_bsr.append(cv2.cvtColor(inputs_bsr[i][..., ::-1] / 255., cv2.COLOR_BGR2GRAY) * 255.)
            y_bsr = np.stack(y_bsr)

        device = jax.devices('cpu' if use_cpu else 'gpu')[0]
        inputs_jax = jax.device_put(inputs_jax, device=device)
        func = jax.jit(partial(rgb2gray))
        y_jax = func(inputs_jax)

        np.testing.assert_allclose(y_bsr.squeeze(), y_jax.squeeze(), atol=1e-4, rtol=1e-4)


class TestResize(parameterized.TestCase):
    def test_1cubic_function(self):
        for _ in tqdm(range(100), desc='Testing cubic ...'):
            inputs = np.random.normal((16, 256, 256, 3))
            inputs_jax = jnp.array(inputs, dtype=jnp.float64)
            inputs_bsr = torch.from_numpy(inputs)

            y_bsr = cubic(inputs_bsr)
            y_jax = cubic_jax(inputs_jax)

            np.testing.assert_allclose(y_bsr.squeeze(), y_jax.squeeze(), atol=1e-4, rtol=1e-4)

    @parameterized.parameters(resize_search_space)
    def test_2bicubic_resize(self, in_length, antialiasing):
        img = np.random.normal(size=(1, in_length, in_length, 1))
        img_jax = jnp.array(img, dtype=jnp.float64)

        y_bsr = imresize(img[0], .5, antialiasing=antialiasing)

        func = partial(imresize_jax, antialiasing=antialiasing)
        # func = jax.jit(func)
        y_jax = func(img_jax)

        np.testing.assert_allclose(y_bsr.squeeze(), y_jax.squeeze(), atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    absltest.main()
