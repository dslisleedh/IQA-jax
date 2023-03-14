import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np

from basicsr.utils.color_util import bgr2ycbcr
from iqa.utils.convert_img import rgb2y, preprocess

from absl.testing import absltest, parameterized

from functools import partial
import itertools


jax.config.parse_flags_with_absl()


search_space = {
    'is_single': [True, False],
    'input_type': ['uint8', 'float32'],
    'use_cpu': [True, False],
}
search_space_list = list(itertools.product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


class YConversionTest(parameterized.TestCase):
    @parameterized.parameters(*search_space)
    def test_y_conversion(self, is_single, input_type, use_cpu):
        if is_single:
            inputs = np.random.randint(0., 256., size=(256, 256, 3))
        else:
            inputs = np.random.randint(0., 256., size=(256, 256, 256, 3))

        if input_type == 'float32':
            inputs_jax = inputs.astype(jnp.float32) / 255.
            inputs_bsr = inputs.astype(np.float32) / 255.
        else:
            inputs_jax = inputs.astype(jnp.uint8)
            inputs_bsr = inputs.astype(np.uint8)

        y_bsr = bgr2ycbcr(inputs_bsr[..., ::-1], y_only=True)  # They use BGR2RGB to get Y

        device = jax.devices('cpu' if use_cpu else 'gpu')[0]
        inputs_jax = jax.device_put(inputs_jax, device=device)
        func = jax.jit(partial(rgb2y))
        y_iqa = func(inputs_jax)

        np.testing.assert_allclose(y_bsr, y_iqa[..., 0])


class PreprocessingTest(parameterized.TestCase):
    @parameterized.parameters(*search_space)
    def test_preprocessing(self, is_single, input_type, use_cpu):
        if is_single:
            inputs = np.random.randint(0., 256., size=(256, 256, 3))
        else:
            inputs = np.random.randint(0., 256., size=(128, 256, 256, 3))

        if input_type == 'float32':
            inputs_jax = inputs.astype(jnp.float32) / 255.
            inputs_bsr = inputs.astype(np.uint8)
        else:
            inputs_jax = inputs.astype(jnp.uint8)
            inputs_bsr = inputs.astype(np.uint8)

        y_bsr = bgr2ycbcr(inputs_bsr[..., ::-1], y_only=True)  # They use BGR2RGB to get Y

        device = jax.devices('cpu' if use_cpu else 'gpu')[0]
        inputs_jax = jax.device_put(inputs_jax, device=device)
        func = jax.jit(partial(preprocess, crop_border=0, to_y=True))
        y_iqa = func(inputs_jax)

        np.testing.assert_allclose(y_bsr, jnp.squeeze(y_iqa).round().astype(jnp.uint8))


if __name__ == '__main__':
    absltest.main()
