import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np

from basicsr.utils.color_util import rgb2ycbcr
from iqa.utils.convert_img import rgb2y

from absl.testing import absltest, parameterized

import itertools


jax.config.update('jax_enable_x64', True)
jax.config.parse_flags_with_absl()


search_space = {
    'is_single': [True, False],
    'input_type': ['uint8', 'float32'],
}
search_space_list = list(itertools.product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


class YConversionTest(parameterized.TestCase):
    @parameterized.parameters(*search_space)
    def test_y_conversion(self, is_single, input_type):
        if is_single:
            inputs = np.random.randint(0., 256., size=(256, 256, 3))
        else:
            inputs = np.random.randint(0., 256., size=(16, 256, 256, 3))

        if input_type == 'float32':
            inputs_jax = inputs.astype(jnp.float32) / 255.
            inputs_bsr = inputs.astype(np.float32) / 255.
        else:
            inputs_jax = inputs.astype(jnp.uint8)
            inputs_bsr = inputs.astype(np.uint8)

        y_bsr = rgb2ycbcr(inputs_bsr, y_only=True)
        y_iqa = rgb2y(inputs_jax)

        np.testing.assert_allclose(y_bsr, y_iqa[..., 0])


if __name__ == '__main__':
    absltest.main()
