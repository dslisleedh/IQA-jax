import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np

from basicsr.utils.color_util import bgr2ycbcr
from basicsr.metrics.metric_util import to_y_channel
from iqa.utils.convert_img import rgb2y, preprocess

from absl.testing import absltest, parameterized

from functools import partial
import itertools


jax.config.parse_flags_with_absl()


search_space = {
    'is_single': [True, False],
    'use_cpu': [True, False],
}
search_space_list = list(itertools.product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


class PreprocessingTest(parameterized.TestCase):
    @parameterized.parameters(*search_space)
    def test_preprocessing(self, is_single, use_cpu):
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


if __name__ == '__main__':
    absltest.main()
