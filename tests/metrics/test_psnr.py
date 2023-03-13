from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized, absltest

from iqa.metrics.psnr import psnr, PSNR

jax.config.update('jax_enable_x64', True)
jax.config.parse_flags_with_absl()

search_space = {
    'crop_border': [0, 4, 8],
    'test_y': [True, False],
    'is_single_input': [True, False],
    'use_class': [True, False],
    'use_gpu': [True, False]
}
search_space_list = list(product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


class TestPSNR(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPSNR, self).__init__(*args, **kwargs)

        self.inputs1 = np.random.randint(0., 256., size=(16, 256, 256, 3), dtype=np.uint8)
        self.inputs2 = np.random.randint(0., 256., size=(16, 256, 256, 3), dtype=np.uint8)

    @parameterized.parameters(*search_space)
    def eval_result(self, crop_border, test_y, is_single_input, use_class, use_gpu):
        device = jax.devices('gpu' if use_gpu else 'cpu')[0]
        if is_single_input:
            inputs1 = self.inputs1[0].copy()
            inputs2 = self.inputs2[0].copy()
        else:
            inputs1 = self.inputs1.copy()
            inputs2 = self.inputs2.copy()

        if use_class:
            metric_psnr = PSNR(crop_border=crop_border, test_y=test_y)
            psnr_call_func = jax.jit(metric_psnr.__call__, device=device)
        else:
            metric_psnr = partial(psnr, crop_border=crop_border, test_y=test_y)
            psnr_call_func = jax.jit(metric_psnr, device=device)

        if is_single_input:
            bsr_psnr = calculate_psnr(inputs1, inputs2, crop_border=crop_border, test_y_channel=test_y)
        else:
            bsr_psnr = []
            for i in range(inputs1.shape[0]):
                bsr_psnr.append(
                    calculate_psnr(inputs1[i], inputs2[i], crop_border=crop_border, test_y_channel=test_y))
            bsr_psnr = np.stack(bsr_psnr, axis=0)

        inputs1 = jnp.array(inputs1, dtype=jnp.uint8)
        inputs2 = jnp.array(inputs2, dtype=jnp.uint8)
        jax_psnr = np.array(psnr_call_func(inputs1, inputs2))

        np.testing.assert_allclose(bsr_psnr, jax_psnr, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    absltest.main()
