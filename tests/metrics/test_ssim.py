import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized, absltest

import cv2
from iqa.metrics import ssim, SSIM
from iqa.metrics.ssim import _get_2d_gaussian_kernel
from basicsr.metrics.psnr_ssim import calculate_ssim

from functools import partial
from itertools import product


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


class TestSSIM(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSSIM, self).__init__(*args, **kwargs)

        self.inputs1 = np.random.randint(0., 256., size=(16, 256, 256, 3), dtype=np.uint8)
        self.inputs2 = np.random.randint(0., 256., size=(16, 256, 256, 3), dtype=np.uint8)

    @parameterized.parameters(
        {'kernel_size': 11, 'sigma': 1.5},
        {'kernel_size': 15, 'sigma': 1.},
        {'kernel_size': 17, 'sigma': .75},
        {'kernel_size': 9, 'sigma': 2.},
    )
    def test_gaussian_kernel(self, kernel_size, sigma):
        cv2_gauss = cv2.getGaussianKernel(kernel_size, sigma)
        cv2_kernel = cv2_gauss @ cv2_gauss.T
        jax_kernel = _get_2d_gaussian_kernel(kernel_size, sigma).squeeze()
        np.testing.assert_allclose(cv2_kernel, jax_kernel, atol=1e-7)

    @parameterized.parameters(*search_space)
    def test_result(self, crop_border, test_y, is_single_input, use_class, use_gpu):
        device = jax.devices('gpu' if use_gpu else 'cpu')[0]
        if is_single_input:
            inputs1 = self.inputs1[0].copy()
            inputs2 = self.inputs2[0].copy()
        else:
            inputs1 = self.inputs1.copy()
            inputs2 = self.inputs2.copy()

        if use_class:
            metric_ssim = SSIM(crop_border=crop_border, test_y=test_y)
            ssim_call_func = jax.jit(metric_ssim.__call__)
        else:
            metric_ssim = partial(ssim, crop_border=crop_border, test_y=test_y)
            ssim_call_func = jax.jit(metric_ssim)

        if is_single_input:  # BasicSR uses BGR2YCbCr to get Y channel. So I reversed the channel.
            bsr_ssim = calculate_ssim(
                inputs1[..., ::-1], inputs2[..., ::-1], crop_border=crop_border, test_y_channel=test_y)
        else:
            bsr_ssim = []
            for i in range(inputs1.shape[0]):
                bsr_ssim.append(
                    calculate_ssim(
                        inputs1[i][..., ::-1], inputs2[i][..., ::-1], crop_border=crop_border, test_y_channel=test_y))
            bsr_ssim = np.stack(bsr_ssim, axis=0)

        inputs1 = jax.device_put(jnp.array(inputs1, dtype=jnp.uint8), device=device)
        inputs2 = jax.device_put(jnp.array(inputs2, dtype=jnp.uint8), device=device)
        jax_ssim = np.array(ssim_call_func(inputs1, inputs2))

        np.testing.assert_allclose(bsr_ssim, jax_ssim, rtol=1e-3, atol=1e-7)


if __name__ == '__main__':
    absltest.main()
