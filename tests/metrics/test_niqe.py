import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized, absltest

from iqa.metrics import niqe as niqe_module_jax
from iqa.metrics.niqe import (
    _estimate_aggd_param as estimate_aggd_param_jax,
    _compute_feature as compute_feature_jax,
    _calculate_niqe as niqe_jax,
    niqe as calculate_niqe_jax
)
from basicsr.metrics import niqe as niqe_module
from basicsr.metrics.niqe import compute_feature, estimate_aggd_param, niqe, calculate_niqe

from functools import partial
from itertools import product
from tqdm import tqdm
import pickle


search_space = {
    'crop_border': [0, 4, 8],
    'use_gpu': [True, False],
    'channel': ['y', 'gray']
}
search_space_list = list(product(*search_space.values()))
search_space = [dict(zip(search_space.keys(), v)) for v in search_space_list]


class TestNIQE(parameterized.TestCase):
    def test_1_estimate_aggd_param(self):
        for _ in tqdm(range(100), desc='Testing estimate_aggd_param'):
            block = np.random.randint(0, 256, (96, 96, 1))
            block = block.astype(np.float32)
            block_jax = jnp.array(block)

            res = np.array(estimate_aggd_param(block))
            res_jax = np.array(jax.jit(estimate_aggd_param_jax)(block_jax))

            np.testing.assert_allclose(res, res_jax, rtol=1e-3, atol=1e-7)

    def test_2_compute_features_from_blocks(self):
        for _ in tqdm(range(100), desc='Testing compute_features_from_blocks'):
            block = np.random.randint(0, 256, (96, 96, 1))
            block = block.astype(np.float32)
            block_jax = jnp.array(block)

            res = np.array(compute_feature(block))
            res_jax = np.array(jax.jit(compute_feature_jax)(block_jax))

            np.testing.assert_allclose(res, res_jax, rtol=1e-3, atol=1e-7)

    def test_3_compare_pris_params(self):
        module_path = niqe_module.__file__
        module_path = '/'.join(module_path.split('/')[:-1]) + '/niqe_pris_params.npz'
        niqe_pris_params = np.load(module_path)
        pris_params = {
            'mu_pris_param': niqe_pris_params['mu_pris_param'],
            'cov_pris_param': niqe_pris_params['cov_pris_param'],
            'gaussian_window': niqe_pris_params['gaussian_window']
        }

        module_path_jax = niqe_module_jax.__file__
        module_path_jax = '/'.join(module_path_jax.split('/')[:-1]) + '/niqe_pris_params.pkl'
        with open(module_path_jax, 'rb') as f:
            pris_params_jax = pickle.load(f)

        for k, v in pris_params.items():
            np.testing.assert_allclose(v, pris_params_jax[k], rtol=1e-3, atol=1e-7)

    def test_4_compare_niqe(self):
        module_path = niqe_module.__file__
        module_path = '/'.join(module_path.split('/')[:-1]) + '/niqe_pris_params.npz'
        niqe_pris_params = np.load(module_path)
        mu_pris_param = niqe_pris_params['mu_pris_param']
        cov_pris_param = niqe_pris_params['cov_pris_param']
        gaussian_window = niqe_pris_params['gaussian_window']

        module_path_jax = niqe_module_jax.__file__
        module_path_jax = '/'.join(module_path_jax.split('/')[:-1]) + '/niqe_pris_params.pkl'
        with open(module_path_jax, 'rb') as f:
            pris_params_jax = pickle.load(f)
        mu_pris_param_jax = pris_params_jax['mu_pris_param']
        cov_pris_param_jax = pris_params_jax['cov_pris_param']
        gaussian_window_jax = pris_params_jax['gaussian_window']

        for _ in tqdm(range(100), desc='Testing calculate_niqe'):
            inputs = np.random.randint(0, 256, (96*4, 96*4, 1))
            inputs_bsr = inputs.astype(np.float32)[..., 0]
            inputs_jax = jnp.array(inputs, dtype=jnp.float64)[jnp.newaxis, ...]

            res = niqe(inputs_bsr, mu_pris_param, cov_pris_param, gaussian_window)
            res_jax = jax.jit(niqe_jax)(
                inputs_jax, mu_pris_param_jax, cov_pris_param_jax, gaussian_window_jax)

            np.testing.assert_allclose(res, res_jax, rtol=1e-3, atol=1e-7)

    # @parameterized.parameters(*search_space)
    # def test_5_niqe_wrapper(self, channel, crop_border, use_gpu):
    #     device = jax.devices('gpu' if use_gpu else 'cpu')[0]
    #     for _ in tqdm(range(10), desc='Testing niqe_wrapper'):
    #         inputs = np.random.randint(0, 256, (96*4, 96*4, 3))
    #         inputs_bsr = inputs.astype(np.float32)[..., ::-1]
    #         inputs_jax = jnp.array(inputs, dtype=jnp.float64)[jnp.newaxis, ...]
    #         inputs_jax = jax.device_put(inputs_jax, device)
    #
    #         res = calculate_niqe(inputs_bsr, convert_to=channel, crop_border=crop_border)
    #         func = partial(calculate_niqe_jax, convert_to=channel, crop_border=crop_border)
    #         func = jax.jit(func)
    #         res_jax = func(inputs_jax)
    #
    #         np.testing.assert_allclose(res, res_jax, rtol=1e-3, atol=1e-7)
    #

if __name__ == '__main__':
    absltest.main()
