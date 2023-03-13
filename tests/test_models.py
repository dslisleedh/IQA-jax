import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from iqa.models import inceptionv3

from absl.testing import absltest
from tqdm import tqdm


jax.config.update('jax_enable_x64', True)
jax.config.parse_flags_with_absl()


class InceptionTest(absltest.TestCase):
    def test_weights(self) -> None:
        _, params_flax = inceptionv3.load_model()
        tf.keras.backend.clear_session()
        inception_tf = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')

        params_tf = {}
        for w in inception_tf.weights:
            if w.name.split('/')[0] not in params_tf.keys():
                params_tf[w.name.split('/')[0]] = {}
            params_tf[w.name.split('/')[0]][w.name.split('/')[1]] = jnp.array(w, dtype=jnp.float32)

        # Params
        for p in tqdm(params_flax['params'].keys(), desc='Params'):
            if 'conv2d' in p:
                p_flax = params_flax['params'][p]['kernel']
                p_tf = params_tf[p]['kernel:0']
            elif 'batch_normalization' in p:
                p_flax = params_flax['params'][p]['bias']
                p_tf = params_tf[p]['beta:0']
            else:
                raise ValueError('Unknown param: {}'.format(p))
            np.testing.assert_allclose(p_flax, p_tf)

        # Variables
        for p in tqdm(params_flax['batch_stats'].keys(), desc='Variables'):
            p_flax = params_flax['batch_stats'][p]['mean']
            p_tf = params_tf[p]['moving_mean:0']
            np.testing.assert_allclose(p_flax, p_tf)

            p_flax = params_flax['batch_stats'][p]['var']
            p_tf = params_tf[p]['moving_variance:0']
            np.testing.assert_allclose(p_flax, p_tf)

    def test_outputs(self) -> None:
        inception_flax, params_flax = inceptionv3.load_model()
        tf.keras.backend.clear_session()
        inception_tf = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        x = np.ones((1, 299, 299, 3), dtype=np.float32)
        y_flax = inception_flax.apply(params_flax, jnp.array(x, jnp.float32))
        y_tf = inception_tf(tf.convert_to_tensor(x, dtype=tf.float32))
        np.testing.assert_allclose(y_flax, y_tf, atol=1e-3)


if __name__ == '__main__':
    absltest.main()
