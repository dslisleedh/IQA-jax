import logging
import os

import flax.core
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp

from typing import Sequence, Literal
import pickle

import warnings


# TODO: Fix this
logger = logging.getLogger("iqa")


def _load_params(params_in):
    os.makedirs('~/iqa_jax/params', exist_ok=True)
    os.remove('~/iqa_jax/params/inception_v3.sav')

    if os.path.exists('~/iqa_jax/params/inception_v3.sav'):
        logger.info('Files already exist. Loading from disk.')
        params_bytes = pickle.load(open('~/iqa_jax/params/inception_v3.sav', 'rb'))
        params = flax.core.unfreeze(flax.serialization.from_bytes(params_in, params_bytes))

    else:
        logger.info('Files do not exist. Downloading from tf.keras.')
        warnings.warn('Clearing session. This may cause issues if you are using tf.keras elsewhere.')

        import tensorflow as tf
        tf.keras.backend.clear_session()

        inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        params_tf = {}
        for w in inception.weights:
            if w.name.split('/')[0] not in params_tf.keys():
                params_tf[w.name.split('/')[0]] = {}
            params_tf[w.name.split('/')[0]][w.name.split('/')[1]] = jnp.array(w, dtype=jnp.float32)

        del inception
        tf.keras.backend.clear_session()

        # Overwrite the weights with the ones from the pretrained model
        # Params
        for p in params_in['params'].keys():
            if 'conv2d' in p:
                params_in['params'][p]['kernel'] = jnp.asarray(params_tf[p]['kernel:0'], dtype=jnp.float32)
            elif 'batch_normalization' in p:
                params_in['params'][p]['bias'] = jnp.asarray(params_tf[p]['beta:0'], dtype=jnp.float32)

        # Variables
        for p in params_in['batch_stats'].keys():
            params_in['batch_stats'][p]['mean'] = jnp.asarray(params_tf[p]['moving_mean:0'], dtype=jnp.float32)
            params_in['batch_stats'][p]['var'] = jnp.asarray(params_tf[p]['moving_variance:0'], dtype=jnp.float32)

        params_bytes = flax.serialization.to_bytes(params_in)
        pickle.dump(params_bytes, open('~/iqa_jax/params/inception_v3.sav', 'wb'))
        params = flax.core.unfreeze(params_in)

    return params


def load_model():
    def conv_bn_act(
            x, filters: int, kernel_size: Sequence[int], n: int, strides: Sequence[int] = (1, 1), padding: str = 'SAME'):
        x = nn.Conv(
            features=filters, kernel_size=kernel_size, strides=strides, dtype=jnp.float32,
            padding=padding, use_bias=False, name=f'conv2d_{n}' if n != 0 else 'conv2d')(x)
        x = nn.BatchNorm(
            epsilon=1e-3, use_scale=False, dtype=jnp.float32,
            name=f'batch_normalization_{n}' if n != 0 else 'batch_normalization')(x, use_running_average=True)
        x = nn.relu(x)
        return x

    class InceptionV3(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = x.astype(jnp.float32)
            b, h, w, c = x.shape
            if h != 299 or w != 299:
                x = jax.image.resize(x, (b, 299, 299, c), method='bilinear')

            x = conv_bn_act(x, 32, (3, 3), 0, strides=(2, 2), padding='VALID')
            x = conv_bn_act(x, 32, (3, 3), 1, padding='VALID')
            x = conv_bn_act(x, 64, (3, 3), 2)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')

            x = conv_bn_act(x, 80, (1, 1), 3, padding='VALID')
            x = conv_bn_act(x, 192, (3, 3), 4, padding='VALID')
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')

            # Mixed 0: 35 x 35 x 256
            branch_1x1 = conv_bn_act(x, 64, (1, 1), 5)

            branch_5x5 = conv_bn_act(x, 48, (1, 1), 6)
            branch_5x5 = conv_bn_act(branch_5x5, 64, (5, 5), 7)

            branch_3x3dbl = conv_bn_act(x, 64, (1, 1), 8)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 9)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 10)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 32, (1, 1), 11)
            x = jnp.concatenate([branch_1x1, branch_5x5, branch_3x3dbl, branch_pool], axis=-1)

            # Mixed 1: 35 x 35 x 288
            branch_1x1 = conv_bn_act(x, 64, (1, 1), 12)

            branch_5x5 = conv_bn_act(x, 48, (1, 1), 13)
            branch_5x5 = conv_bn_act(branch_5x5, 64, (5, 5), 14)

            branch_3x3dbl = conv_bn_act(x, 64, (1, 1), 15)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 16)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 17)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 64, (1, 1), 18)
            x = jnp.concatenate([branch_1x1, branch_5x5, branch_3x3dbl, branch_pool], axis=-1)

            # Mixed 2: 35 x 35 x 288
            branch_1x1 = conv_bn_act(x, 64, (1, 1), 19)

            branch_5x5 = conv_bn_act(x, 48, (1, 1), 20)
            branch_5x5 = conv_bn_act(branch_5x5, 64, (5, 5), 21)

            branch_3x3dbl = conv_bn_act(x, 64, (1, 1), 22)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 23)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 24)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 64, (1, 1), 25)
            x = jnp.concatenate([branch_1x1, branch_5x5, branch_3x3dbl, branch_pool], axis=-1)

            # Mixed 3: 17 x 17 x 768
            branch_3x3 = conv_bn_act(x, 384, (3, 3), 26, strides=(2, 2), padding='VALID')

            branch_3x3dbl = conv_bn_act(x, 64, (1, 1), 27)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 28)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 96, (3, 3), 29, strides=(2, 2), padding='VALID')

            branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')
            x = jnp.concatenate([branch_3x3, branch_3x3dbl, branch_pool], axis=-1)

            # Mixed 4: 17 x 17 x 768
            branch_1x1 = conv_bn_act(x, 192, (1, 1), 30)

            branch_7x7 = conv_bn_act(x, 128, (1, 1), 31)
            branch_7x7 = conv_bn_act(branch_7x7, 128, (1, 7), 32)
            branch_7x7 = conv_bn_act(branch_7x7, 192, (7, 1), 33)

            branch_7x7dbl = conv_bn_act(x, 128, (1, 1), 34)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 128, (7, 1), 35)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 128, (1, 7), 36)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 128, (7, 1), 37)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (1, 7), 38)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 192, (1, 1), 39)
            x = jnp.concatenate([branch_1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=-1)

            # Mixed 5: 17 x 17 x 768
            branch_1x1 = conv_bn_act(x, 192, (1, 1), 40)

            branch_7x7 = conv_bn_act(x, 160, (1, 1), 41)
            branch_7x7 = conv_bn_act(branch_7x7, 160, (1, 7), 42)
            branch_7x7 = conv_bn_act(branch_7x7, 192, (7, 1), 43)

            branch_7x7dbl = conv_bn_act(x, 160, (1, 1), 44)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 160, (7, 1), 45)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 160, (1, 7), 46)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 160, (7, 1), 47)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (1, 7), 48)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 192, (1, 1), 49)
            x = jnp.concatenate([branch_1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=-1)

            # Mixed 6: 17 x 17 x 768
            branch_1x1 = conv_bn_act(x, 192, (1, 1), 50)

            branch_7x7 = conv_bn_act(x, 160, (1, 1), 51)
            branch_7x7 = conv_bn_act(branch_7x7, 160, (1, 7), 52)
            branch_7x7 = conv_bn_act(branch_7x7, 192, (7, 1), 53)

            branch_7x7dbl = conv_bn_act(x, 160, (1, 1), 54)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 160, (7, 1), 55)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 160, (1, 7), 56)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 160, (7, 1), 57)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (1, 7), 58)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 192, (1, 1), 59)
            x = jnp.concatenate([branch_1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=-1)

            # Mixed 7: 17 x 17 x 768
            branch_1x1 = conv_bn_act(x, 192, (1, 1), 60)

            branch_7x7 = conv_bn_act(x, 192, (1, 1), 61)
            branch_7x7 = conv_bn_act(branch_7x7, 192, (1, 7), 62)
            branch_7x7 = conv_bn_act(branch_7x7, 192, (7, 1), 63)

            branch_7x7dbl = conv_bn_act(x, 192, (1, 1), 64)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (7, 1), 65)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (1, 7), 66)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (7, 1), 67)
            branch_7x7dbl = conv_bn_act(branch_7x7dbl, 192, (1, 7), 68)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 192, (1, 1), 69)
            x = jnp.concatenate([branch_1x1, branch_7x7, branch_7x7dbl, branch_pool], axis=-1)

            # Mixed 8: 8 x 8 x 1280
            branch_3x3 = conv_bn_act(x, 192, (1, 1), 70)
            branch_3x3 = conv_bn_act(branch_3x3, 320, (3, 3), 71, strides=(2, 2), padding='VALID')

            branch_7x7x3 = conv_bn_act(x, 192, (1, 1), 72)
            branch_7x7x3 = conv_bn_act(branch_7x7x3, 192, (1, 7), 73)
            branch_7x7x3 = conv_bn_act(branch_7x7x3, 192, (7, 1), 74)
            branch_7x7x3 = conv_bn_act(branch_7x7x3, 192, (3, 3), 75, strides=(2, 2), padding='VALID')

            branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')
            x = jnp.concatenate([branch_3x3, branch_7x7x3, branch_pool], axis=-1)

            # Mixed 9: 8 x 8 x 2048
            branch_1x1 = conv_bn_act(x, 320, (1, 1), 76)

            branch_3x3 = conv_bn_act(x, 384, (1, 1), 77)
            branch_3x3_1 = conv_bn_act(branch_3x3, 384, (1, 3), 78)
            branch_3x3_2 = conv_bn_act(branch_3x3, 384, (3, 1), 79)
            branch_3x3 = jnp.concatenate([branch_3x3_1, branch_3x3_2], axis=-1)

            branch_3x3dbl = conv_bn_act(x, 448, (1, 1), 80)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 384, (3, 3), 81)
            branch_3x3dbl_1 = conv_bn_act(branch_3x3dbl, 384, (1, 3), 82)
            branch_3x3dbl_2 = conv_bn_act(branch_3x3dbl, 384, (3, 1), 83)
            branch_3x3dbl = jnp.concatenate([branch_3x3dbl_1, branch_3x3dbl_2], axis=-1)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 192, (1, 1), 84)
            x = jnp.concatenate([branch_1x1, branch_3x3, branch_3x3dbl, branch_pool], axis=-1)

            # Mixed 10: 8 x 8 x 2048
            branch_1x1 = conv_bn_act(x, 320, (1, 1), 85)

            branch_3x3 = conv_bn_act(x, 384, (1, 1), 86)
            branch_3x3_1 = conv_bn_act(branch_3x3, 384, (1, 3), 87)
            branch_3x3_2 = conv_bn_act(branch_3x3, 384, (3, 1), 88)
            branch_3x3 = jnp.concatenate([branch_3x3_1, branch_3x3_2], axis=-1)

            branch_3x3dbl = conv_bn_act(x, 448, (1, 1), 89)
            branch_3x3dbl = conv_bn_act(branch_3x3dbl, 384, (3, 3), 90)
            branch_3x3dbl_1 = conv_bn_act(branch_3x3dbl, 384, (1, 3), 91)
            branch_3x3dbl_2 = conv_bn_act(branch_3x3dbl, 384, (3, 1), 92)
            branch_3x3dbl = jnp.concatenate([branch_3x3dbl_1, branch_3x3dbl_2], axis=-1)

            branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding='SAME')
            branch_pool = conv_bn_act(branch_pool, 192, (1, 1), 93)
            x = jnp.concatenate([branch_1x1, branch_3x3, branch_3x3dbl, branch_pool], axis=-1)

            return x

    module = InceptionV3()
    params_init = flax.core.unfreeze(module.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3))))
    params = _load_params(params_init)
    return module, params
