import jax
import jax.numpy as jnp
import jax.lax as lax

import flax
import flax.linen as nn

from typing import Sequence

import os

import logging

logger = logging.getLogger(__name__)


def save_pretrained_model(return_values: bool = False, clear_session: bool = True):
    os.makedirs('./configs', exist_ok=True)
    os.makedirs('./params', exist_ok=True)
    import tensorflow as tf

    inception = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    params = {}
    for w in inception.weights:
        if w.name.split('/')[0] not in params.keys():
            params[w.name.split('/')[0]] = {}
        params[w.name.split('/')[0]][w.name.split('/')[1]] = jnp.array(w, dtype=jnp.float32)
    jnp.save('./params/inceptionv3.npz', params)

    configs = {}
    for layer in inception.layers:
        configs[layer.name] = layer.get_config()
    jnp.save('./config/inceptionv3.npz', configs)

    del inception
    if clear_session:
        tf.keras.backend.clear_session()
    if return_values:
        return params, configs


def load_model():
    # TODO: Remove params from git and make it manually downloadable from tf.keras
    if os.path.exists('./config/inceptionv3.npz') and os.path.exists('./config/inceptionv3.npz'):
        params = jnp.load('./config/inceptionv3.npz', allow_pickle=True)
        configs = jnp.load('./config/inceptionv3.npz', allow_pickle=True)

    else:
        params, config = save_pretrained_model(True)

    class Conv(nn.Module):
        name: str

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(
                configs[self.name]['filters'], kernel_size=configs[self.name]['kernel_size'],
                strides=configs[self.name]['strides'], padding=configs[self.name]['padding'],
                use_bias=configs[self.name]['use_bias'],
                kernel_init=lambda r, s, d: jnp.array(params[self.name]['kernel:0'], dtype=jnp.float32),
                bias_init=lambda r, s, d: jnp.array(params[self.name]['bias:0'], dtype=jnp.float32),
            )(x)
            return x

    class BatchNorm(nn.Module):
        name: str

        def setup(self):
            c = params[self.name]['moving_mean:0'].shape[0]
            self.mm = self.param(
                'moving_mean', lambda r, s: jnp.array(params[self.name]['moving_mean:0'], dtype=jnp.float32).reshape(s),
                (1, 1, 1, c)
            )
            self.mv = self.param(
                'moving_variance',
                lambda r, s: jnp.array(params[self.name]['moving_variance:0'], dtype=jnp.float32).reshape(s),
                (1, 1, 1, c)
            )
            self.e = self.param(
                'epsilon', lambda r, s: jnp.ones(s, dtype=jnp.float32) * configs[self.name]['epsilon'], (1, 1, 1, c)
            )

            if configs[self.name]['scale']:
                self.use_scale = True
                self.s = self.param(
                    'scale', lambda r, s: jnp.array(params[self.name]['gamma:0'], dtype=jnp.float32), (1, 1, 1, c)
                )
            else:
                self.use_scale = False

            if configs[self.name]['center']:
                self.use_bias = True
                self.b = self.param(
                    'bias', lambda r, s: jnp.array(params[self.name]['beta:0'], dtype=jnp.float32).reshape(s),
                    (1, 1, 1, c)
                )
            else:
                self.use_bais = False

        def __call__(self, x):
            x = x - self.mm
            mul = lax.rsqrt(self.mv + self.e)
            if self.use_scale:
                mul = mul * self.s
            x = x * mul
            if self.use_bias:
                x = x + self.b
            return x


    class ConvBN(nn.Module):
        n: int

        def setup(self):
            conv_name = f'conv2d_{self.n}' if self.n > 0 else 'conv2d'
            bn_name = f'batch_normalization_{self.n}' if self.n > 0 else 'batch_normalization'
            self.conv = Conv(conv_name)
            self.bn = BatchNorm(bn_name)

        def __call__(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = nn.relu(x)
            return x


    class Pool(nn.Module):
        name: str

        def setup(self):
            if 'max' in self.name:
                self.pool = 'max'
            else:
                self.pool = 'avg'
            self.pool_size = configs[self.name]['pool_size']
            self.strides = configs[self.name]['strides']
            self.padding = configs[self.name]['padding']

        def __call__(self, x):
            if self.pool == 'max':
                x = nn.max_pool(x, window_shape=self.pool_size, strides=self.strides, padding=self.padding)
            else:
                x = nn.avg_pool(x, window_shape=self.pool_size, strides=self.strides, padding=self.padding)
            return x

    class InceptionV3(nn.Module):
        @nn.compact
        def __call__(self, x):
            b, h, w, c = x.shape
            if h != 299 or w != 299:
                x = jax.image.resize(x, (b, 299, 299, c), method='bilinear')

            x = ConvBN(0)(x)
            x = ConvBN(1)(x)
            x = ConvBN(2)(x)
            x = Pool('max_pooling2d')(x)

            x = ConvBN(3)(x)
            x = ConvBN(4)(x)
            x = Pool('max_pooling2d_1')(x)

            # Mixed 0: 35 x 35 x 256
            branch1x1 = ConvBN(5)(x)

            branch5x5 = ConvBN(6)(x)
            branch5x5 = ConvBN(7)(branch5x5)

            branch3x3dbl = ConvBN(8)(x)
            branch3x3dbl = ConvBN(9)(branch3x3dbl)
            branch3x3dbl = ConvBN(10)(branch3x3dbl)

            branch_pool = Pool('average_pooling2d')(x)
            branch_pool = ConvBN(11)(branch_pool)
            x = jnp.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)

            # Mixed 1: 35 x 35 x 288
            branch1x1 = ConvBN(12)(x)

            branch5x5 = ConvBN(13)(x)
            branch5x5 = ConvBN(14)(branch5x5)

            branch3x3dbl = ConvBN(15)(x)
            branch3x3dbl = ConvBN(16)(branch3x3dbl)
            branch3x3dbl = ConvBN(17)(branch3x3dbl)

            branch_pool = Pool('average_pooling2d_1')(x)
            branch_pool = ConvBN(18)(branch_pool)
            x = jnp.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)

            # Mixed 2: 35 x 35 x 288
            branch1x1 = ConvBN(19)(x)

            branch5x5 = ConvBN(20)(x)
            branch5x5 = ConvBN(21)(branch5x5)

            branch3x3dbl = ConvBN(22)(x)
            branch3x3dbl = ConvBN(23)(branch3x3dbl)
            branch3x3dbl = ConvBN(24)(branch3x3dbl)

            branch_pool = Pool('average_pooling2d_2')(x)
            branch_pool = ConvBN(25)(branch_pool)
            x = jnp.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)

            # Mixed 3: 17 x 17 x 768
            branch3x3 = ConvBN(26)(x)

            branch3x3dbl = ConvBN(27)(x)
            branch3x3dbl = ConvBN(28)(branch3x3dbl)
            branch3x3dbl = ConvBN(29)(branch3x3dbl)

            branch_pool = Pool('max_pooling2d_2')(x)
            x = jnp.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1)

            # Mixed 4: 17 x 17 x 768
            branch1x1 = ConvBN(30)(x)

            branch7x7 = ConvBN(31)(x)
            branch7x7 = ConvBN(32)(branch7x7)
            branch7x7 = ConvBN(33)(branch7x7)

            branch7x7dbl = ConvBN(34)(x)
            branch7x7dbl = ConvBN(35)(branch7x7dbl)
            branch7x7dbl = ConvBN(36)(branch7x7dbl)
            branch7x7dbl = ConvBN(37)(branch7x7dbl)
            branch7x7dbl = ConvBN(38)(branch7x7dbl)

            branch_pool = Pool('average_pooling2d_3')(x)
            branch_pool = ConvBN(39)(branch_pool)
            x = jnp.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1)

            # Mixed 5: 17 x 17 x 768
            branch1x1 = ConvBN(40)(x)

            branch7x7 = ConvBN(41)(x)
            branch7x7 = ConvBN(42)(branch7x7)
            branch7x7 = ConvBN(43)(branch7x7)

            branch7x7dbl = ConvBN(44)(x)
            branch7x7dbl = ConvBN(45)(branch7x7dbl)
            branch7x7dbl = ConvBN(46)(branch7x7dbl)
            branch7x7dbl = ConvBN(47)(branch7x7dbl)
            branch7x7dbl = ConvBN(48)(branch7x7dbl)

            branch_pool = Pool('average_pooling2d_4')(x)
            branch_pool = ConvBN(49)(branch_pool)
            x = jnp.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1)

            # Mixed 6: 17 x 17 x 768
            branch1x1 = ConvBN(50)(x)

            branch7x7 = ConvBN(51)(x)
            branch7x7 = ConvBN(52)(branch7x7)
            branch7x7 = ConvBN(53)(branch7x7)

            branch7x7dbl = ConvBN(54)(x)
            branch7x7dbl = ConvBN(55)(branch7x7dbl)
            branch7x7dbl = ConvBN(56)(branch7x7dbl)
            branch7x7dbl = ConvBN(57)(branch7x7dbl)
            branch7x7dbl = ConvBN(58)(branch7x7dbl)

            branch_pool = Pool('average_pooling2d_5')(x)
            branch_pool = ConvBN(59)(branch_pool)
            x = jnp.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1)

            # Mixed 7: 17 x 17 x 768
            branch1x1 = ConvBN(60)(x)

            branch7x7 = ConvBN(61)(x)
            branch7x7 = ConvBN(62)(branch7x7)
            branch7x7 = ConvBN(63)(branch7x7)

            branch7x7dbl = ConvBN(64)(x)
            branch7x7dbl = ConvBN(65)(branch7x7dbl)
            branch7x7dbl = ConvBN(66)(branch7x7dbl)
            branch7x7dbl = ConvBN(67)(branch7x7dbl)
            branch7x7dbl = ConvBN(68)(branch7x7dbl)

            branch_pool = Pool('average_pooling2d_6')(x)
            branch_pool = ConvBN(69)(branch_pool)
            x = jnp.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1)

            # Mixed 8: 8 x 8 x 1280
            branch3x3 = ConvBN(70)(x)
            branch3x3 = ConvBN(71)(branch3x3)

            branch7x7x3 = ConvBN(72)(x)
            branch7x7x3 = ConvBN(73)(branch7x7x3)
            branch7x7x3 = ConvBN(74)(branch7x7x3)
            branch7x7x3 = ConvBN(75)(branch7x7x3)

            branch_pool = Pool('max_pooling2d_3')(x)
            x = jnp.concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1)

            # Mixed 9: 8 x 8 x 2048
            branch1x1 = ConvBN(76)(x)

            branch3x3 = ConvBN(77)(x)
            branch3x3_1 = ConvBN(78)(branch3x3)
            branch3x3_2 = ConvBN(79)(branch3x3)
            branch3x3 = jnp.concatenate([branch3x3_1, branch3x3_2], axis=-1)

            branch3x3dbl = ConvBN(80)(x)
            branch3x3dbl = ConvBN(81)(branch3x3dbl)
            branch3x3dbl_1 = ConvBN(82)(branch3x3dbl)
            branch3x3dbl_2 = ConvBN(83)(branch3x3dbl)
            branch3x3dbl = jnp.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=-1)

            branch_pool = Pool('average_pooling2d_7')(x)
            branch_pool = ConvBN(84)(branch_pool)
            x = jnp.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1)

            # Mixed 10: 8 x 8 x 2048
            branch1x1 = ConvBN(85)(x)

            branch3x3 = ConvBN(86)(x)
            branch3x3_1 = ConvBN(87)(branch3x3)
            branch3x3_2 = ConvBN(88)(branch3x3)
            branch3x3 = jnp.concatenate([branch3x3_1, branch3x3_2], axis=-1)

            branch3x3dbl = ConvBN(89)(x)
            branch3x3dbl = ConvBN(90)(branch3x3dbl)
            branch3x3dbl_1 = ConvBN(91)(branch3x3dbl)
            branch3x3dbl_2 = ConvBN(92)(branch3x3dbl)
            branch3x3dbl = jnp.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=-1)

            branch_pool = Pool('average_pooling2d_8')(x)
            branch_pool = ConvBN(93)(branch_pool)
            x = jnp.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1)

            x = jnp.mean(x, axis=(1, 2))
            return x

    module = InceptionV3()
    params = module.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
    module = module.bind(params)
    return module
