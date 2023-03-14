import jax
import warnings
from . import metrics
from . import models
from . import utils


jax.config.update('jax_enable_x64', True)
warnings.warn(
    "This library enables jax_enable_x64 by default for precision. "
    "Be aware that other other jax libraries may not be compatible with this."
)
