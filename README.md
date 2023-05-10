# IQA-jax
Image Quality Assessment library for Jax.  
Implementations are Jax.numpy ported versions of the original Numpy-based [BasicSR](https://github.com/XPixelGroup/BasicSR).  

## NOTE
<b>Not all functions have been tested. Functions marked as tested below ensure that their output is consistent with BasicSR (MATLAB).</b>  

## HOW TO USE  

```bash
pip install iqa-jax
```

## Example
```python
from iqa.metrics import psnr

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

inputs_1 = jnp.array(np.random.randint(0., 256., size=(16, 256, 256, 3), dtype=np.uint8))
inputs_2 = jnp.array(np.random.randint(0., 256., size=(16, 256, 256, 3), dtype=np.uint8))

metric = jax.jit(partial(psnr, crop_border=0, test_y=False))
psnr_val = metric(inputs_1, inputs_2)
```


## Metrics
 - [X] PSNR
 - [X] SSIM
 - [X] NIQE
 - [X] FID

## Tests
 - [X] PSNR
 - [X] SSIM
 - [X] NIQE
 - [ ] FID
 - [ ] InceptionV3
 - [X] RGB2Y Conversion
 - [X] RGB2Gray Conversion
 - [X] MATLAB's .5 scale bicubic resize
