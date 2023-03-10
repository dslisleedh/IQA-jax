# IQA-jax
Image Quality Assessment library for jax. Converts the Numpy implementations from BasicSR into Jax.Numpy.

## NOTE
<b>Current implementations are not tested. There's no sure the outputs are same with BasicSR(MATLAB)</b>  
Jax only supports cudnn 8.2 or 8.6 but BasicSR currently support cudnn 8.5. Be aware version issues.  

## Metrics 
 - [X] PSNR
 - [X] SSIM
 - [ ] NIQE
 - [X] FID

## TODO
 - [ ] Testing Codes(Compare Numpy(BasicSR), Jax.Numpy(CPU), Jax.Numpy(GPU))  ~Maybe Pytorch CPU?~
 - [X] InceptionV3 Using flax.
