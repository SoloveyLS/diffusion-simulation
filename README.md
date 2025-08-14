There are a lot of cases when I need to test some diffusion ideas.
This is a simple way to do so with a laptop without GPU.

### Current state:
- Flow Matching approach:
  - straight trajectories (noise added)
  - equidistant scheduler
- initial distribution of `N(0,1)`
- target distribution of `cond * torch.ones((B,C,H,W))`
- model 'predicts' a velocity:
  - velocity is a linear combination of the 'unconditioned' (`cond = 0`) and 'conditioned' velocities
  - the 'unconditioned' prediction shifts towards the 'conditioned' as the trajectory slowly drift towards it
- velocity is predicted with noise:
  - term with linear decay 
    (the closer to the target distribution, the more accurate the 'prediction' is)
  - term with no decay
    (there is yet a small amount of noise in the final prediction)
- callback that works with the latents in the same way as the `diffusers`

### TODO:
- [ ] Make it working properly
- [ ] Make it completely integrated into `diffusers` interface:
  - [ ] scheduler (required)
  - [ ] transformer / unet (required)
  - [ ] callback interface (required)
  - [ ] VAE (unnecessary, `vae = lambda x: x` is sufficient enough)
  - [ ] condition encoder (unnecessary, `text_encoder = lambda x: x` is sufficient enough)
- [ ] Add the trajectory visualization:
  - [ ] distance from the target point for this `t` 
    - along the ideal trajectory     (                             won't be even close to the actual diffusion)
    - normal to the ideal trajectory (CFG and noising values should be set to be close to the actual diffusion)
  - [ ] distance from the ideal trajectory itself
- [ ] Add Score-Matching / DDIM simulations?