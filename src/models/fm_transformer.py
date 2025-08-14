import torch
import torch.nn as nn
from typing import Optional

# --------------------------
# "Transformer" (really: a tiny deterministic module that
#    builds the noise_pred exactly as you described)
# --------------------------
class FlowMatchingTransformer(nn.Module):
    """
    Pretends to be a transformer.  In reality it just produces
    the hand-crafted noise_pred required for the demo.
    """
    def __init__(
        self,
        total_steps: int = 28,
        base_std_uc: float = 0.20,   # starting std for UN-cond noise
        base_std_c:  float = 0.05,   # starting std for COND   noise
        end_std:     float = 0.01,
    ):
        super().__init__()
        self.T = total_steps
        self.base_std_uc = base_std_uc
        self.base_std_c  = base_std_c
        self.end_std = end_std

    def forward(
        self,
        latents: torch.Tensor,
        t: int,
        condition: Optional[float] = None
    ) -> torch.Tensor:
        """
        Args
        ----
        latents   : current sample (B, …)  - value unused, but kept for API
        t         : current time-step (0 … T)
        condition : scalar that scales the conditioned target
        """
        device = latents.device

        # build both targets ---------------------------------
        # simplest condition ever made
        cond_target   = torch. ones_like(latents, device=device) # all ones
        uncond_target = torch. ones_like(latents, device=device) * latents.mean()
        #torch.zeros_like(latents, device=device) # all zeros

        cond_target = cond_target * (condition or 0)   # scale by prompt strength

        # time-dependent mixing ------------------------------
        # uncond is basically 'the most probable direction' from the model POV
        # with denoising the 'most probable direction' slowly moves towards the condition
        progress = t / self.T
        w_uc = progress          # weight for un-cond → large at start
        w_c  = 1.0 - progress    # weight for cond    → grows towards t==0

        # time-dependent Gaussian noise ----------------------
        # there is always an error in the prediction
        std_uc = self.base_std_uc * progress + self.end_std # vanishes as t→0
        std_c  = self.base_std_c  * progress + self.end_std

        noise_uc = torch.randn_like(latents) * std_uc
        noise_c  = torch.randn_like(latents) * std_c

        pred_uc = uncond_target + noise_uc
        pred_c  = cond_target   + noise_c

        # final linear blend
        noise_pred = w_uc * pred_uc + w_c * pred_c
        return noise_pred