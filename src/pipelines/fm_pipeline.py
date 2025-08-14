import torch
from typing import (
    Optional,
    Callable,
    Dict,
)
from ..models import FlowMatchingTransformer

# --------------------------
# 2) Really tiny “scheduler” that drives the reverse process
# --------------------------
class DiffusionSimulation:
    def __init__(
        self,
        model: FlowMatchingTransformer=FlowMatchingTransformer(),
        step_strength: float = 1.0,        # 1.0 == full Euler step
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.h = step_strength             # how aggressively we chase noise_pred
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        shape,
        num_inference_steps: Optional[int] = 28,
        condition: Optional[float] = None,
        callback: Optional[
            Callable[[int, int, torch.Tensor], Optional[Dict[str, torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        """
        Returns the final latents after the full reverse pass.
        """
        self.model.T = num_inference_steps
        # initial white noise -------------------------------------------------
        latents = torch.randn(shape, device=self.device)

        # fixed target (for instrumentation only) ----------------------------
        if condition is None:
            target = torch.zeros_like(latents)           # unconditional
        else:
            target = torch.ones_like(latents) * condition  # conditional

        # reverse loop: T … 0 -----------------------------------------------
        for t in reversed(range(num_inference_steps)):
            noise_pred = self.model(latents, t, condition)

            # one Euler step towards noise_pred
            latents = latents + self.h * (noise_pred - latents) / (t + 1)

            # external callback (OPTIONAL) ----------------------------------
            if callback is not None:
                cb_out = callback(
                    self, 
                    i=t, 
                    t=t / num_inference_steps * 1000, 
                    latents=latents
                )
                # user may return {'latents': new_tensor}
                if isinstance(cb_out, dict):
                    latents = cb_out.get("latents", latents)

            # ----- instrumentation ----------------------------------------
            diff = target - latents
            print(
                f"t={t:04d} | ‖diff‖={diff.norm():8.4f} | "
                f"mean|·|={diff.abs().mean():6.4f} | std={diff.std():6.4f}"
            )

        return latents