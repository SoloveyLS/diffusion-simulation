from src.models import FlowMatchingTransformer
from src.pipelines import DiffusionSimulation


# --------------------------
# 3) Quick demo
# --------------------------
if __name__ == "__main__":
    STEPS = 25
    model = FlowMatchingTransformer()
    diffusion = DiffusionSimulation(model)

    # optional example callback – prints every 200 steps
    def my_cb(pipe, i, t, latents):
        if t % 200 == 0:
            print(f"Callback - at global step {i} (t={t})")
        return {}  # could also return {'latents': tweaked_latents}

    print("\n=====  CONDITIONAL RUN  (condition = 2.0)  =====")
    final_latents_cond = diffusion(
        shape=(2, 3), 
        num_inference_steps=STEPS,
        condition=2.0, 
        callback=my_cb
    )

    print("\n=====  UNCONDITIONAL RUN  (condition = None)  =====")
    final_latents_uncond = diffusion(
        shape=(2, 3), 
        num_inference_steps=STEPS,
        condition=None
    )