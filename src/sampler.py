import torch
import numpy as np

def abar(t):
    return torch.cos((t + 0.008) / 1.008 * (np.pi / 2)) ** 2

@torch.no_grad()
def sample_ddim(model, class_idx, steps=50, num_points=2048, device="cpu"):
    model.eval()
    x = torch.randn(1, num_points, 3, device=device)
    c = torch.tensor([class_idx], device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, steps, device=device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i].unsqueeze(0)
        t_next = timesteps[i + 1].unsqueeze(0)

        alpha_t = abar(t)
        alpha_next = abar(t_next)

        pred_noise = model(x, t, c)
        x0_hat = (x - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x0_hat = torch.clamp(x0_hat, -1.1, 1.1)

        if i == len(timesteps) - 2:
            return x0_hat.squeeze(0).cpu().numpy()

        dir_xt = torch.sqrt(1.0 - alpha_next) * pred_noise
        x = torch.sqrt(alpha_next) * x0_hat + dir_xt

    return x.squeeze(0).cpu().numpy()