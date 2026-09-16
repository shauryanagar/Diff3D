import math
import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1)
        return x + self.act(self.norm2(self.conv2(h)))

class CondPointDiffusionUNet(nn.Module):
    def __init__(self, num_classes=8, time_dim=128, class_dim=128):
        super().__init__()
        cond_dim = time_dim + class_dim
        self.class_emb = nn.Embedding(num_classes, class_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.in_proj = nn.Conv1d(3, 128, 1)
        self.res1 = ResBlock1D(128, cond_dim)
        self.down = nn.Conv1d(128, 256, 1)
        self.res2 = ResBlock1D(256, cond_dim)

        self.global_pool = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.SiLU(),
            nn.Conv1d(512, 256, 1)
        )

        self.up = nn.Conv1d(512, 128, 1)
        self.res3 = ResBlock1D(128, cond_dim)
        self.out_proj = nn.Conv1d(128, 3, 1)

    def forward(self, x_t, t, class_labels):
        half = 64
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x_t.device) / (half - 1))
        args = t[:, None] * freqs[None, :]
        t_emb = self.time_mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))

        c_emb = self.class_emb(class_labels)
        cond = torch.cat([t_emb, c_emb], dim=-1)

        x = x_t.permute(0, 2, 1)
        h1 = self.res1(self.in_proj(x), cond)
        h2 = self.res2(self.down(h1), cond)

        g = torch.max(h2, dim=-1, keepdim=True)[0]
        g = self.global_pool(g).repeat(1, 1, x.shape[-1])

        h3 = self.res3(self.up(torch.cat([h2, g], dim=1)), cond)
        out = self.out_proj(h3 + h1)
        return out.permute(0, 2, 1)