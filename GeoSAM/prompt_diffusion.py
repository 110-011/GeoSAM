# === Prompt Diffusion Loss 与 DDIM 推理模块 ===

import torch
import torch.nn.functional as F
from torch import nn
from toolbox.backbone.Segformer.Segformer import mit_b0
import numpy as np

class VID(nn.Module):
	'''
	Variational Information Distillation for Knowledge Transfer
	https://zpascal.net/cvpr2019/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf
	'''
	def __init__(self,out_channels, init_var, eps=1e-6):
		super(VID, self).__init__()
		self.eps = eps
		self.alpha = nn.Parameter(
				np.log(np.exp(init_var-eps)-1.0) * torch.ones(out_channels)
			)

	def forward(self, fm_s, fm_t):
		pred_var  = torch.log(1.0+torch.exp(self.alpha)) + self.eps
		pred_var  = pred_var.view(1, -1, 1, 1)
		neg_log_prob = 0.5 * (torch.log(pred_var) + (fm_s-fm_t)**2 / pred_var)
		loss = torch.mean(neg_log_prob)

		return loss

vidloss = VID(out_channels=6, init_var=5).cuda()

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# 用于时间步编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# 条件调制卷积块（FiLM）
class FiLMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return F.relu(gamma * self.conv(x) + beta)

# 主体扩散模型 Segformer
class PromptDiffusionNet(nn.Module):
    def __init__(self, in_channels=1, cond_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(128),
            nn.Linear(128, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim)
        ).cuda()
        self.encoder = mit_b0()
        self.channels = [32, 64, 160, 256]
        self.FB1 = FiLMBlock(in_channels, 3, cond_dim)
        self.FB2 = FiLMBlock(self.channels[0], self.channels[0], cond_dim)
        self.FB3 = FiLMBlock(self.channels[1], self.channels[1], cond_dim)
        self.FB4 = FiLMBlock(self.channels[2], self.channels[2], cond_dim)

        self.decoder1 = DecoderBlock(in_channels=self.channels[3], out_channels=self.channels[2])
        self.decoder2 = DecoderBlock(in_channels=self.channels[2], out_channels=self.channels[1])
        self.decoder3 = DecoderBlock(in_channels=self.channels[1], out_channels=self.channels[0])
        self.decoder4 = nn.Sequential(DecoderBlock(in_channels=self.channels[0], out_channels=self.channels[0]),
                                      DecoderBlock(in_channels=self.channels[0], out_channels=6))

    def forward(self, x_t, t, cond_feat):
        t_emb = self.time_embed(t)
        cond = cond_feat + t_emb

        h1 = self.FB1(x_t, cond)
        f1 = self.encoder.stage1(h1)
        h2 = self.FB2(f1, cond)
        f2 = self.encoder.stage2(h2)
        h3 = self.FB3(f2, cond)
        f3 = self.encoder.stage3(h3)
        h4 = self.FB4(f3, cond)
        f4 = self.encoder.stage4(h4)

        d1 = self.decoder1(f4)
        d2 = self.decoder2(d1 + f3)
        d3 = self.decoder3(d2 + f2)
        out = self.decoder4(d3 + f1)

        return out


# DDPM MSE 训练损失函数（标准）
def diffusion_loss(prompt_diffusion_model, x_0, t, cond_feat):
    noise = torch.randn_like(x_0)

    betas = torch.linspace(1e-4, 0.02, 1000).to(x_0.device)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # [1000]

    alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

    pred_noise = prompt_diffusion_model(x_t, t, cond_feat)
    return F.mse_loss(pred_noise, noise) + vidloss(pred_noise, noise)

# === DDIM 推理器 ===
class DDIMSampler:
    def __init__(self, model, n_steps=50, eta=0.0):
        self.model = model
        self.n_steps = n_steps
        self.eta = eta
        self.device = next(model.parameters()).device
        self.betas = torch.linspace(1e-4, 0.02, 1000).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample(self, shape, cond_feat):
        B = shape[0]
        x = torch.randn(shape).to(self.device)  # 初始噪声
        t_schedule = torch.linspace(999, 1, self.n_steps, dtype=torch.long).to(self.device)

        for i in range(self.n_steps):
            t = t_schedule[i].expand(B)
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

            with torch.no_grad():
                eps_theta = self.model(x, t, cond_feat)

            if i == self.n_steps - 1:
                x = (x - torch.sqrt(1 - alpha_bar) * eps_theta) / torch.sqrt(alpha_bar)
            else:
                next_t = t_schedule[i + 1]
                alpha_bar_next = self.alpha_bars[next_t].view(-1, 1, 1, 1)

                sigma = self.eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_next)
                x = (torch.sqrt(alpha_bar_next) * ((x - torch.sqrt(1 - alpha_bar) * eps_theta) / torch.sqrt(alpha_bar))) + sigma * torch.randn_like(x)

        return x

# === Prompt 生成器封装类 ===
class PromptGenerator(nn.Module):
    def __init__(self, diffusion_model, ddim_steps=5):
        super().__init__()
        self.diffusion = diffusion_model
        self.ddim = DDIMSampler(diffusion_model, n_steps=ddim_steps)

    def generate_prompt(self, cond_feat, shape=(1, 1, 256, 256)):
        return self.ddim.sample(shape=shape, cond_feat=cond_feat)

    def training_loss(self, x_0, t, cond_feat):
        return diffusion_loss(self.diffusion, x_0, t, cond_feat)
