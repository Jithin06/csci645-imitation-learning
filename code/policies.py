"""Student policy for PandaPickCube: ACT (Action Chunking with Transformers), state-based, in PyTorch.

Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware".
Adapted from github.com/tonyzhaozh/act (detr/models/detr_vae.py): same CVAE structure
(transformer encoder -> z; transformer encoder-decoder with learned queries -> action chunk),
L1 + beta*KL loss, z = 0 at test time. Changes: the ResNet image backbone is removed and the
66-D state is a single input token.

Interface (used by train_student.py / evaluate.py):
  policy.compute_loss(batch) -> (loss, logs)      batch from train_student.DemoDataset
  policy.predict_chunk(obs_hist) -> (B, H, 8)     normalised obs history in, actions out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ACTPolicy(nn.Module):

    def __init__(self, obs_dim=66, act_dim=8, chunk=20, d_model=256, nhead=8,
                 enc_layers=4, dec_layers=4, ff=1024, dropout=0.1, latent=32, kl_weight=10.0):
        super().__init__()
        self.chunk, self.latent, self.kl_weight = chunk, latent, kl_weight
        self.obs_horizon = 1
        layer = lambda: nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        # --- CVAE encoder q(z | obs, action chunk): only used during training
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.enc_obs = nn.Linear(obs_dim, d_model)
        self.enc_act = nn.Linear(act_dim, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, chunk + 2, d_model) * 0.02)
        self.vae_encoder = nn.TransformerEncoder(layer(), enc_layers)
        self.to_latent = nn.Linear(d_model, 2 * latent)
        # --- decoder p(action chunk | obs, z)
        self.dec_z = nn.Linear(latent, d_model)
        self.dec_obs = nn.Linear(obs_dim, d_model)
        self.mem_pos = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.encoder = nn.TransformerEncoder(layer(), enc_layers)
        self.queries = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.head = nn.Linear(d_model, act_dim)

    def _decode(self, obs, z):
        mem = torch.stack([self.dec_z(z), self.dec_obs(obs)], dim=1) + self.mem_pos
        mem = self.encoder(mem)
        q = self.queries.expand(obs.shape[0], -1, -1)
        return self.head(self.decoder(q, mem))

    def compute_loss(self, batch):
        obs = batch["obs"][:, -1]                       # (B, 66)
        act, pad = batch["actions"], batch["is_pad"]    # (B, K, 8), (B, K)
        B = obs.shape[0]
        tokens = torch.cat([self.cls.expand(B, -1, -1), self.enc_obs(obs)[:, None], self.enc_act(act)], 1)
        key_pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=obs.device), pad], 1)
        h = self.vae_encoder(tokens + self.enc_pos, src_key_padding_mask=key_pad)[:, 0]
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        pred = self._decode(obs, z)
        l1 = (F.l1_loss(pred, act, reduction="none").mean(-1) * ~pad).sum() / (~pad).sum()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        loss = l1 + self.kl_weight * kl
        return loss, {"l1": l1.item(), "kl": kl.item()}

    @torch.no_grad()
    def predict_chunk(self, obs_hist):
        obs = obs_hist[:, -1]
        z = torch.zeros(obs.shape[0], self.latent, device=obs.device)
        return self._decode(obs, z)

def build_policy(algo, cfg):
    if algo == "act":
        return ACTPolicy(chunk=cfg["chunk"], kl_weight=cfg["kl_weight"])
    raise ValueError(algo)

DEFAULTS = {
    "act": dict(chunk=20, kl_weight=10.0, obs_horizon=1, action_horizon=1,
                lr=1e-4, weight_decay=1e-4, batch_size=256, steps=40000, ema=0.0),
}