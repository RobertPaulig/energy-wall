import torch, torch.nn as nn, torch.nn.functional as F
from .physics import PHYSICS_CONFIG

class AnalogLinear(nn.Module):
    def __init__(self, in_features, out_features, physics_type, bias=False, seed=123):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.physics = PHYSICS_CONFIG[physics_type]
        self.accumulate_energy = True
        self.energy_fJ = 0.0
        g = torch.Generator().manual_seed(seed + in_features + out_features)
        self.register_buffer('rtn_mask', (torch.rand(out_features, in_features, generator=g) < self.physics['rtn_fraction']).float())
        self.register_buffer('rtn_state', torch.zeros(out_features, in_features))
        self.p01 = 1.0 / max(1, self.physics['rtn_tau_off'])
        self.p10 = 1.0 / max(1, self.physics['rtn_tau_on'])

    def forward(self, x):
        w = self.linear.weight
        noise = torch.randn_like(w) * self.physics['read_noise_std'] if self.physics['read_noise_std']>0 else 0.0
        if self.rtn_mask.sum()>0:
            rnd = torch.rand_like(w)
            to_on = (self.rtn_state < 0.5) & (rnd < self.p01)
            to_off = (self.rtn_state > 0.5) & (rnd < self.p10)
            self.rtn_state = torch.where(to_on, torch.ones_like(w), self.rtn_state)
            self.rtn_state = torch.where(to_off, torch.zeros_like(w), self.rtn_state)
            rtn = self.rtn_mask * (self.rtn_state * 2.0 - 1.0) * self.physics['rtn_amp']
        else:
            rtn = 0.0
        w_eff = w + noise + rtn
        y = F.linear(x, w_eff, None)
        batch, in_f = x.shape[0], x.shape[1]
        out_f = y.shape[1]
        n_mac = batch * in_f * out_f
        e = n_mac * self.physics['e_fJ_mxv'] + (batch*in_f) * self.physics['e_fJ_wire'] + out_f * self.physics['e_fJ_adc']
        if self.accumulate_energy:
            self.energy_fJ += e
        return y