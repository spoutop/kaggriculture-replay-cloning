import torch, torch.nn as nn, torch.nn.functional as F
import codec as C, encoder as E

class Res(nn.Module):
    def __init__(self, ch, gdim):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch); self.n2 = nn.GroupNorm(8, ch); self.film = nn.Linear(gdim, 2 * ch)

    def forward(self, x, g):
        h = self.c1(F.gelu(self.n1(x), approximate='tanh'))
        a, b = self.film(g).chunk(2, dim=-1)
        h = self.n2(h) * (1 + a[..., None, None]) + b[..., None, None]
        return x + self.c2(F.gelu(h, approximate='tanh'))

N_TIME = 24 + 4 + 3

def time_feats(glob):
    hour = torch.round(glob[:, 2] * 24).long().clamp(0, 23); day = torch.round(glob[:, 1] * 30); step = glob[:, 0] * 720
    return torch.cat(
        [F.one_hot(hour, 24).to(glob.dtype), F.one_hot(hour % 4, 4).to(glob.dtype), (day >= 28).to(glob.dtype)[:, None], (day >= 29).to(glob.dtype)[:, None],
         (1 - (720 - step) / 48).clamp(0, 1)[:, None]], dim=-1)

N_LEAD = 4

def lead_feats(glob):
    own = torch.expm1(glob[:, 5].float() * 12.0); opp = torch.expm1(glob[:, 11].float() * 12.0); lead = own - opp
    return torch.stack([torch.tanh(lead / 3000.0), torch.tanh(lead / 12000.0), (lead / (own + opp + 1000.0)).clamp(-1, 1), torch.tanh((own + opp) / 200000.0)], dim=-1).to(glob.dtype)

class Net(nn.Module):
    def __init__(self, ch=96, gdim=256, blocks=6, n_teams=16, gextra=0, ghist=0, glead=0, vpriv=0, uhist=0, uattn=0, gplan=0):
        super().__init__()
        self.ch, self.gdim, self.gextra, self.ghist, self.glead, self.vpriv = ch, gdim, int(gextra), int(ghist), int(glead), int(vpriv)

        self.uhist, self.uattn, self.gplan = int(uhist), int(uattn), int(gplan)
        if self.uhist:
            self.uh1 = nn.Embedding(C.N_UNIT + 1, 64); self.uh2 = nn.Embedding(C.N_UNIT + 1, 64); nn.init.zeros_(self.uh1.weight); nn.init.zeros_(self.uh2.weight)
        if self.uattn:
            self.ua_q = nn.Linear(ch + 64, 64); self.ua_k = nn.Linear(ch + 64, 64); self.ua_v = nn.Linear(ch + 64, 64); self.ua_o = nn.Linear(64, 64); nn.init.zeros_(self.ua_o.weight); nn.init.zeros_(self.ua_o.bias)
        self.temb = nn.Embedding(n_teams, gdim)
        self.g0 = nn.Sequential(nn.Linear(E.F_GLOB + int(gextra) + int(ghist) + int(glead) + int(gplan), gdim), nn.GELU(approximate='tanh'), nn.Linear(gdim, gdim), nn.GELU(approximate='tanh'))
        self.stem = nn.Conv2d(E.C_TILE + 2, ch, 3, padding=1)
        self.blocks = nn.ModuleList([Res(ch, gdim) for _ in range(blocks)])
        self.ostem = nn.Sequential(nn.Conv2d(E.C_TILE, ch // 2, 3, padding=1), nn.GELU(approximate='tanh'), nn.Conv2d(ch // 2, ch // 2, 3, padding=1), nn.GELU(approximate='tanh'))
        self.g1 = nn.Sequential(nn.Linear(gdim + 2 * ch + ch, gdim), nn.GELU(approximate='tanh'), nn.Linear(gdim, gdim), nn.GELU(approximate='tanh'))
        self.uenc = nn.Sequential(nn.Linear(E.F_UNIT + 1, 64), nn.GELU(approximate='tanh'))
        self.uhead = nn.Sequential(nn.Linear(ch + 64 + gdim, 384), nn.GELU(approximate='tanh'), nn.Linear(384, 256), nn.GELU(approximate='tanh'), nn.Linear(256, C.N_UNIT))
        self.slot = nn.Embedding(C.MAX_SLOTS, 32); self.kemb = nn.Embedding(C.N_MK, 32)
        self.khead = nn.Sequential(nn.Linear(gdim + 32, 256), nn.GELU(approximate='tanh'), nn.Linear(256, C.N_MK))
        self.qhead = nn.Sequential(nn.Linear(gdim + 64, 256), nn.GELU(approximate='tanh'), nn.Linear(256, C.MAX_Q))
        self.vhead = nn.Sequential(nn.Linear(gdim + int(vpriv), 256),
                                    nn.GELU(approximate='tanh'), nn.Linear(256, 1))
        coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), indexing='ij'))
        self.register_buffer('coords', coords)

    def trunk(self, grid, glob, team=None, ghist=None, gplan=None):
        B = grid.shape[0]; base = glob
        if self.gextra: glob = torch.cat([glob, time_feats(base)], dim=-1)
        if self.glead: glob = torch.cat([glob, lead_feats(base)], dim=-1)
        if self.ghist: glob = torch.cat([glob, ghist.to(glob.dtype) if ghist is not None else glob.new_zeros(B, self.ghist)], dim=-1)
        if self.gplan: glob = torch.cat([glob, gplan.to(glob.dtype) if gplan is not None else glob.new_zeros(B, self.gplan)], dim=-1)
        g = self.g0(glob) + (self.temb.weight[0][None] if team is None else self.temb(team))
        x = self.stem(torch.cat([grid[:, 0], self.coords.expand(B, -1, -1, -1)], dim=1))
        for blk in self.blocks:
            x = blk(x, g)
        o = self.ostem(grid[:, 1])

        pooled = torch.cat([x.mean((2, 3)), x.flatten(2).max(-1).values, o.mean((2, 3)), o.flatten(2).max(-1).values], dim=-1)
        g = self.g1(torch.cat([g, pooled], dim=-1))
        return x, g

    def unit_logits(self, fmap, g, units, xy, umask, uh=None):
        B, U = xy.shape[:2]
        idx = (xy[..., 1] * 10 + xy[..., 0]).long()
        feat = fmap.flatten(2).gather(2, idx[:, None, :].expand(-1, fmap.shape[1], -1)).transpose(1, 2)
        order = torch.arange(U, device=xy.device, dtype=units.dtype)[None, :, None].expand(B, -1, -1) / U
        ue = self.uenc(torch.cat([units, order], dim=-1))
        if self.uhist:
            if uh is None: uh = torch.full((B, U, 2), C.N_UNIT, dtype=torch.long, device=xy.device)
            ue = ue + self.uh1(uh[..., 0]) + self.uh2(uh[..., 1])
        if self.uattn:
            tok = torch.cat([feat, ue], dim=-1); pres = umask.any(-1)
            att = (self.ua_q(tok) @ self.ua_k(tok).transpose(1, 2)) / 8.0
            att = torch.softmax(att.masked_fill(~pres[:, None, :], -1e9).float(), -1).to(tok.dtype)
            ue = ue + self.ua_o(att @ self.ua_v(tok))
        lg = self.uhead(torch.cat([feat, ue, g[:, None, :].expand(-1, U, -1)], dim=-1))
        return lg.masked_fill(~umask, -1e9)

    def kind_logits(self, g, mmask):
        B = g.shape[0]
        se = self.slot.weight[None].expand(B, -1, -1)
        lg = self.khead(torch.cat([g[:, None, :].expand(-1, C.MAX_SLOTS, -1), se], dim=-1))
        return lg.masked_fill(~mmask[:, None, :], -1e9)

    def qty_logits(self, g, kinds, qmax):
        B = g.shape[0]
        se = self.slot.weight[None].expand(B, -1, -1)
        lg = self.qhead(torch.cat([g[:, None, :].expand(-1, C.MAX_SLOTS, -1), se, self.kemb(kinds)], dim=-1))
        qm = qmax.gather(1, kinds).clamp(min=1)
        ar = torch.arange(C.MAX_Q, device=g.device)[None, None, :]
        return lg.masked_fill(ar >= qm[..., None], -1e9)

    def value(self, g, priv=None):
        if self.vpriv: g = torch.cat([g, priv.to(g.dtype) if priv is not None else g.new_zeros(g.shape[0], self.vpriv)], dim=-1)
        return self.vhead(g).squeeze(-1)

    def forward(self, grid, units, xy, glob, umask, mmask, qmax, kinds, team=None, ghist=None, uh=None, gplan=None):
        fmap, g = self.trunk(grid, glob, team, ghist, gplan)
        return self.unit_logits(fmap, g, units, xy, umask, uh), self.kind_logits(g, mmask), self.qty_logits(g, kinds, qmax), self.value(g)
