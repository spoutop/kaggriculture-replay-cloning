import argparse, glob, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import codec as C
from model import Net

def load(files):
    keys = ['grid', 'units', 'xy', 'n_units', 'glob', 'umask', 'mmask', 'qmax', 'a_unit', 'a_kind', 'a_q']
    parts = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys: parts[k].append(d[k])
    return {k: np.concatenate(v) for k, v in parts.items()}

def batch(D, idx, dev):
    g = lambda k, dt: torch.as_tensor(D[k][idx]).to(dev, dtype=dt, non_blocking=True)
    b = dict(grid=g('grid', torch.float32), units=g('units', torch.float32), xy=g('xy', torch.int64), glob=g('glob', torch.float32), umask=g('umask', torch.bool),
             mmask=g('mmask', torch.bool), qmax=g('qmax', torch.int64), a_unit=g('a_unit', torch.int64), a_kind=g('a_kind', torch.int64), a_q=g('a_q', torch.int64))
    qm = b['qmax'].gather(1, b['a_kind']).clamp(min=1)
    b['a_q'] = torch.minimum(b['a_q'], qm - 1)
    return b

def losses(net, b):
    ul, kl, ql, v = net(b['grid'], b['units'], b['xy'], b['glob'], b['umask'], b['mmask'], b['qmax'], b['a_kind'])
    present = b['a_unit'] >= 0
    lu = F.cross_entropy(ul[present], b['a_unit'][present])
    lk = F.cross_entropy(kl.flatten(0, 1), b['a_kind'].flatten())
    hasq = (b['a_kind'] != C.MK_NONE) & (b['a_kind'] != C.MK_HIRE) & (b['a_kind'] != C.MK_LAND)
    lq = F.cross_entropy(ql[hasq], b['a_q'][hasq]) if hasq.any() else ul.sum() * 0
    with torch.no_grad():
        au = (ul[present].argmax(-1) == b['a_unit'][present]).float().mean()
        nonpass = present & (b['a_unit'] != C.A_PASS)
        aunp = (ul[nonpass].argmax(-1) == b['a_unit'][nonpass]).float().mean()
        ak = (kl.argmax(-1) == b['a_kind']).float().mean()
        nn_ = b['a_kind'] != C.MK_NONE
        aknn = (kl.argmax(-1)[nn_] == b['a_kind'][nn_]).float().mean()
        aq = (ql[hasq].argmax(-1) == b['a_q'][hasq]).float().mean() if hasq.any() else au * 0
    return lu, lk, lq, dict(acc_unit=au.item(), acc_unit_nonpass=aunp.item(), acc_kind=ak.item(), acc_kind_nonnone=aknn.item(), acc_q=aq.item())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=512); ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--device', default='cuda'); ap.add_argument('--ch', type=int, default=96); ap.add_argument('--blocks', type=int, default=6)
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    files = sorted(glob.glob(f'{a.data}/*.npz')); val = files[::10]; tr = [f for f in files if f not in val]
    Dt, Dv = load(tr), load(val)
    n = len(Dt['glob']); print(f'train {len(tr)} games {n:,} states | val {len(val)} games {len(Dv["glob"]):,} states | device {dev}', flush=True)
    net = Net(ch=a.ch, blocks=a.blocks).to(dev); print(f'params {sum(p.numel() for p in net.parameters())/1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_total = a.epochs * (n // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps_total, pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log = open(out / 'log.jsonl', 'a'); best = 1e9
    for ep in range(a.epochs):
        net.train(); perm = np.random.permutation(n); t0 = time.time(); agg = {}
        for i in range(n // a.bs):
            b = batch(Dt, np.sort(perm[i * a.bs:(i + 1) * a.bs]), dev)
            lu, lk, lq, m = losses(net, b); loss = lu + lk + 0.5 * lq
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            for k, v in dict(m, lu=lu.item(), lk=lk.item(), lq=lq.item()).items(): agg[k] = agg.get(k, 0) + v
        tr_m = {k: v / (n // a.bs) for k, v in agg.items()}
        net.eval(); agg = {}; nb = 0
        with torch.no_grad():
            for i in range(0, len(Dv['glob']), 1024):
                b = batch(Dv, np.arange(i, min(i + 1024, len(Dv['glob']))), dev); lu, lk, lq, m = losses(net, b); nb += 1
                for k, v in dict(m, lu=lu.item(), lk=lk.item(), lq=lq.item()).items(): agg[k] = agg.get(k, 0) + v
        va = {k: v / nb for k, v in agg.items()}; vloss = va['lu'] + va['lk'] + 0.5 * va['lq']
        row = dict(epoch=ep, secs=round(time.time() - t0, 1), train=tr_m, val=va, val_loss=vloss); log.write(json.dumps(row) + '\n'); log.flush()
        print(f"ep {ep:2d} {row['secs']:5.0f}s | train lu {tr_m['lu']:.3f} lk {tr_m['lk']:.3f} lq {tr_m['lq']:.3f} | val lu {va['lu']:.3f} lk {va['lk']:.3f} lq {va['lq']:.3f} | val acc unit {va['acc_unit']:.3f} (non-pass {va['acc_unit_nonpass']:.3f}) kind {va['acc_kind']:.3f} (non-none {va['acc_kind_nonnone']:.3f}) q {va['acc_q']:.3f}", flush=True)
        torch.save({'state': net.state_dict(), 'ch': a.ch, 'blocks': a.blocks, 'epoch': ep, 'val': va}, out / 'last.pt')
        if vloss < best:
            best = vloss; torch.save({'state': net.state_dict(), 'ch': a.ch, 'blocks': a.blocks, 'epoch': ep, 'val': va}, out / 'best.pt')

if __name__ == '__main__':
    main()
