import collections, math
import numpy as np

def paired_bar(A, B, label='bar', headroom=(0.05, 0.95), z=1.96):
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    d = np.array([A[k] - B[k] for k in keys], float)
    seeds = [k[0] for k in keys]
    by = collections.defaultdict(list)
    for x, s in zip(d, seeds):
        by[s].append(x)
    m = np.array([np.mean(v) for v in by.values()])
    k = len(m)
    est = d.mean()
    se = m.std(ddof=1) / math.sqrt(k) if k > 1 else float('inf')
    se_iid = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float('inf')
    mbar = len(d) / k
    gm = d.mean()
    between = sum(len(v) * (np.mean(v) - gm) ** 2 for v in by.values())
    within = sum(float(np.sum((np.array(v) - np.mean(v)) ** 2)) for v in by.values())
    msb, msw = between / max(1, k - 1), within / max(1, len(d) - k)
    rho = (msb - msw) / (msb + (mbar - 1) * msw) if (msb + (mbar - 1) * msw) else 0.0
    lo, hi = est - z * se, est + z * se
    return dict(label=label, n=len(d), seeds=k, per_seed=mbar, est=est, se=se, se_iid=se_iid,
                inflation=se / se_iid if se_iid else float('nan'), rho=rho, lo=lo, hi=hi,
                verdict='RESOLVED +' if lo > 0 else ('RESOLVED -' if hi < 0 else 'not resolved'),
                p_disagree=float(np.mean(d != 0)))

def fmt(r):
    return (f"  {r['label']:<28s} {r['est']:+.4f}  SE {r['se']:.4f} (iid {r['se_iid']:.4f}, x{r['inflation']:.2f}; "
            f"rho {r['rho']:.3f})  95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['verdict']}   "
            f"n={r['n']} seeds={r['seeds']} disagree={r['p_disagree']:.3f}")

def headroom_keys(B, lo=0.05, hi=0.95):
    by = collections.defaultdict(list)
    for (s, seat, opp), v in B.items():
        by[opp].append(v)
    return {o for o, v in by.items() if lo < float(np.mean(v)) < hi}
