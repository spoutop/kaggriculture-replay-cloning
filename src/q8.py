import sys, numpy as np
z = np.load(sys.argv[1]); out = {}; n8 = err = 0
for k in z.files:
    a = z[k]
    if not k.startswith('_') and a.dtype.kind == 'f' and a.ndim >= 2:
        w = a.astype(np.float32); m = np.abs(w).reshape(w.shape[0], -1).max(1); s = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(w / s.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8); out[k] = q; out['_q8s__' + k] = s; n8 += 1
        err = max(err, float(np.abs(q.astype(np.float32) * s.reshape((-1,) + (1,) * (w.ndim - 1)) - w).max()))
    else: out[k] = a
np.savez(sys.argv[2], **out); print(sys.argv[2].split('/')[-1], 'int8 arrays', n8, 'max |dequant - src|', f'{err:.2e}')
