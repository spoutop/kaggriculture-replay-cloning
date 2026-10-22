import sys, numpy as np
KEEP = ('stem.', 'uhead.', 'khead.', 'qhead.', 'vhead.')
z = np.load(sys.argv[1]); q8 = {}; dq = {}; nq = nk = 0
for k in z.files:
    a = z[k]
    if k.startswith('_') or a.dtype.kind != 'f' or a.ndim < 2 or k.startswith(KEEP):
        q8[k] = a; dq[k] = a if k.startswith('_') else a.astype(np.float32); nk += (a.ndim >= 2 and not k.startswith('_')); continue
    w = a.astype(np.float32); m = np.abs(w).reshape(w.shape[0], -1).max(1); s = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    sb = s.reshape((-1,) + (1,) * (w.ndim - 1)); q = np.clip(np.rint(w / sb), -127, 127).astype(np.int8)
    q8[k] = q; q8['_q8s__' + k] = s; dq[k] = q.astype(np.float32) * sb; nq += 1
np.savez(sys.argv[2], **q8)
if len(sys.argv) > 3: np.savez(sys.argv[3], **dq)
print(sys.argv[2].split('/')[-1], 'int8', nq, 'kept 2-D', nk)
