import json, os, sys, time
import torch
run, e0, e1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cks = [torch.load(f'{run}/ck_epoch{e:02d}.pt', map_location='cpu', weights_only=False) for e in range(e0, e1 + 1)]
keys = list(cks[-1]['state']); assert all(list(c['state']) == keys for c in cks), 'state keys differ inside the window'
avg = {}
for k in keys:
    t = cks[-1]['state'][k]
    avg[k] = torch.stack([c['state'][k].double() for c in cks]).mean(0).to(t.dtype) if t.is_floating_point() else t.clone()
    if not t.is_floating_point(): assert all(torch.equal(c['state'][k], t) for c in cks), f'non-float tensor {k} differs across epochs'
ck = dict(cks[-1]); ck['state'] = avg; ck['swa'] = dict(run=os.path.abspath(run), epochs=[e0, e1])
os.makedirs(out, exist_ok=True); torch.save(ck, f'{out}/ck_epoch{e1:02d}.pt')
json.dump(dict(run=os.path.abspath(run), epochs=list(range(e0, e1 + 1)), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(f'{out}/SWA.json', 'w'))
print('wrote', f'{out}/ck_epoch{e1:02d}.pt', 'from', len(cks), 'checkpoints')
