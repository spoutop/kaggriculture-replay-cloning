import torch
import codec as C

OLD_WHEAT = (1, 2, 3, 4, 6, 8, 12, 16); OLD_FERT = (1, 2, 3, 4, 6, 8)
V2_WHEAT = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16); V2_FERT = (1, 2, 3, 4, 5, 6, 7, 8); V2_PLACE = (1, 2, 3, 4, 6, 8)

def old_layout(n_unit):
    ua = [['PASS'], ['NORTH'], ['SOUTH'], ['EAST'], ['WEST'], ['WATER'], ['HARVEST'], ['FERTILIZE'], ['DIG'],
          ['BUILD_COOP'], ['BUILD_PASTURE'], ['FEED'], ['CARE'], ['COLLECT_FERTILIZER'], ['DROP']]
    v2 = n_unit == 111
    ua += [['PLANT', c] for c in C.CROPS]
    ua += [['PICKUP', 'WHEAT', q] for q in (V2_WHEAT if v2 else OLD_WHEAT)]
    ua += [['PICKUP', 'FERTILIZER', q] for q in (V2_FERT if v2 else OLD_FERT)]
    ua += [['PICKUP', a, 1] for a in C.ANIMALS] + [['PICKUP', a, 2] for a in C.ANIMALS]
    ua += [['PLACE', a, 1] for a in C.ANIMALS]
    ua += [['PLACE', p, 99] for p in C.PRODUCTS]
    if v2: ua += [['PLACE', p, q] for p in C.PRODUCTS for q in V2_PLACE]
    assert len(ua) == n_unit, (len(ua), n_unit)
    return ua

def source_row(new, old_index):
    key = tuple(new)
    if key in old_index: return old_index[key], True
    op = new[0]
    if op == 'PICKUP':
        it, q = new[1], new[2]
        cands = [(k, i) for k, i in old_index.items() if len(k) == 3 and k[0] == 'PICKUP' and k[1] == it]
        if cands:
            if q >= 99: return max(cands, key=lambda t: t[0][2])[1], False
            return min(cands, key=lambda t: (abs(t[0][2] - q), t[0][2]))[1], False
        return None, False
    if op == 'PLACE':
        it = new[1]
        if ('PLACE', it, 99) in old_index: return old_index[('PLACE', it, 99)], False
    raise KeyError(f'no source for {new}')

def adapt_state(st, bias_new=-2.0, bias_q=-3.0, verbose=True):
    W = st['uhead.4.weight']; b = st['uhead.4.bias']; n_old = W.shape[0]
    if n_old != C.N_UNIT:
        old = old_layout(n_old); oi = {tuple(a): i for i, a in enumerate(old)}
        pick_rows = [i for a, i in oi.items() if a[0] == 'PICKUP']
        W2 = torch.zeros(C.N_UNIT, W.shape[1], dtype=W.dtype, device=W.device); b2 = torch.zeros(C.N_UNIT, dtype=b.dtype, device=b.device)
        n_exact = n_new = 0
        for j, a in enumerate(C.UNIT_ACTIONS):
            i, exact = source_row(a, oi)
            if i is None:
                W2[j] = W[pick_rows].mean(0); b2[j] = b[pick_rows].mean() + 2 * bias_new; n_new += 1
            elif exact:
                W2[j] = W[i]; b2[j] = b[i]; n_exact += 1
            else:
                W2[j] = W[i]; b2[j] = b[i] + bias_new; n_new += 1
        st['uhead.4.weight'] = W2; st['uhead.4.bias'] = b2
        if verbose: print(f'vocab_adapt: unit head {n_old} -> {C.N_UNIT} ({n_exact} rows copied by name, {n_new} new rows from relatives, bias {bias_new})')
    Wq = st['qhead.2.weight']; bq = st['qhead.2.bias']; q_old = Wq.shape[0]
    if q_old != C.MAX_Q:
        assert q_old < C.MAX_Q
        Wq2 = torch.cat([Wq, Wq[-1:].expand(C.MAX_Q - q_old, -1)], 0); bq2 = torch.cat([bq, bq[-1:].expand(C.MAX_Q - q_old) + bias_q], 0)
        st['qhead.2.weight'] = Wq2.clone(); st['qhead.2.bias'] = bq2.clone()
        if verbose: print(f'vocab_adapt: quantity head {q_old} -> {C.MAX_Q} (new rows = row {q_old - 1}, bias {bias_q})')
    return st
