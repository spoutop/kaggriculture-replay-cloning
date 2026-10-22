import numpy as np

CODEC_TAG = 'trackA_v3'
BOARD = 10
ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
CROPS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON']
ANIMALS = ['GOOSE', 'COW', 'SHEEP']
PRODUCTS = ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER']
ITEMS = PRODUCTS + ANIMALS
PRODUCE7 = [p for p in PRODUCTS if p not in ('WHEAT', 'FERTILIZER')]
CROP_FIRST_YIELD = {'WHEAT': 2, 'CARROT': 2, 'TOMATO': 8, 'STRAWBERRY': 10, 'MELON': 10}
SEED_COST = {'WHEAT': 10, 'CARROT': 20, 'TOMATO': 50, 'STRAWBERRY': 100, 'MELON': 80}
ANIMAL_COST = {'GOOSE': 300, 'COW': 400, 'SHEEP': 500}
STRUCTURE = {'GOOSE': 'COOP', 'COW': 'PASTURE', 'SHEEP': 'PASTURE'}
LAND_PRICES = [1000, 2000, 4000]
VOCAB2 = True
FLIPMASK = True
_env = __import__('os').environ
if _env.get('RL1_VOCAB2', '1') != '1' or _env.get('RL1_FLIPMASK', '1') != '1':
    raise RuntimeError('trackA codec v3 has VOCAB2 and FLIPMASK built in; RL1_VOCAB2/RL1_FLIPMASK=0 in the environment is a configuration error')
PICK_WHEAT_Q = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16)
PICK_FERT_Q = (1, 2, 3, 4, 5, 6, 7, 8)
PLACE_PART_Q = (1, 2, 3, 4, 6, 8)

PICK_WHEAT_X = (9, 11, 13, 14, 15, 20, 24)
PICK_FERT_X = (9, 10, 12, 16)
PLACE_PART_X = (5, 7)
ALL_Q = 99

UNIT_ACTIONS = [['PASS'], ['NORTH'], ['SOUTH'], ['EAST'], ['WEST'], ['WATER'], ['HARVEST'], ['FERTILIZE'], ['DIG'],
                ['BUILD_COOP'], ['BUILD_PASTURE'], ['FEED'], ['CARE'], ['COLLECT_FERTILIZER'], ['DROP']]
A_PLANT0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PLANT', c] for c in CROPS]
A_PKW0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', 'WHEAT', q] for q in PICK_WHEAT_Q]
A_PKF0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', 'FERTILIZER', q] for q in PICK_FERT_Q]
A_PKA0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', a, 1] for a in ANIMALS] + [['PICKUP', a, 2] for a in ANIMALS]
A_PLA0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PLACE', a, 1] for a in ANIMALS]
A_PLP0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PLACE', p, 99] for p in PRODUCTS]
A_PLQ0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PLACE', p, q] for p in PRODUCTS for q in PLACE_PART_Q]
N_UNIT_VOCAB2 = len(UNIT_ACTIONS)
A_PKALL0 = len(UNIT_ACTIONS); PICK_ALL_ITEMS = ['WHEAT', 'FERTILIZER'] + ANIMALS; UNIT_ACTIONS += [['PICKUP', it, ALL_Q] for it in PICK_ALL_ITEMS]
A_PKP0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', p, q] for p in PRODUCE7 for q in (1, ALL_Q)]
A_PKWX0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', 'WHEAT', q] for q in PICK_WHEAT_X]
A_PKFX0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', 'FERTILIZER', q] for q in PICK_FERT_X]
A_PLQX0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PLACE', p, q] for p in PRODUCTS for q in PLACE_PART_X]
PICK_PRODUCE_X = (2, 3, 4)
A_PKPX0 = len(UNIT_ACTIONS); UNIT_ACTIONS += [['PICKUP', p, q] for p in PRODUCE7 for q in PICK_PRODUCE_X]
N_UNIT = len(UNIT_ACTIONS)
A_PASS, A_N, A_S, A_E, A_W, A_WATER, A_HARVEST, A_FERT, A_DIG, A_COOP, A_PASTURE, A_FEED, A_CARE, A_COLLECT, A_DROP = range(15)
MOVES = {A_N: (0, -1), A_S: (0, 1), A_E: (1, 0), A_W: (-1, 0)}
MAX_UNITS = 16

_EXACT = {tuple(a): i for i, a in enumerate(UNIT_ACTIONS) if len(a) == 3}

MK_KINDS = [None] + [('SELL', p) for p in PRODUCTS] + [('BUY_PRODUCT', 'WHEAT'), ('BUY_PRODUCT', 'FERTILIZER')] + \
           [('BUY_SEED', c) for c in CROPS] + [('BUY_ANIMAL', a) for a in ANIMALS] + [('HIRE',), ('BUY_LAND',)]
N_MK = len(MK_KINDS)
MK_INDEX = {k: i for i, k in enumerate(MK_KINDS)}
MK_NONE, MK_HIRE, MK_LAND = 0, MK_INDEX[('HIRE',)], MK_INDEX[('BUY_LAND',)]
MAX_SLOTS = 10
MAX_Q = 100

def _nearest(q, classes):
    return min(range(len(classes)), key=lambda i: (abs(classes[i] - q), classes[i]))

def encode_unit(a):
    if not isinstance(a, list) or not a:
        return A_PASS, a in (None, [], ['PASS'])
    op = a[0]
    simple = {'PASS': A_PASS, 'NORTH': A_N, 'SOUTH': A_S, 'EAST': A_E, 'WEST': A_W, 'WATER': A_WATER, 'HARVEST': A_HARVEST,
              'FERTILIZE': A_FERT, 'DIG': A_DIG, 'BUILD_COOP': A_COOP, 'BUILD_PASTURE': A_PASTURE, 'FEED': A_FEED, 'CARE': A_CARE,
              'COLLECT_FERTILIZER': A_COLLECT, 'DROP': A_DROP}
    if op in simple:
        return simple[op], True
    if op == 'PLANT' and len(a) > 1 and a[1] in CROPS:
        return A_PLANT0 + CROPS.index(a[1]), True
    if op in ('PICKUP', 'PLACE') and len(a) > 1 and a[1] in ITEMS:
        try:
            q = int(a[2]) if len(a) > 2 else 1
        except (TypeError, ValueError):
            q = 1
        it = a[1]
        if op == 'PICKUP':
            if q >= ALL_Q and it in PICK_ALL_ITEMS + PRODUCE7:
                return _EXACT[('PICKUP', it, ALL_Q)], True
            i = _EXACT.get(('PICKUP', it, q))
            if i is not None:
                return i, True
            if it == 'WHEAT':
                qs = PICK_WHEAT_Q + PICK_WHEAT_X; j = _nearest(q, qs); return _EXACT[('PICKUP', it, qs[j])], False
            if it == 'FERTILIZER':
                qs = PICK_FERT_Q + PICK_FERT_X; j = _nearest(q, qs); return _EXACT[('PICKUP', it, qs[j])], False
            if it in ANIMALS:
                return _EXACT[('PICKUP', it, 2 if q >= 2 else 1)], False
            qs = (1,) + PICK_PRODUCE_X; j = _nearest(q, qs); return _EXACT[('PICKUP', it, qs[j])], False
        if it in ANIMALS:
            return A_PLA0 + ANIMALS.index(it), q == 1 or q >= ALL_Q
        if q < ALL_Q:
            i = _EXACT.get(('PLACE', it, q))
            if i is not None:
                return i, True
            qs = PLACE_PART_Q + PLACE_PART_X; j = _nearest(q, qs); return _EXACT[('PLACE', it, qs[j])], False
        return A_PLP0 + PRODUCTS.index(it), True
    return A_PASS, False

def decode_unit(i):
    return list(UNIT_ACTIONS[int(i)])

def encode_order(o):
    if not isinstance(o, list) or not o:
        return (MK_NONE, 1), False
    if o[0] == 'HIRE':
        return (MK_HIRE, 1), True
    if o[0] == 'BUY_LAND':
        return (MK_LAND, 1), True
    if len(o) >= 3 and (o[0], o[1]) in MK_INDEX:
        try:
            q = int(o[2])
        except (TypeError, ValueError):
            return (MK_NONE, 1), False
        if q <= 0:
            return (MK_NONE, 1), False
        return (MK_INDEX[(o[0], o[1])], min(q, MAX_Q)), q <= MAX_Q
    return (MK_NONE, 1), False

def decode_order(kind, q):
    k = MK_KINDS[int(kind)]
    if k is None:
        return None
    if len(k) == 1:
        return [k[0]]
    return [k[0], k[1], int(q)]

def encode_action(action):
    action = action if isinstance(action, dict) else {}
    units = [action.get('farmer') or ['PASS']] + list(action.get('hands') or [])
    exact = True; u = []
    for a in units:
        i, ok = encode_unit(a); u.append(i); exact &= ok
    slots = []
    for o in (action.get('market') or [])[:MAX_SLOTS]:
        s, ok = encode_order(o); slots.append(s); exact &= ok
    return u, slots, exact

def decode_action(unit_idx, slots, n_hands=None):
    units = [decode_unit(i) for i in unit_idx]
    if n_hands is not None:
        left = MAX_UNITS - 1 - int(n_hands); kept = []
        for k, q in slots:
            if k == MK_HIRE:
                if left <= 0: kept.append((MK_NONE, 1)); continue
                left -= 1
            kept.append((k, q))
        slots = kept

    market = [decode_order(k, q) for k, q in slots]
    while market and market[-1] is None: market.pop()
    market = [[] if m is None else m for m in market]
    return {'farmer': units[0] if units else ['PASS'], 'hands': units[1:], 'market': market}

MARKET_DEPTH = {'WHEAT': 400, 'CARROT': 450, 'TOMATO': 200, 'STRAWBERRY': 100, 'MELON': 300, 'EGG': 332, 'MILK': 122, 'WOOL': 105, 'FERTILIZER': 200}

def overflow_sale(obs, action, shed_capacity=100, keep_wheat=12):
    step = int(obs.get('step', 0))
    if step % 24 != 23: return action
    me = int(obs['player']); priv = obs['private']; shed = dict(priv['shed']); market = list(action.get('market') or [])
    cargo = sum(int(v) for inv in priv.get('inventories', []) for v in inv.values())
    selling = {}
    for m in market:
        if m and m[0] == 'SELL': selling[m[1]] = selling.get(m[1], 0) + int(m[2])
    excess = sum(int(v) for v in shed.values()) + cargo - sum(min(q, int(shed.get(k, 0))) for k, q in selling.items()) - shed_capacity
    free = MAX_SLOTS - len(market)
    if excess <= 0 or free <= 0: return action
    for p_ in sorted(PRODUCTS, key=lambda k: -MARKET_DEPTH[k]):
        have = int(shed.get(p_, 0)) - selling.get(p_, 0) - (keep_wheat if p_ == 'WHEAT' else 0)
        while have > 0 and excess > 0 and free > 0:
            q = min(have, excess, MAX_Q); market.append(['SELL', p_, q]); have -= q; excess -= q; free -= 1
    out = dict(action); out['market'] = market
    return out

def fix_plant_budget(unit_idx, seeds):
    left = {c: int(seeds.get(c, 0)) for c in CROPS}
    out = list(unit_idx)
    for n, i in enumerate(out):
        if A_PLANT0 <= i < A_PLANT0 + len(CROPS):
            c = CROPS[i - A_PLANT0]
            if left[c] > 0:
                left[c] -= 1
            else:
                out[n] = A_PASS
    return out

def fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def unit_mask(farm, private, idx, day, shed_capacity=100):
    m = _unit_mask_base(farm, private, idx, day, shed_capacity)
    up = [farm['farmer']] + list(farm['hands']); x, y = int(up[idx][0]), int(up[idx][1]); tile = farm['tiles'][y][x]
    if tile == 'LOCKED': return m
    invs = private['inventories']; inv = invs[idx] if idx < len(invs) else {}
    if isinstance(tile, dict) and tile.get('kind') == 'PLANT' and inv.get('FERTILIZER', 0) > 0: m[A_FERT] = True
    if any((int(up[j][0]), int(up[j][1])) == (x, y) for j in range(idx)) and not (isinstance(tile, dict) and 'animal' in tile):
        m[A_DIG] = True; m[A_WATER] = True; m[A_HARVEST] = True
        if inv.get('FERTILIZER', 0) > 0: m[A_FERT] = True
        seeds = private.get('seeds', {})
        for k, c in enumerate(CROPS):
            if seeds.get(c, 0) > 0: m[A_PLANT0 + k] = True
    return m

def _unit_mask_base(farm, private, idx, day, shed_capacity=100):
    m = np.zeros(N_UNIT, dtype=bool)
    m[A_PASS] = True
    pos = farm['farmer'] if idx == 0 else farm['hands'][idx - 1]
    x, y = int(pos[0]), int(pos[1])
    if y > 0: m[A_N] = True
    if y < BOARD - 1: m[A_S] = True
    if x < BOARD - 1: m[A_E] = True
    if x > 0: m[A_W] = True
    invs = private['inventories']
    inv = invs[idx] if idx < len(invs) else {}
    shed_real = private['shed']; shed = shed_real; lower_at_access = False
    if (x, y) in ACCESS:

        shed = dict(shed_real); units_pos = [farm['farmer']] + list(farm['hands'])
        for j in range(min(idx, len(invs))):
            if (int(units_pos[j][0]), int(units_pos[j][1])) in ACCESS:
                lower_at_access = True
                for it, v in invs[j].items(): shed[it] = shed.get(it, 0) + int(v)

    room_real = shed_capacity - sum(int(v) for v in shed_real.values())
    if room_real <= 0 and lower_at_access and sum(int(v) for v in shed_real.values()) > 0: room_real = 1
    if (x, y) in ACCESS:
        room = room_real
        if any(v > 0 for v in inv.values()):
            m[A_DROP] = True
        if shed.get('WHEAT', 0) > 0:
            m[A_PKW0:A_PKW0 + len(PICK_WHEAT_Q)] = True; m[A_PKWX0:A_PKWX0 + len(PICK_WHEAT_X)] = True; m[A_PKALL0 + 0] = True
        if shed.get('FERTILIZER', 0) > 0:
            m[A_PKF0:A_PKF0 + len(PICK_FERT_Q)] = True; m[A_PKFX0:A_PKFX0 + len(PICK_FERT_X)] = True; m[A_PKALL0 + 1] = True
        for k, a in enumerate(ANIMALS):
            if shed.get(a, 0) > 0:
                m[A_PKA0 + k] = True; m[A_PKALL0 + 2 + k] = True
            if shed.get(a, 0) > 1:
                m[A_PKA0 + len(ANIMALS) + k] = True
        for k, p in enumerate(PRODUCE7):
            if shed.get(p, 0) > 0:
                m[A_PKP0 + 2 * k] = True; m[A_PKP0 + 2 * k + 1] = True
                m[A_PKPX0 + k * len(PICK_PRODUCE_X):A_PKPX0 + (k + 1) * len(PICK_PRODUCE_X)] = True
        if room > 0:
            for k, p in enumerate(PRODUCTS):
                if inv.get(p, 0) > 0:
                    m[A_PLP0 + k] = True
                    for j, q in enumerate(PLACE_PART_Q):
                        if inv.get(p, 0) > q: m[A_PLQ0 + k * len(PLACE_PART_Q) + j] = True
                    for j, q in enumerate(PLACE_PART_X):
                        if inv.get(p, 0) > q: m[A_PLQX0 + k * len(PLACE_PART_X) + j] = True
    tile = farm['tiles'][y][x]

    for k, a in enumerate(ANIMALS):
        if inv.get(a, 0) > 0:
            if isinstance(tile, dict) and tile.get('kind') == STRUCTURE[a] and 'animal' not in tile:
                m[A_PLA0 + k] = True
            elif (x, y) in ACCESS and room_real > 0:
                m[A_PLA0 + k] = True
    if tile == 'LOCKED':
        return m
    if tile is None:
        m[A_COOP] = True; m[A_PASTURE] = True
        seeds = private.get('seeds', {})
        for k, c in enumerate(CROPS):
            if seeds.get(c, 0) > 0:
                m[A_PLANT0 + k] = True
        return m
    kind = tile.get('kind')
    if 'animal' in tile:
        if not tile['fed_today'] and inv.get('WHEAT', 0) > 0: m[A_FEED] = True
        if not tile['cared_today']: m[A_CARE] = True
        if tile['fertilizer_available']: m[A_COLLECT] = True
        if tile.get('yield_units', 0) > 0: m[A_HARVEST] = True
        return m
    m[A_DIG] = True
    if kind == 'PLANT':
        if not tile['watered_today']: m[A_WATER] = True
        if inv.get('FERTILIZER', 0) > 0 and tile.get('fertilized_until_day', -1) < day + 2: m[A_FERT] = True
        if tile.get('yield_units', 0) > 0 and day - tile['planted_day'] >= CROP_FIRST_YIELD[tile['crop']]: m[A_HARVEST] = True
    return m

def market_mask(farm, private, prices, shed_capacity=100):
    m = np.zeros(N_MK, dtype=bool); qmax = np.ones(N_MK, dtype=np.int64)
    m[MK_NONE] = True
    shed = private['shed']; room = shed_capacity - sum(shed.values())
    carried = {}
    for inv in private.get('inventories', []):
        for it, v in inv.items():
            carried[it] = carried.get(it, 0) + int(v)

    money = float(farm['money']) + sum((int(shed.get(p, 0)) + carried.get(p, 0)) * float(prices.get(p, 0)) for p in PRODUCTS)
    room = max(room, 1) if any(int(shed.get(p, 0)) > 0 for p in PRODUCTS) else room
    for p in PRODUCTS:
        n = int(shed.get(p, 0)) + carried.get(p, 0)
        if n > 0:
            i = MK_INDEX[('SELL', p)]; m[i] = True; qmax[i] = min(n, MAX_Q)
    for p in ('WHEAT', 'FERTILIZER'):
        pr = float(prices.get(p, 1e9))
        if room > 0 and money >= pr + 1:

            i = MK_INDEX[('BUY_PRODUCT', p)]; m[i] = True; qmax[i] = max(1, min(MAX_Q, int(money // max(pr, 1.0))))

    for p in ('WHEAT', 'FERTILIZER'):
        b = MK_INDEX[('BUY_PRODUCT', p)]
        if m[b]:
            i = MK_INDEX[('SELL', p)]; had = int(qmax[i]) if m[i] else 0; m[i] = True; qmax[i] = min(MAX_Q, had + int(qmax[b]))
    for c in CROPS:
        if money >= SEED_COST[c]:
            i = MK_INDEX[('BUY_SEED', c)]; m[i] = True; qmax[i] = max(1, min(MAX_Q, int(money // SEED_COST[c])))
    for a in ANIMALS:
        if room > 0 and money >= ANIMAL_COST[a]:
            i = MK_INDEX[('BUY_ANIMAL', a)]; m[i] = True; qmax[i] = max(1, min(MAX_Q, int(money // ANIMAL_COST[a])))
    if money >= fib(int(farm.get('hires_today', 0))) and len(farm['hands']) + 1 < MAX_UNITS:
        m[MK_HIRE] = True
    nq = len(farm.get('unlocked_quadrants', []))
    if nq < 4 and money >= LAND_PRICES[nq - 1]:
        m[MK_LAND] = True
    return m, qmax

def normalize_units(rec, obs):
    if not isinstance(rec, dict): return rec
    import copy
    rec = copy.deepcopy(rec); me = int(obs['player']); farm = obs['farms'][me]; priv = obs['private']
    pos = [farm['farmer']] + list(farm['hands']); shed = {k: int(v) for k, v in priv['shed'].items()}
    invs = [dict(x) for x in priv.get('inventories', [])]
    us = [rec.get('farmer')] + list(rec.get('hands', []) or [])
    demand = {}
    for u in us:
        if isinstance(u, list) and len(u) >= 2 and u[0] == 'PLANT': demand[u[1]] = demand.get(u[1], 0) + 1
    blocked = {c for c, n in demand.items() if n > int(priv.get('seeds', {}).get(c, 0))}
    for i, u in enumerate(us):
        if not isinstance(u, list) or not u: continue
        if u[0] == 'PLANT' and len(u) >= 2 and u[1] in blocked:
            us[i] = ['PASS']; continue
        if i >= len(pos): continue
        acc = (int(pos[i][0]), int(pos[i][1])) in ACCESS
        while len(invs) <= i: invs.append({})
        inv = invs[i]
        if u[0] == 'PICKUP' and len(u) > 1:
            try: q = int(u[2]) if len(u) >= 3 else 1
            except (TypeError, ValueError): continue
            if not acc or q <= 0: continue
            av = shed.get(u[1], 0); n = min(q, av)
            if n <= 0: continue
            if q >= av: us[i] = ['PICKUP', u[1], ALL_Q]
            shed[u[1]] = av - n; inv[u[1]] = inv.get(u[1], 0) + n
        elif u[0] == 'PLACE' and len(u) > 1:
            try: q = int(u[2]) if len(u) >= 3 else 1
            except (TypeError, ValueError): q = 1
            it = u[1]
            tile = farm['tiles'][int(pos[i][1])][int(pos[i][0])]
            if it in ANIMALS and isinstance(tile, dict) and tile.get('kind') == STRUCTURE[it] and 'animal' not in tile:
                if inv.get(it, 0) > 0: inv[it] -= 1
                continue
            have = int(inv.get(it, 0))
            if it in PRODUCTS and q >= have: us[i] = ['PLACE', it, ALL_Q]
            if acc and q > 0:
                room = max(0, 100 - sum(shed.values())); n = min(q, have, room)
                if n > 0: inv[it] = have - n; shed[it] = shed.get(it, 0) + n
        elif u[0] == 'DROP' and acc:
            for it, n in list(inv.items()):
                room = max(0, 100 - sum(shed.values())); tk = min(int(n), room)
                if tk > 0: shed[it] = shed.get(it, 0) + tk
                del inv[it]
    rec['farmer'] = us[0] if us else ['PASS']
    if 'hands' in rec or len(us) > 1: rec['hands'] = us[1:]
    return rec

def normalize_place(rec, inventories):
    raise RuntimeError('trackA codec v3: use normalize_units(rec, obs)')
