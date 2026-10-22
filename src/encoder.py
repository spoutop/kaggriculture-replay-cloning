import numpy as np
import codec as C

C_TILE = 28
F_UNIT = 4 + len(C.ITEMS)
BASE_PRICE = {'WHEAT': 25, 'CARROT': 35, 'TOMATO': 60, 'STRAWBERRY': 120, 'MELON': 250, 'EGG': 50, 'MILK': 160, 'WOOL': 200, 'FERTILIZER': 100}
MARKET_T = {'WHEAT': 400, 'CARROT': 450, 'TOMATO': 200, 'STRAWBERRY': 100, 'MELON': 300, 'EGG': 332, 'MILK': 122, 'WOOL': 105, 'FERTILIZER': 200}
SHOPS = ['BAKERY', 'PIZZA_SHOP', 'BRUNCH_SPOT', 'YARN_STORE', 'ICE_CREAM_SHOP', 'PET_CAFE', 'SMOOTHIE_SHOP', 'FARMERS_MARKET']
ANIMAL_FIRST = {'GOOSE': 4, 'COW': 8, 'SHEEP': 6}
QUADS = ['NE', 'SW', 'SE']
F_GLOB = 5 + 2 * (3 + len(QUADS)) + len(C.CROPS) + len(C.ITEMS) + 1 + 2 * len(C.PRODUCTS) + len(SHOPS) + 3
_CROP_I = {c: i for i, c in enumerate(C.CROPS)}
_ANIM_I = {a: i for i, a in enumerate(C.ANIMALS)}
_ACCESS = set(C.ACCESS)

def _tiles(farm, day, step, out):
    tiles = farm['tiles']
    for y in range(10):
        row = tiles[y]
        for x in range(10):
            t = row[x]
            if t is None:
                out[1, y, x] = 1.0
            elif t == 'LOCKED':
                out[0, y, x] = 1.0
            else:
                kind = t.get('kind')
                if 'animal' in t:
                    out[10 + _ANIM_I[t['animal']], y, x] = 1.0
                    out[20, y, x] = 1.0 if t.get('fed_today') else 0.0
                    out[21, y, x] = 1.0 if t.get('cared_today') else 0.0
                    out[22, y, x] = float(t.get('consecutive_unfed', 0))
                    out[23, y, x] = 1.0 if t.get('fertilizer_available') else 0.0
                    out[24, y, x] = t.get('yield_units', 0) / 6.0
                    age = day - t.get('placed_day', day)
                    out[25, y, x] = min(1.0, max(0.0, (ANIMAL_FIRST[t['animal']] - age) / 8.0))
                elif kind == 'PLANT':
                    crop = t['crop']
                    out[3 + _CROP_I[crop], y, x] = 1.0
                    age = day - t['planted_day']
                    out[13, y, x] = min(age, 24) / 12.0
                    out[14, y, x] = 1.0 if t.get('watered_today') else 0.0
                    out[15, y, x] = float(t.get('consecutive_unwatered', 0))
                    out[16, y, x] = t.get('yield_units', 0) / 6.0
                    out[17, y, x] = max(0, t.get('fertilized_until_day', -1) - day + 1) / 3.0
                    out[18, y, x] = min(1.0, max(0.0, (C.CROP_FIRST_YIELD[crop] - age) / 10.0))
                    mls = t.get('max_lifespan_step', -1)
                    if mls >= 0:
                        out[19, y, x] = 1.0 if step >= mls else max(0.0, 1.0 - (mls - step) / 48.0) * 0.5
                elif kind == 'WEED':
                    out[2, y, x] = 1.0
                elif kind == 'COOP':
                    out[8, y, x] = 1.0
                elif kind == 'PASTURE':
                    out[9, y, x] = 1.0
    for (ax, ay) in _ACCESS:
        out[27, ay, ax] = 1.0
    out[26, farm['farmer'][1], farm['farmer'][0]] += 1.0 / 3.0
    for h in farm['hands']:
        out[26, h[1], h[0]] += 1.0 / 3.0

def encode(obs):
    me = int(obs['player']); opp = 1 - me
    step = int(obs.get('step', 0)); day = step // 24; hour = step % 24
    farms = obs['farms']; farm = farms[me]; ofarm = farms[opp]; priv = obs['private']
    grid = np.zeros((2, C_TILE, 10, 10), dtype=np.float32)
    _tiles(farm, day, step, grid[0]); _tiles(ofarm, day, step, grid[1])
    n_units = min(1 + len(farm['hands']), C.MAX_UNITS)
    units = np.zeros((C.MAX_UNITS, F_UNIT), dtype=np.float32); xy = np.zeros((C.MAX_UNITS, 2), dtype=np.int64)
    invs = priv['inventories']
    for i in range(min(n_units, C.MAX_UNITS)):
        p = farm['farmer'] if i == 0 else farm['hands'][i - 1]
        xy[i, 0] = p[0]; xy[i, 1] = p[1]
        units[i, 0] = 1.0; units[i, 1] = 1.0 if i == 0 else 0.0; units[i, 2] = p[0] / 9.0; units[i, 3] = p[1] / 9.0
        inv = invs[i] if i < len(invs) else {}
        for k, it in enumerate(C.ITEMS):
            v = inv.get(it, 0)
            if v:
                units[i, 4 + k] = min(v, 40) / 10.0
    g = []
    g += [step / 720.0, day / 30.0, hour / 24.0, 1.0 if hour >= 20 else 0.0, float(me)]
    for f in (farm, ofarm):
        g += [np.log1p(max(0.0, float(f['money']))) / 12.0, f.get('hires_today', 0) / 12.0, len(f['hands']) / 15.0]
        uq = set(f.get('unlocked_quadrants', []))
        g += [1.0 if q in uq else 0.0 for q in QUADS]
    seeds = priv.get('seeds', {}); shed = priv['shed']
    g += [min(seeds.get(c, 0), 30) / 10.0 for c in C.CROPS]
    g += [min(shed.get(it, 0), 100) / 25.0 for it in C.ITEMS]
    g += [sum(shed.values()) / 100.0]
    mk = obs['market']
    g += [float(mk['prices'][p]) / BASE_PRICE[p] - 1.0 for p in C.PRODUCTS]
    g += [float(np.clip((mk['inventory'][p] - 10000) / MARKET_T[p], -3.0, 3.0)) for p in C.PRODUCTS]
    shops = obs['town'].get('unlocked_shops', [])
    g += [shops.count(s) / 3.0 for s in SHOPS]
    g += [len(shops) / 8.0, ((3 - day % 3) % 3) / 3.0 if len(shops) < 8 else 0.0, 1.0 if len(shops) >= 8 else 0.0]
    glob = np.asarray(g, dtype=np.float32)
    assert glob.shape[0] == F_GLOB, (glob.shape, F_GLOB)
    return grid, units, xy, n_units, glob

F_HIST = 3 * len(C.PRODUCTS)
HIST_LAGS = (1, 4, 24)

class InvHist:
    def __init__(self): self.buf = []; self.step = -1
    def push(self, obs):
        step = int(obs.get('step', 0))
        if step < self.step or (step == 0 and self.step != 0): self.buf = []; self.step = -1
        if step != self.step: self.buf.append([float(obs['market']['inventory'][p]) for p in C.PRODUCTS]); self.step = step
        return hist_block(self.buf)

def hist_block(buf):
    out = np.zeros(F_HIST, dtype=np.float32); cur = buf[-1]
    for j, lag in enumerate(HIST_LAGS):
        if len(buf) > lag:
            old = buf[-1 - lag]
            for i, p in enumerate(C.PRODUCTS): out[j * len(C.PRODUCTS) + i] = np.clip((cur[i] - old[i]) / MARKET_T[p], -3.0, 3.0)
    return out

def masks(obs):
    me = int(obs['player']); farm = obs['farms'][me]; priv = obs['private']; day = int(obs.get('step', 0)) // 24
    um = np.zeros((C.MAX_UNITS, C.N_UNIT), dtype=bool); um[:, C.A_PASS] = True
    for i in range(min(1 + len(farm['hands']), C.MAX_UNITS)):
        um[i] = C.unit_mask(farm, priv, i, day)
    mm, qmax = C.market_mask(farm, priv, obs['market']['prices'])
    return um, mm, qmax
