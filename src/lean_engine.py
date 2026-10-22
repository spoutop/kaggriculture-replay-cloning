from types import SimpleNamespace
from kaggle_environments.utils import Struct
from kaggle_environments.envs.kaggriculture import kaggriculture as K

DEFAULT_CONFIG = dict(actTimeout=1, boardSize=10, episodeSteps=720, farmHandCostMult=1, marketParams={}, maxMarketOrdersPerTurn=10,
                      runTimeout=1200, seed=None, shedCapacity=100, startingMoney=3000, townCenterSellInterval=24,
                      townShopSellInterval=4, townShopUnlockInterval=3, turnsPerDay=24, weedSpawnChance=0.005)

class LeanGame:
    __slots__ = ('env', 'state', 'step_no', 'episode_steps')

    def __init__(self, seed, config=None):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        cfg['seed'] = None
        self.env = SimpleNamespace(configuration=Struct(**cfg), info={'seed': int(seed)}, done=False)
        self.state = [Struct(observation=Struct(remainingOverageTime=60, step=0), action=None, status='ACTIVE', reward=0, info={}) for _ in range(2)]
        K.interpreter(self.state, self.env)
        self.step_no = 0
        self.episode_steps = int(cfg['episodeSteps'])
        for s in self.state:
            s.observation.step = 0

    @property
    def done(self):
        return self.state[0].status == 'DONE'

    @property
    def rewards(self):
        return [float(s.reward) for s in self.state]

    def obs(self, i):
        return self.state[i].observation

    def step(self, a0, a1):
        self.state[0].action = a0
        self.state[1].action = a1
        K.interpreter(self.state, self.env)
        self.step_no += 1
        for s in self.state:
            s.observation.step = self.step_no
        if self.done:
            self.env.done = True

def play(agent0, agent1, seed, config=None):
    g = LeanGame(seed, config)
    cfgd = g.env.configuration
    while not g.done:
        g.step(agent0(g.obs(0), cfgd), agent1(g.obs(1), cfgd))
    return g
