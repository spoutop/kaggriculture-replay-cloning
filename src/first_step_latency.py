import json, os, sys, time
t0 = time.perf_counter()
from kaggle_environments import make
from kaggle_environments.agent import get_last_callable, read_file
t_imp = time.perf_counter()
d = os.path.abspath(sys.argv[1]); p = os.path.join(d, 'main.py')
f = get_last_callable(read_file(p, p), path=p); t_load = time.perf_counter()
env = make('kaggriculture'); env.reset(); obs = env.state[0].observation; cfg = env.configuration
t1 = time.perf_counter(); a = f(obs, cfg); t2 = time.perf_counter()
r = dict(bundle=d, kaggle_env_import_s=round(t_imp - t0, 3), load_main_s=round(t_load - t_imp, 3), first_call_s=round(t2 - t1, 3),
         load_plus_first_s=round((t_load - t_imp) + (t2 - t1), 3), errors=f.__globals__.get('_ERRORS'), action_ok=isinstance(a, dict))
print(json.dumps(r));
if len(sys.argv) > 3 and sys.argv[2] == '--out': json.dump(r, open(sys.argv[3], 'w'))
