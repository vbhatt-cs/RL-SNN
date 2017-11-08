import brian2 as b
import numpy as np

import env


@b.check_units(idx=1, result=b.Hz)
def inp_rates(idx):
    la, ra = e.test_inp()
    print la, ra
    la = la.repeat(2)
    ra = ra.repeat(2)
    l_idx = b.arange(len(idx), step=2)
    r_idx = l_idx + 1
    r = np.zeros(len(idx))
    r[l_idx] = la / theta_max * 50
    r[r_idx] = ra / theta_max * 50
    print r
    return r * b.Hz


b.defaultclock.dt = 1 * b.ms
b.prefs.codegen.target = 'numpy'

num_parts = 3
theta_max = 25.0
e = env.WormFoodEnv((2, 3), num_parts=num_parts, theta_max=theta_max)

N_i = num_parts * 4
N_h = 200
N_o = num_parts * 4

inp_eq = 'rates: Hz '
inp_th = 'rand() < rates*dt'

inp_group = b.NeuronGroup(N_i, inp_eq, threshold=inp_th)

inp_group.run_regularly('rates=inp_rates(i)', dt=b.defaultclock.dt)
M = b.StateMonitor(inp_group, 'rates', record=True)

b.run(100 * b.ms)

for ri in M.rates:
    b.plot(M.t / b.ms, ri / b.Hz)

b.show()
