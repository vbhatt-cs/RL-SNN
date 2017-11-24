import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle

import env


def inp_rates(_e, idx):
    la, ra = _e.obs()
    la = la.repeat(2)
    ra = ra.repeat(2)
    l_idx = np.arange(len(idx), step=2)
    r_idx = l_idx + 1
    r = np.zeros(len(idx))
    r[l_idx] = la
    r[r_idx] = ra
    return r


def sigma(_v, _th, _dt, _tau, _beta):
    sv = _dt / _tau * np.exp(_beta * (_v - _th))
    sv = sv.clip(0, 1 - 1e-8)
    return sv


def get_reward(_e):
    return _e.r


def act(_a, idx):
    _a = _a.clip(0, 1)
    l_idx = np.arange(len(idx), step=2)
    r_idx = l_idx + 1

    a_avg = (_a[l_idx] + _a[r_idx]) / 2

    l_idx = np.arange(len(a_avg), step=2)
    r_idx = l_idx + 1

    theta = (a_avg[l_idx] - a_avg[r_idx]) * theta_max
    e.step(theta)
    return _a


for i in range(1):
    rng = np.random.RandomState(i)
    num_parts = 20
    theta_max = 25.0
    e = env.WormFoodEnv((11, 11), num_parts=num_parts, theta_max=theta_max)

    N_i = num_parts * 4
    N_h = num_parts * 10
    N_o = num_parts * 4

    T = 30
    dt = 1e-3

    # Neurons
    tau_i = 20e-3  # Time constant for LIF leak
    v_r = 10e-3  # Reset potential
    th_i = 16e-3  # Threshold potential
    tau_sigma = 20e-3
    beta_sigma = 0.2e+3

    f_i = np.zeros(N_i, dtype=np.bool_)
    V_h = np.zeros(N_o + N_h) + v_r
    f_h = np.zeros(N_o + N_h, dtype=np.bool_)

    # Synapses
    tau_z = 5e-3
    # a_plus = 1.05
    # a_minus = 0.005
    w_min_i = -0.1e-3 * 5
    w_max_i = 1.5e-3 * 5
    gamma_i = 0.25 * (w_max_i - w_min_i) * 1e-3
    # gamma_i = 1e-6

    # Synapses from hidden and output neurons
    w_min = -0.4e-3
    w_max = 1e-3
    gamma = 0.25 * (w_max - w_min) * 1e-3
    # gamma = 1e-6

    # Synapse variables
    syn_ih_conn = np.zeros((N_i, N_h + N_o), dtype=np.bool_)
    syn_hh_conn = np.zeros((N_h + N_o, N_h + N_o), dtype=np.bool_)
    syn_ih_w = np.zeros((N_i, N_h + N_o))
    syn_hh_w = np.zeros((N_h + N_o, N_h + N_o))
    syn_ih_z = np.zeros((N_i, N_h + N_o))
    syn_hh_z = np.zeros((N_h + N_o, N_h + N_o))

    syn_ih_grad_V = np.zeros((N_i, N_h + N_o))
    syn_hh_grad_V = np.zeros((N_h + N_o, N_h + N_o))
    # syn_ih_p_plus = np.zeros((N_i, N_h + N_o))
    # syn_ih_p_minus = np.zeros((N_i, N_h + N_o))
    # syn_hh_p_plus = np.zeros((N_h + N_o, N_h + N_o))
    # syn_hh_p_minus = np.zeros((N_h + N_o, N_h + N_o))

    syn_ih_zeta = np.zeros((N_i, N_h + N_o))
    syn_hh_zeta = np.zeros((N_h + N_o, N_h + N_o))
    n_conn = int((N_h + N_o) * 0.15)

    # Connect and initialize weights
    for _i in range(N_i):
        fanout = rng.choice(N_h + N_o, n_conn)
        _w = rng.uniform(w_min_i, w_max_i, fanout.shape) / 100
        syn_ih_conn[_i, fanout] = 1
        syn_ih_w[_i, fanout] = _w

    for _i in range(N_h + N_o):
        fanout = rng.choice(N_h + N_o, n_conn)
        _w = rng.uniform(w_min, w_max, fanout.shape) / 100
        syn_hh_conn[_i, fanout] = 1
        syn_hh_w[_i, fanout] = _w

    # Activations
    activations = np.zeros(N_o)
    tau_e = 2
    nu_e = 25

    # Monitoring
    # V_h_mon = []
    # f_i_mon_i = []
    # f_i_mon_t = []
    # f_h_mon_i = []
    # f_h_mon_t = []
    # z_mon = []
    # zeta_mon = []
    # w_mon = []

    print e.disToFood

    # Run simulation
    for t in tqdm.tqdm(np.arange(0, T, dt)):
        if int(t / dt) % 10000 == 0:
            e.change_food()
        # if int(t / dt) % 1000 == 0:
        #     e.render()

        # Observation
        rates = inp_rates(e, np.arange(N_i))

        # V_h(t)
        V_h = V_h * np.exp(-dt / tau_i) + (syn_ih_conn * syn_ih_w).T.dot(f_i) + (syn_hh_conn * syn_hh_w).T.dot(f_h)

        # # p+(t)
        # syn_ih_p_plus = syn_ih_p_plus * np.exp(-dt / tau_i) + a_plus * f_i[:, np.newaxis]
        # syn_hh_p_plus = syn_hh_p_plus * np.exp(-dt / tau_i) + a_plus * f_h[:, np.newaxis]
        #
        # # p-(t)
        # syn_ih_p_minus = syn_ih_p_minus * np.exp(-dt / tau_i) + a_minus * f_h
        # syn_hh_p_minus = syn_hh_p_minus * np.exp(-dt / tau_i) + a_minus * f_h

        # f_i(t)
        f_i = rng.rand(N_i) < rates * dt

        # f_h(t)
        sig = sigma(V_h, th_i, dt, tau_sigma, beta_sigma)
        f_h = rng.rand(N_h + N_o) < sig
        V_h = V_h * (1 - f_h) + v_r * f_h

        # dV(t)/dw
        syn_ih_grad_V = syn_ih_grad_V * np.exp(-dt / tau_i) + syn_ih_conn * f_i[:, np.newaxis]
        syn_hh_grad_V = syn_hh_grad_V * np.exp(-dt / tau_i) + syn_hh_conn * f_h[:, np.newaxis]

        # zeta(t)
        syn_ih_zeta = beta_sigma * syn_ih_grad_V * f_h - beta_sigma * sig / (1 - sig) * syn_ih_grad_V * (1 - f_h)
        syn_hh_zeta = beta_sigma * syn_hh_grad_V * f_h - beta_sigma * sig / (1 - sig) * syn_hh_grad_V * (1 - f_h)

        # # zeta(t)
        # syn_ih_zeta = syn_ih_p_plus * f_h + syn_ih_p_minus * f_i[:, np.newaxis]
        # syn_hh_zeta = syn_hh_p_plus * f_h + syn_hh_p_minus * f_h[:, np.newaxis]

        # Action
        activations = activations * np.exp(-dt / tau_e) + (1 - np.exp(-1 / (nu_e * tau_e))) * f_h[N_h:]
        activations = act(activations, np.arange(N_o))

        # z(t+1)
        syn_ih_z = syn_ih_z * np.exp(-dt / tau_z) + syn_ih_zeta
        syn_hh_z = syn_hh_z * np.exp(-dt / tau_z) + syn_hh_zeta

        # w(t+1)
        syn_ih_w += gamma_i * get_reward(e) * syn_ih_z
        syn_ih_w = syn_ih_w.clip(w_min_i, w_max_i)
        syn_hh_w += gamma * get_reward(e) * syn_hh_z
        syn_hh_w = syn_hh_w.clip(w_min, w_max)

        # if int(t / dt) % 1000 == 0:
        #     gamma_i -= gamma_i_step
        #     gamma -= gamma_step

        # for _n in np.where(f_i == 1)[0]:
        #     f_i_mon_i.append(_n)
        #     f_i_mon_t.append(t)

        # for _n in np.where(f_h == 1)[0]:
        #     f_h_mon_i.append(_n)
        #     f_h_mon_t.append(t)

        # z_mon.append(syn_ih_z[2][1])
        # zeta_mon.append(syn_ih_zeta[2][1])
        # w_mon.append(syn_ih_w[2][1])

    print e.disToFood

    with open('logs/s' + str(i), 'wb') as f:
        pickle.dump(e, f)

# e.plot()
# plt.figure()
# plt.plot(e.d_history)

# plt.figure()
# plt.plot(f_i_mon_t, f_i_mon_i, '.k')
# plt.figure()
# plt.plot(f_h_mon_t, f_h_mon_i, '.k')

# plt.figure()
# plt.plot(z_mon)
# plt.plot(zeta_mon)
#
# plt.figure()
# plt.plot(w_mon)

# plt.show()
