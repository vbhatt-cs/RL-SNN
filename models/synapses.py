class GapRL:
    def __init__(self, get_reward, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma):
        self.model = '''
                    r = get_reward(i) : 1 (constant over dt)
                    sig = sigma(v_post, dt) : 1 (constant over dt)
                    f = int(rand() < sig) : 1 (constant over dt)
                    dz1/dt = -z1 / tau_z : 1 / volt (clock-driven)
                    z = z1 + f * zeta - (1-f) * zeta * sig / (1 - sig) : 1 / volt (constant over dt)
                    dzeta/dt = -zeta / tau_i : 1 / volt (clock-driven)
                    w : volt
                    '''

        self.on_pre = '''
                    v_post += w
                    zeta += beta_sigma
                    '''

        self.on_post = '''
                    zeta = 0 / volt
                    '''


class Activator:
    def __init__(self, tau_e, nu_e):
        self.model = '''
                    a_post = a1 : 1 (summed)
                    da1/dt = -a1 / tau_e : 1 (clock-driven)
                    '''

        self.on_pre = '''
                    a1 += 1 - exp(-1/(nu_e * tau_e))
                    '''
