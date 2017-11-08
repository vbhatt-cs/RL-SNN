class GapRL:
    def __init__(self, get_reward, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma):
        self.model = '''
                    r = get_reward(i) : 1 (constant over dt)
                    sig = sigma(v_post, dt) : 1 (constant over dt)
                    dz1/dt = -z1 / tau_z : 1 / volt (clock-driven)
                    z = z1 - zeta * sig / (1 - sig) : 1 / volt (constant over dt)
                    dzeta/dt = -zeta / tau_i : 1 / volt (clock-driven)
                    w : volt
                    '''

        self.on_pre = '''
                    v_post += w
                    zeta += beta_sigma
                    '''

        self.on_post = '''
                    z += zeta * (1 + sig) / (1 - sig)  
                    zeta = 0 / volt
                    '''
