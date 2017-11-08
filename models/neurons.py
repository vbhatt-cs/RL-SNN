import brian2 as b


class LIF:
    def __init__(self, tau_i, v_r, sigma, tau_sigma, beta_sigma):
        # self.equ = b.Equations('v = v1 + v2 : volt')
        # self.equ += b.Equations('dv1/dt = -v1/tau_i : volt')
        # self.equ += b.Equations('dv2/dt = -v2/tau_i : volt')
        self.equ = b.Equations('dv/dt = -v/tau_i : volt')
        self.threshold = 'rand() < sigma(v, dt)'
        self.reset = 'v = v_r'
