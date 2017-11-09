class GapRL:
    def __init__(self, get_reward, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma):
        self.model = '''
                    r = get_reward(i) : 1 (constant over dt)
                    sig = sigma(v_post, dt) : 1 (constant over dt)
                    w : volt
                    dz/dt = -z/tau_z : 1/volt (clock-driven)
                    prevSpike : second
                    '''

        self.on_pre = '''
                    v_post += w
                    prevSpike = t 
                    '''

        self.on_post = '''
                    z += beta_sigma*exp(-(t - prevSpike)/tau_i)
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

class SynIzhi:
    def __init__(self, get_reward, sigma, k, beta, gamma, w_min, w_max, beta_sigma, strength, E):
        #self.k = b.exp(-b.defaultclock.dt/tau_g)
        #self.beta = b.exp(-b.defaultclock.dt/tau_z)
        #dz/dt = (beta - 1)*z + zeta : 1/second
        self.model = '''
                    r = get_reward(i) : 1 (constant over dt)
                    sig = sigma(vm_post, dt) : 1 (constant over dt)
                    z = beta*z + zeta : volt
                    w = w + gamma*r*z : volt
                    psigBypw = -beta_sigma*sig*strength*g*(E - vm_post)*(E - vm_post)/2 : volt
                    zeta = f*(1/sig*psigBypw) + (1-f)*(-1/(1-sig)*psigBypw) : volt

                    I_post = strength*w*g*(E - vm_post) : volt/second (summed)
                    dg/dt = (k-1)*g + f: 1

                    f = 0 : 1
                    '''
        self.on_pre = '''
                    f = 1 
                    '''

        self.on_post = '''
                    '''

class SynIzhi2:
    def __init__(self, get_reward, sigma, tau_z, tau_i, gamma, w_min, w_max, beta_sigma, strength, E, tau_g):
        self.model = '''
                    r = get_reward(i) : 1 (constant over dt)
                    sig = sigma(vm_post, dt) : 1 (constant over dt)
                    w : volt
                    dz/dt = -z/tau_z : volt (clock-driven)
                    dg/dt = -g/tau_g : 1/volt/second (clock-driven)
                    prevSpike : second
                    '''

        self.on_pre = ''' 
                    prevSpike = t
                    g += 1/volt/second 
                    I_post += strength*w*g*(E - vm_post)
                    '''

        self.on_post = '''
                    z += beta_sigma*exp(-(t - prevSpike)/tau_i)
                    '''

