import brian2 as b
import numpy as np

#from models.neurons import LIF
from models.synapses2 import SynIzhi2, Activator
import env

@b.check_units(idx=1, result=b.Hz)
def inp_rates(idx):
    la, ra = e.obs()
    la = la.repeat(2)
    ra = ra.repeat(2)
    l_idx = b.arange(len(idx), step=2)
    r_idx = l_idx + 1
    r = np.zeros(len(idx))
    r[l_idx] = la
    r[r_idx] = ra
    return r * b.Hz

@b.check_units(voltage=b.volt, dt=b.second, result=1)
def sigma(voltage, dt):
    sv = dt / tau_sigma * b.exp(beta_sigma * (voltage - th_i))
    sv.clip(0, 1)
    return sv


@b.check_units(idx=1, result=1)
def get_reward(idx):
    return e.r

@b.check_units(activation=1, idx=1, result=1)
def act(activation, idx):
    activation = activation.clip(0, 1)
    l_idx = b.arange(len(idx), step=2)
    r_idx = l_idx + 1

    a_avg = (activation[l_idx] + activation[r_idx]) / 2

    l_idx = b.arange(len(a_avg), step=2)
    r_idx = l_idx + 1

    theta = (a_avg[l_idx] - a_avg[r_idx]) * theta_max
    e.step(theta)
    return activation

if __name__ == '__main__':


    b.defaultclock.dt = 1 * b.ms
    b.prefs.codegen.target = 'numpy'

    num_parts = 20
    theta_max = 25.0
    e = env.WormFoodEnv((2, 3), num_parts=num_parts, theta_max=theta_max)

    tau_sigma = 20 * b.ms
    th_i = 16 * b.mV  # Threshold potential

    a_I = 0.02/b.ms; b_I = 0.2/b.ms
    eqs = '''dvm/dt = (0.04/ms/mV)*vm**2+(5/ms)*vm+140*mV/ms- u + I : volt
             du/dt = a*(b*vm-u) : volt/second
             I : volt/second
             '''
             #I = i_pulse/nA*volt/second*int(t >= p_start and t <= p_end): volt/second

    #input neurons fire poisson spike trains, with firing rate proportional to activation (between 0 and 50Hz)
    #activation of 2 per segment proportional to |orientation - leftmost angle|;
    #activation of other 2 proportional to |orientation - rightmost angle|
    eqs_inp = eqs + '''a = 0.00002/second : 1/second
                       b = 0.0002/second : 1/second
                       '''#parameters for regular spiking...d = +8
    #inp = b.NeuronGroup(80, eqs_inp,threshold='vm > -50*mV',reset='''vm = -65*mV
    #                                                                u += 8*volt/second
    #                                                                ''')
    inp_eq = 'rates: Hz '
    inp_th = 'rand() < rates*dt'

    inp = b.NeuronGroup(80, inp_eq, threshold=inp_th)
    inp.run_regularly('rates=inp_rates(i)', dt=b.defaultclock.dt)

    eqs_ex = eqs + '''a = 0.00002/second : 1/second
                        b = 0.0002/second : 1/second
                    ''' #parameters for regular spiking...d = +8
    h_ex = b.NeuronGroup(130, eqs_ex,threshold='vm > -50*mV',reset='''vm = -65*mV
                                                                        u += 8*volt/second
                                                                        ''')

    eqs_inh = eqs + '''a = 0.0001/second : 1/second
                        b = 0.0002/second : 1/second
                        ''' #parameters for fast spiking type...a = 0.1/ms
    h_inh = b.NeuronGroup(70, eqs_inh,threshold='vm > -50*mV',reset='''vm = -65*mV
                                                                        u += 2*volt/second
                                                                        ''')

    eqs_out = eqs + '''a = 0.00002/second : 1/second
                        b = 0.0002/second : 1/second
                        '''#parameters for regular spiking...d = +8
    out = b.NeuronGroup(80, eqs_out,threshold='vm > -50*mV',reset='''vm = -65*mV
                                                                u += 8*volt/second
                                                                ''')


    #####Synapses#####
    #Each timestep postsynaptic potential changes by s_ij*w_ij(t)*g_ij(t)*(E_ij - V_i(t))*dt
    #E_ij is reversal potential as mentioned by E below
    #s_ij is constant positive strength as mentioned below by s
    #g_ij is conductance which varies as g_ij(t) = g_ij(t-dt)*exp(-dt/taug) + f_j(t)
    #f_j(t) is spiketrain of presynaptic neuron, taug is a decay time constant
    
    strength_ex = 0.1
    strength_inh = 0.2
    E_ex = 0*b.mV
    E_inh = -90*b.mV
    tau_g = 5*b.ms

    tau_z = 5*b.ms
    gamma_ex = 0.025#*b.volt*b.second
    gamma_inh = -0.025#*b.volt*b.second
    w_min = 0*b.mV
    w_max = 1*b.mV
    beta_sigma = 0.2/b.mV
    k = b.exp(-b.defaultclock.dt/tau_g)
    beta = b.exp(-b.defaultclock.dt/tau_z)
    tau_i = 20*b.ms

    #s = 0.1, E = 0mV, taug = 5ms
    syn_ex = SynIzhi2(get_reward, sigma, tau_z, tau_i, gamma_ex, w_min, w_max, beta_sigma, strength_ex, E_ex, tau_g)
    S_inp_ex = b.Synapses(inp, h_ex, model = syn_ex.model, on_pre = syn_ex.on_pre)
    #S_inp_ex.run_regularly('w = clip(w + gamma_ex * r * z, w_min, w_max)', dt=b.defaultclock.dt)

    #s = 0.1, E = 0mV, taug = 5ms
    #syn2 = SynIzhi(get_reward, sigma, k, beta, gamma_ex, w_min, w_max, beta_sigma, strength_ex, E_ex)
    S_inp_inh = b.Synapses(inp, h_inh, model = syn_ex.model, on_pre = syn_ex.on_pre)
    #S_inp_inh.run_regularly('w = clip(w + gamma_ex * r * z, w_min, w_max)', dt=b.defaultclock.dt)

    #s = 0.2, E = -90mV, taug = 5ms
    syn_inh = SynIzhi2(get_reward, sigma, tau_z, tau_i, gamma_inh, w_min, w_max, beta_sigma, strength_inh, E_inh, tau_g)
    S_inh_ex = b.Synapses(h_inh, h_ex, model = syn_inh.model, on_pre = syn_inh.on_pre)
    #S_inh_ex.run_regularly('w = clip(w + gamma_inh * r * z, w_min, w_max)', dt=b.defaultclock.dt)

    #s = 0.1, E = 0mV, taug = 5ms
    #syn4 = SynIzhi(get_reward, sigma, k, beta, gamma_ex, w_min, w_max, beta_sigma, strength_ex, E_ex)
    S_ex_inh = b.Synapses(h_ex, h_inh, model = syn_ex.model, on_pre = syn_ex.on_pre)
    #S_ex_inh.run_regularly('w = clip(w + gamma_ex * r * z, w_min, w_max)', dt=b.defaultclock.dt)

    #s = 0.1, E = 0mV, taug = 5ms
    #syn5 = SynIzhi(get_reward, sigma, k, beta, gamma_ex, w_min, w_max, beta_sigma, strength_ex, E_ex)
    S_ex_out = b.Synapses(h_ex, out, model = syn_ex.model, on_pre = syn_ex.on_pre)
    #S_ex_out.run_regularly('w = clip(w + gamma_ex * r * z, w_min, w_max)', dt=b.defaultclock.dt)

    #s = 0.2, E = -90mV, taug = 5ms
    #syn6 = SynIzhi(get_reward, sigma, k, beta, gamma_inh, w_min, w_max, beta_sigma, strength_inh, E_inh)
    S_inh_out = b.Synapses(h_inh, out, model = syn_inh.model, on_pre = syn_inh.on_pre)
    #S_inh_out.run_regularly('w = clip(w + gamma_inh * r * z, w_min, w_max)', dt=b.defaultclock.dt)

    S_inp_ex.connect(p=0.15)
    S_inh_out.connect(p=0.15)
    S_inh_ex.connect(p=0.15)
    S_ex_inh.connect(p=0.15)
    S_ex_out.connect(p=0.15)
    S_inh_out.connect(p=0.15)


    ######motor activations######
    #a(t) = a(t-dt)exp(-dt/taue) + (1-exp(-1/ve/taue))*f(t)
    #f(t) indicates whether neuron has fired or not. scaled such that a = 1 for 25Hz firing freq
    #a hardbounded btw 0 and 1
    #2 effectors made of two neurons each
    #first effector gives a+ (by avg of activation of its two neurons)
    #second effector give a-
    #overall articulation  = (a+ - a-)*theta_max

    act_group = b.NeuronGroup(80, 'a : 1')
    act_group.run_regularly('a = act(a, i)', dt=b.defaultclock.dt)

    tau_e = 2 * b.second
    nu_e = 25 * b.Hz
    syn_oa = Activator(tau_e, nu_e)
    oa_group = b.Synapses(out, act_group, model=syn_oa.model, on_pre=syn_oa.on_pre)
    oa_group.connect(j='i')


    #now the actual simulation
    print e.disToFood
    b.run(1 * b.second)
    print e.disToFood
    e.plot()
    b.figure()
    b.plot(e.d_history)