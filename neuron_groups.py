from brian2 import *
from izhikevich_constants import *

# Neuron groups with heterogeneity as presented in Simple Model of Spiking
# Neurons, IEEE Transactions on Neural Networks (2003) 14:1569-1572,
# Eugene Izhikevich.
def createGroup(N):
    return NeuronGroup(N,
        izhikevich_equations,
        threshold=threshold_equations,
        reset=reset_equations,
        method='rk4'
    )


def ExcitatoryNeuronGroup(N):
    G = createGroup()
    G.rnd = 'rand()'
    G.a = 0.02/ms
    G.b = 0.2/ms
    G.c = '(-65 + 15 * (rnd ** 2)) * mV'
    G.d = '(8 - 6 * (rnd ** 2)) * mV / ms'
    return G

def InhibitoryNeuronGroup(N):
    G = createGroup(N)
    G.rnd = 'rand()'
    G.a = '(0.02 + 0.08 * rnd) / ms'
    G.b = '(0.25 - 0.05 * rnd) / ms'
    G.c = -65 * mV
    G.d = 2 * mV / ms
    return G

Vt=30*mV

izhikevich_equations = '''
  dv/dt=(0.04/(mV*ms))*(v**2)+(5/ms)*v+140*mV/ms-u+I : volt
  du/dt=a*(b*v-u) : volt/second
  I : volt/second
  rnd : 1
  a : 1/second
  b : 1/second
  c : volt
  d : volt/second
'''
  
reset_equations = '''
  v=c
  u=u+d
'''

threshold_equations = 'v>=Vt'
