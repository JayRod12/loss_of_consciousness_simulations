from brian2.units import *

# Izhikevich excitatory and inhibitory neurons

a = 0.02/ms
c = -65*mV
b_ex = 0.2/ms
d_ex = 8*mV/ms
b_in = 0.25/ms
d_in = 2*mV/ms
Vt=30*mV

EXCITATORY_NEURON_EQS = '''
  dv/dt=(0.04/(mV*ms))*(v**2)+(5/ms)*v+140*mV/ms-u+I : volt
  du/dt=a*(b_ex*v-u) : volt/second
  I : volt/second
'''

INHIBITORY_NEURON_EQS = '''
  dv/dt=(0.04/(mV*ms))*(v**2)+(5/ms)*v+140*mV/ms-u+I : volt
  du/dt=a*(b_in*v-u) : volt/second
  I : volt/second
'''
  
EXCITATORY_RESET_EQ = '''
  v=c
  u=u+d_ex
'''

INHIBITORY_RESET_EQ = '''
  v=c
  u=u+d_in
'''

THRES_EQ = 'v>=Vt'
