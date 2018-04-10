from brian2.units import *

# Izhikevich excitatory and inhibitory neurons

#a = 0.02/ms
#c = -65*mV
#b = 0.2/ms
#d = 8*mV/ms
#b = 0.25/ms
#d = 2*mV/ms
Vt=30*mV

EXCITATORY_NEURON_EQS = '''
  dv/dt=(0.04/(mV*ms))*(v**2)+(5/ms)*v+140*mV/ms-u+I : volt
  du/dt=a*(b*v-u) : volt/second
  I : volt/second
  rnd : 1
  a : 1/second
  b : 1/second
  c : volt
  d : volt/second
'''

INHIBITORY_NEURON_EQS = '''
  dv/dt=(0.04/(mV*ms))*(v**2)+(5/ms)*v+140*mV/ms-u+I : volt
  du/dt=a*(b*v-u) : volt/second
  I : volt/second
  rnd : 1
  a : 1/second
  b : 1/second
  c : volt
  d : volt/second
'''
  
EXCITATORY_RESET_EQ = '''
  v=c
  u=u+d
'''

INHIBITORY_RESET_EQ = '''
  v=c
  u=u+d
'''

THRES_EQ = 'v>=Vt'
