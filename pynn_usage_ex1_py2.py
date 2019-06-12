#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:15:36 2019

@author: luxi
"""
#import pyNN
#from pyNN import *
from pyNN.brian import *
#import  pandas
import os
import numpy
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel
#import matplotlib
#matplotlib.pyplot.ion()
#from brian import *
setup(timestep=0.1,min_delay=2.0) 
ifcell=create(IF_cond_exp,{'i_offset':0.11,'tau_refrac':3.0,'v_thresh':-51.0})
times=map(float, range(5,105,10))
source=create(SpikeSourceArray,{'spike_times':times})
#returns an ID object? which provides access to parameter of cell models
print(type(ifcell[0]))
print(ifcell[0].tau_refrac)
#refrac is in millisecond 
ifcell[0].tau_m=12.5
print(ifcell[0].get_parameters())
#v_reset is in uv and v_rest is in mv
#no v_init
connect(source,ifcell,weight=0.006,delay=2.0)
record_v(ifcell,'ifcell.pkl')
#ifcell.record('v')
ifcell.record('spikes')
ifcell[0:2].record(('v', 'gsyn_exc'))

run(200.0)

for (population, variables, filename) in simulator.state.write_on_end:
    io = get_io(filename)
    population.write_data(filename, variables)
simulator.state.write_on_end = []

end()
data = ifcell.get_data().segments[0]

vm = data.filter(name="v")[0]
gsyn = data.filter(name="gsyn_exc")[0]

Figure(
    Panel(vm, ylabel="Membrane potential (mV)"),
    Panel(gsyn, ylabel="Synaptic conductance (uS)"),
    Panel(data.spiketrains, xlabel="Time (ms)", xticks=True)
).save("simulation_results.png")

'''
v_value=ifcell.get_data()
print(v_value.segments[0].analogsignals[0].shape)
plt.figure()
plt.plot(v_value.segments[0].analogsignals[0])
plt.plot(v_value.segments[0].analogsignals[1])
plt.show()
'''

import pickle
f = open('ifcell.pkl', 'rb')
ifcell_load = pickle.load(f)
#print(ifcell_load)
#plt.figure()
#plt.plot(ifcell_load)
#type(IF_cond_exp)
#import matplotlib
#matplotlib.pyplot.ion()
#para=ifcell.tau_refrac
#print(para)
#ifcell._set_parameters('i_offset',0.11)