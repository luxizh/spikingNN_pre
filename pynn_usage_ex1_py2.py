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
run(200.0)
for (population, variables, filename) in simulator.state.write_on_end:
    io = get_io(filename)
    population.write_data(filename, variables)
simulator.state.write_on_end = []
#end()
#type(IF_cond_exp)
#import matplotlib
#matplotlib.pyplot.ion()
#para=ifcell.tau_refrac
#print(para)
#ifcell._set_parameters('i_offset',0.11)