#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:15:36 2019

@author: luxi
"""
#import pyNN
#from pyNN import *
from pyNN.brian import *
#from brian import *
#setup(timestep=0.1,min_delay=2.0) 
ifcell=create(IF_cond_exp,{'i_offset':0.11,'tau_refrac':3.0,'v_thresh':-51.0})
#times=map(float, range(5,105,10))
#source=create(SpikeSourceArray,{'spike_times':times})
print(type(ifcell[0]))
ifcell[0].tau_refrac
ifcell[0].get_parameters()
#type(IF_cond_exp)
#import matplotlib
#matplotlib.pyplot.ion()
#para=ifcell.tau_refrac
#print(para)
#ifcell._set_parameters('i_offset',0.11)