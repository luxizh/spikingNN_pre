import pyNN.brian as sim

sim.setup()

p=sim.Population(5,sim.IF_cond_exp())
p.set(tau_m=15.0)
print(p.get('tau_m'))

p[0,2,4].set(tau_m=10)
print(p.get('tau_m'))
print(p[0].tau_m)


#random value
from pyNN.random import RandomDistribution, NumpyRNG
gbar_na_distr = RandomDistribution('normal', (20.0, 2.0), rng=NumpyRNG(seed=85524))
p = sim.Population(7, sim.HH_cond_exp(gbar_Na=gbar_na_distr))
print(p.get('gbar_Na'))
print(p[0].gbar_Na)

#setting from an array
import numpy as np
p = sim.Population(6, sim.SpikeSourcePoisson(rate=np.linspace(10.0, 20.0, num=6)))
print(p.get('rate'))

#using function to calculate
from numpy import sin, pi
p = sim.Population(8, sim.IF_cond_exp(i_offset=lambda i: sin(i*pi/8)))
print(p.get('i_offset'))

#Setting parameters as a function of spatial position
from pyNN.space import Grid2D


sim.end()