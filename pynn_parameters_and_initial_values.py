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
grid = Grid2D(dx=10.0, dy=10.0)
p = sim.Population(16, sim.IF_cond_alpha(), structure=grid)
def f_v_thresh(pos):
    x, y, z = pos.T
    return -50 + 0.5*x - 0.2*y
p.set(v_thresh=lambda i: f_v_thresh(p.position_generator(i)))
print(p.get('v_thresh').reshape((4,4)))

#multiple types
n = 1000
parameters = {
    'tau_m': RandomDistribution('uniform', (10.0, 15.0)),
    'cm':    0.85,
    'v_rest': lambda i: np.cos(i*pi*10/n),
    'v_reset': np.linspace(-75.0, -65.0, num=n)}
p = sim.Population(n, sim.IF_cond_alpha(**parameters))
print(p.get('tau_m'))
p.set(v_thresh=lambda i: -65 + i/n, tau_refrac=5.0)
print(p.get('v_thresh','tau_refrac','cm'))
print(p.get('tau_refrac'))
print(p[0].get_parameters())

#time series
#unsuccessful test for the following code
#celltype = sim.SpikeSourceArray(np.array([5.0, 15.0, 45.0, 99.0]))
'''
celltype = sim.SpikeSourceArray([sim.Sequence([5.0, 15.0, 45.0, 99.0]),
                             sim.Sequence([2.0, 5.3, 18.9]),
                             sim.Sequence([17.8, 88.2, 100.1])
                            ])
'''
#error take one argument but 2 given
#generate function
number = int(2 * simtime * input_rate / 1000.0)
#simtime ??

def generate_spike_times(i):
    gen = lambda: sim.Sequence(numpy.add.accumulate(numpy.random.exponential(1000.0 / input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()

celltype = sim.SpikeSourceArray(spike_times=generate_spike_times)

cell_type = sim.GIF_cond_exp(
    # this parameter has the same value in all neurons in the population
    tau_gamma=(1.0, 10.0, 100.0),  # Time constants for spike-frequency adaptation in ms.
    # the following parameter has different values for each neuron
    a_eta=[(0.1, 0.1, 0.1),        # Post-spike increments for spike-triggered current in nA
           (0.0, 0.0, 0.0),
           (0.0, 0.0, 0.0),
           (0.0, 0.0, 0.0)]
        )
#no definition for GIF
sim.end()