import pyNN.brian as sim
from pyNN.random import (NumpyRNG,RandomDistribution)
#import matplotlib.pyplot as plt
import numpy as np

sim.setup(timestep=0.01)

'''
Synapse types
'''
#=====Fixed synaptic weight=====
syn = sim.StaticSynapse(weight=0.04, delay=0.5)
#random
w = RandomDistribution('gamma', [10, 0.004], rng=NumpyRNG(seed=4242))
syn = sim.StaticSynapse(weight=w, delay=0.5)
#specify parameters as a function of the distance- um
syn = sim.StaticSynapse(weight=w, delay="0.2 + 0.01*d")

#=====Short-term synaptic plasticity=====
depressing_synapse = sim.TsodyksMarkramSynapse(weight=w, delay=0.2, U=0.5,tau_rec=800.0, tau_facil=0.0)
tau_rec = RandomDistribution('normal', [100.0, 10.0])
facilitating_synapse = sim.TsodyksMarkramSynapse(weight=w, delay=0.5, U=0.04,tau_rec=tau_rec)

#=====Spike-timing-dependent plasticity=====

stdp = sim.STDPMechanism(
          weight=0.02,  # this is the initial value of the weight
          #delay="0.2 + 0.01*d",
          timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,A_plus=0.01, A_minus=0.012),
          weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.04),
          dendritic_delay_fraction=0)
#Error: The pyNN.brian backend does not currently support dendritic delays:
# for the purpose of STDP calculations all delays are assumed to be axonal
#for brian dendritic_delay_fraction=0 default value 1.0
'''
Connection algorithms
'''

connector = sim.AllToAllConnector(allow_self_connections=False)  # no autapses
#default True

connector = sim.OneToOneConnector()

#Connecting neurons with a fixed probability
connector = sim.FixedProbabilityConnector(p_connect=0.2)

#Connecting neurons with a position-dependent probability
DDPC = sim.DistanceDependentProbabilityConnector
connector = DDPC("exp(-d)")
connector = DDPC("d<3")
#The constructor requires a string d_expression, which should be a distance expression, 
# as described above for delays, but returning a probability (a value between 0 and 1)

#Divergent/fan-out connections
#connects each pre-synaptic neuron to exactly n post-synaptic neurons chosen at random
connector = sim.FixedNumberPostConnector(n=30)

distr_npost = RandomDistribution(distribution='binomial', n=100, p=0.3)
connector = sim.FixedNumberPostConnector(n=distr_npost)

#Divergent/fan-in connections
#connects each post-synaptic neuron to n pre-synaptic neurons
connector = sim.FixedNumberPreConnector(5)
distr_npre = RandomDistribution(distribution='poisson', lambda_=5)
connector = sim.FixedNumberPreConnector(distr_npre)

#Specifying a list of connections
connections = [
  (0, 0, 0.0, 0.1),
  (0, 1, 0.0, 0.1),
  (0, 2, 0.0, 0.1),
  (1, 5, 0.0, 0.1)
]
connector = sim.FromListConnector(connections, column_names=["weight", "delay"])

#Specifying an explicit connection matrix
connections = np.array([[0, 1, 1, 0],
                           [1, 1, 0, 1],
                           [0, 0, 1, 0]],
                          dtype=bool)
connector = sim.ArrayConnector(connections)

'''
Projections
'''

'''
cell types
'''
#brian.list_standard_models()
#refractory_period = RandomDistribution('uniform', [2.0, 3.0], rng=NumpyRNG(seed=4242))
#Brian does not support heterogenerous refractory periods with CustomRefractoriness
ctx_parameters={'cm': 0.25, 'tau_m': 20.0, 'v_rest': -60, 'v_thresh': -50, \
'tau_refrac': 3.0,'v_reset': -60, 'v_spike': -50.0, 'a': 1.0, \
'b': 0.005, 'tau_w': 600, 'delta_T': 2.5,'tau_syn_E': 5.0, 'e_rev_E': 0.0, \
'tau_syn_I': 10.0, 'e_rev_I': -80 }
tc_parameters = ctx_parameters.copy()
tc_parameters.update({'a': 20.0, 'b': 0.0})

thalamocortical_type = sim.EIF_cond_exp_isfa_ista(**tc_parameters)
cortical_type = sim.EIF_cond_exp_isfa_ista(**ctx_parameters)

'''
populations
'''
#tec_cells=create(thalamocortical_type,n=100)
#tc_cells = Population(100, thalamocortical_type)
#ctx_cells = Population(500, cortical_type)

from pyNN.space import Grid2D, RandomStructure, Sphere
tc_cells = sim.Population(100, thalamocortical_type,
                      structure=RandomStructure(boundary=Sphere(radius=200.0)),
                      initial_values={'v': -70.0},
                      label="Thalamocortical neurons")
from pyNN.random import RandomDistribution
v_init = RandomDistribution('uniform', (-70.0, -60.0))
ctx_cells = sim.Population(500, cortical_type,
                       structure=Grid2D(dx=10.0, dy=10.0),
                       initial_values={'v': v_init},
                       label="Cortical neurons")
pre=tc_cells[:50]
post=ctx_cells[:50]
excitatory_connections = sim.Projection(pre, post, sim.AllToAllConnector(),
                                        sim.StaticSynapse(weight=0.123))
#full example
from pyNN.space import Space
rng = NumpyRNG(seed=64754)
sparse_connectivity = sim.FixedProbabilityConnector(0.1, rng=rng)
weight_distr = RandomDistribution('normal', [0.01, 1e-3], rng=rng)
facilitating = sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0, tau_facil=1000.0,
                                     weight=weight_distr, delay=lambda d: 0.1+d/100.0)
space = Space(axes='xy')
#specifying periodic boundary conditions
#space = Space(periodic_boundaries=((0,500), (0,500), None))
#calculates distance on the surface of a torus of circumference 500 µm 
#(wrap-around in the x- and y-dimensions but not z)
inhibitory_connections = sim.Projection(pre, post,
                                    connector=sparse_connectivity,
                                    synapse_type=stdp,#facilitating,
                                    receptor_type='inhibitory',
                                    space=space,
                                    label="inhibitory connections")

#Accessing weights and delays
#format can be list or array
print(excitatory_connections.get('weight', format='list')[3:7])
#suppress the coordinates of the connection in list
print(excitatory_connections.get('weight', format='list', with_address=False)[3:7])
print(inhibitory_connections.get('delay', format='array')[:3,:5])
#other parameters
print(inhibitory_connections.get('A_plus', format='list')[0:4])#,with_address=False

#print(inhibitory_connections.get(['weight', 'delay'], format='list'))
connection_data = inhibitory_connections.get(['weight', 'delay'], format='list')
for connection in connection_data[:5]:
   src, tgt, w, d = connection
   print("weight = %.4f  delay = %4.2f" % (w, d))

weights, delays = inhibitory_connections.get(['weight', 'delay'], format='array')
exists = ~np.isnan(weights)
#filtered out the non-existent connections using numpy.isnan()
for w, d in zip(weights[exists].flat, delays[exists].flat)[:5]:
    print("weight = %.4f  delay = %4.2f" % (w, d))

#Projection.save() TO DO

#connections can find name after enter .
list(inhibitory_connections.connections)[0].weight
list(inhibitory_connections.connections)[10].i_group

#Modifying weights and delays
#set()
excitatory_connections.set(weight=0.02)
excitatory_connections.set(weight=RandomDistribution('gamma', [1, 0.1]),delay=0.3)
inhibitory_connections.set(weight=weight_distr)

#use connections
#however almost always less efficient than using list- or array-based access
for c in list(inhibitory_connections.connections)[:5]:
    c.weight *= 2
print(inhibitory_connections.get('weight', format='list', with_address=False)[3:7])
print("finish")
