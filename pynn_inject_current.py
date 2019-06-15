import pyNN.brian as sim 
import matplotlib.pyplot as plt
from pyNN.utility.plotting import Figure, Panel

sim.setup(timestep=0.1,min_delay=2.0) 
ifcell=sim.create(sim.IF_cond_exp,{'i_offset':0.11,'tau_refrac':3.0,'v_thresh':-51.0},n=10)
'''
pulse = sim.DCSource(amplitude=0.5, start=20.0, stop=80.0)
pulse.inject_into(ifcell[3:7])

ifcell.record('v')
sim.run(100.0)

data = ifcell.get_data()
sim.end()
'''
'''
vm = data.filter(name="v")[0]

Figure(
    Panel(vm, ylabel="Membrane potential (mV)")
).show
'''


'''
sine = sim.ACSource(start=50.0, stop=450.0, amplitude=1.0, offset=1.0,
                frequency=10.0, phase=180.0)
ifcell.inject(sine)
ifcell.record('v')
sim.run(500.0)

data = ifcell.get_data()
sim.end()

'''
'''
steps = sim.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                        amplitudes=[0.4, 0.6, -0.2, 0.2])
steps.inject_into(ifcell[(1,7,9)])
ifcell.record('v')
sim.run(250.0)

data = ifcell.get_data()
sim.end()

'''
noise = sim.NoisyCurrentSource(mean=1.5, stdev=1.0, start=50.0, stop=450.0, dt=1.0)
ifcell.inject(noise)
ifcell.record('v')
sim.run(500.0)

data = ifcell.get_data()
sim.end()


for segment in data.segments:
    vm = segment.analogsignals[0]
    print(type(vm))
    for i in range(vm.shape[1]):
        #v=vm[:,i]
        #print(v.shape)
        plt.plot(vm.times, vm[:,i],label=str(i))
#plt.plot(vm.times, pulse.amplitudes,label='current')
plt.legend(loc="upper left")
plt.xlabel("Time (%s)" % vm.times.units._dimensionality)
plt.ylabel("Membrane potential (%s)" % vm.units._dimensionality)
plt.show()