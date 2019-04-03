import pyNN.brian as sim  # can of course replace `nest` with `neuron`, `brian`, etc.

sim.setup()

def report_time(t):
     print("The time is %g" % t)
     return t + 100.0

sim.run_until(300.0, callbacks=[report_time])
sim.end()