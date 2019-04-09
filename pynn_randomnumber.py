#import numpy
from pyNN.random import RandomDistribution, NumpyRNG, GSLRNG, NativeRNG
rng = NumpyRNG(seed=824756)
print(rng.next(5, 'normal', {'mu': 1.0, 'sigma': 0.2}))


'''
rng = GSLRNG(seed=824756, type='ranlxd2')  # RANLUX algorithm of Luescher
rng.next(5, 'normal', {'mu': 1.0, 'sigma': 0.2})
#fail to import 
#error: Cannot import pygsl
#need pygsl package, cannot be installed with pip
'''

gamma = RandomDistribution('gamma', (2.0, 0.3), rng=NumpyRNG(seed=72386))
print(gamma.next(5))
#by name
gamma = RandomDistribution('gamma', k=2.0, theta=0.3, rng=NumpyRNG(seed=72386))
#differece
print(gamma.next())
print(gamma.next(1))

norm=NativeRNG(seed=72386)
print(norm.next(5))