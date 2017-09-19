import numpy as np

from pdmse import *

txthd = '-'*30 + '\n* \n* '
txtft = '\n* \n' + '-'*30


def pprint(value, end='\n'):
    value = txthd + value + txtft
    print(value, end=end)
    return


"""
Maxima of normal random vectors
"""
pprint('Finding maxima of normal random vectors')

s = 5
n = 10000
k = 10

def compareMaxToUB(s, n, k):
    z = np.random.randn(n, k)
    z = np.sort(np.abs(z), axis=0)
    zT = np.sum(z[-s:, :], axis=0)
    ub = 4 * s * np.log(n)
    difference = ub - zT
    return np.sum(difference < 0)


print('ub - zT < 0 {} times.'.format(compareMaxToUB(5, 100000, 1000)))
