import numpy as np

from pdmse import pd_homotopy

def bppdStats(n=10, k=100, **kwargs):
    eta = kwargs.get('eta', 1)
    random_state = kwargs.get('random_state', None)
    theta = kwargs.get('theta', np.sqrt(n))
    s = kwargs.get('s', 1)
    if random_state:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    # Output matrices
    Z = np.zeros((n,k))
    Q = np.zeros((n,k))
    # Set-up
    x = np.zeros(n)
    x[-s:] = n
    j = 0
    while j < k:
        z = rng.randn(n)
        if np.dot(z,z) >= theta**2 + theta:
            y = x + eta*z
            xhat = pd_homotopy(y, sigmaSquared=theta**2)
            q = xhat - y
            Z[:, j] = z
            Q[:, j] = q
            j += 1
    return (Z, Q)
