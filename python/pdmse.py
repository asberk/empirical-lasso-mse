# #
# Functions to compute the proximal denoising mean-squared error in batch
# n-plicate for different (log-scaled) signal lengths.
# #

# from spgl1 import spgSetParms
import numpy as np
import jsonWrite
try:
    from spgl1 import spgl1
except ImportError as ie:
    print('trying alternative import for spgl1...', end='')
    import addToPath
    addToPath.spgl1()
    try:
        from spgl1 import spgl1
        print('success!')
    except Exception as e2:
        print()
        print(e2)


# # # # #
# Helper methods

def softThresh(y, lam=None):
    if lam is None or lam == 0:
        return y
    elif lam < 0:
        raise ValueError('lam must be greater than zero')
    return np.sign(y) * np.maximum(0, np.abs(y)-lam)


def unsort(increasingVector, sortOrder):
    unsortedVector = np.zeros(increasingVector.size)
    for idx, orderedIdx in enumerate(sortOrder):
        unsortedVector[orderedIdx] = increasingVector[idx]
    return unsortedVector


# # # # #
# Methods to recover solutions to the proximal denoising problem

def pd_golden_tau(y, tau, **kwargs):
    """
    pd_golden_tau returns the solution xstar to the proximal denoising problem
    xstar = argmin( norm(x-y, 2) s.t. norm(x, 1) ≤ tau )
    where y = x + eta*z, z ~ N(0, I) is a Gaussian random vector. This
    function uses the interval shortening method that converges with
    rate equal to the golden ratio.
    Input:
    tau: threshold value for the one-norm of x.
    y: assume y is already ravelled to an np vector (y = y.ravel())
    returnLambda: whether to also return the threshold value.
    """
    optTol = kwargs.get('optTol', 1e-6)
    maxIters = kwargs.get('maxIters', 500)
    verbose = kwargs.get('verbose', False)

    y_ell1 = np.linalg.norm(y, 1)

    if tau is None:
        # Try filthy heuristic
        sortOrder = np.argsort(np.abs(y))
        ySorted = y[sortOrder]
        biggestDiff = np.argmax(np.diff(np.abs(ySorted)))
        # eta_estimate = ySorted[:biggestDiff].std()
        # tau_estimate = np.linalg.norm(ySorted[biggestDiff:], 1)
        tau = tau_estimate
    elif tau <= 0:
        return np.zeros(y.shape)
    elif tau >= y_ell1:
        return y
    else:
        if not (tau > 0 and tau < y_ell1):
            raise ValueError('tau must be a number' +
                             ' in the range (0, norm(y,1))')

    lam_max = np.max(np.abs(y))
    lam_min = 0
    lam_j = lam_max/2
    iters = 0
    while True:
        y_lam_j = softThresh(y, lam_j)
        y_ell1_j = np.linalg.norm(y_lam_j, 1)
        iters += 1
        if np.abs(y_ell1_j - tau) < optTol:
            if verbose:
                print('Convergence attained.')
            break
        elif iters > maxIters:
            if verbose:
                print('maxIters reached; convergence not attained.')
            break
        if y_ell1_j > tau:
            lam_min = lam_j
            lam_j = (lam_min + lam_max)/2
        else:  # y_ell1_j < tau
            lam_max = lam_j
            lam_j = (lam_min + lam_max)/2
    return y_lam_j


def pd_ls_homotopy(y, tau, **kwargs):
    """
    pd_ls_homotopy(y, tau, **kwargs) is a proximal denoising solver
    for the LS-tau problem using the homotopy method.
    
    Input:
    y : the input signal x + noise
    tau : the thresholding parameter in the minimization problem 
          x_hat := argmin_w { norm(y-w, 2) : norm(w, 1) <= tau }
    returnLambda : default: false
    """
    returnLambda = kwargs.get('returnLambda', False)

    y_ell1 = np.linalg.norm(y, 1)
    if tau is None:
        raise ValueError('Enter a value for tau in the range [0, norm(y, 1)]')
    elif tau <= 0:
        return np.zeros(y.shape)
    elif tau >= y_ell1:
        return y
    else:
        if not (tau > 0 and tau < y_ell1):
            raise ValueError('tau must be a number' +
                             ' in the range (0, norm(y, 1))')
    n = y.size
    lam = np.insert(np.sort(np.abs(y)), 0, 0)

    S = np.zeros(lam.size)
    S[0] = y_ell1

    for j in range(1, n+1):
        S[j] = S[j-1] - (n-(j-1))*(lam[j]-lam[j-1])
        if S[j] <= tau:
            lam_star = (S[j-1] - tau)/(n-(j-1)) + lam[j-1]
            if returnLambda:
                return (softThresh(y, lam_star), lam_star)
            else:
                return softThresh(y, lam_star)
    if returnLambda:
        return (np.zeros(n), ySorted[-1])
    else:
        return np.zeros(n)


def pd_homotopy(y, sigmaSquared=None, **kwargs):
    """
    pd_homotopy returns the solution xstar to the proximal denoising problem
    xstar = argmin( norm(x,1) s.t. norm(x-y, 2) ≤ sigma )
    where y = x + eta*z, z ~ N(0, I) is a Gaussian random vector
    Input:
    sigmaSquared: threshold value; sigmaSquared = sigma**2 where
                  sigma is as above. If sigmaSquared is None
                  try to find sigma in kwargs. If sigma is not a
                  kwarg then set sigma = sqrt(n) where n = y.size.
    y: assume y is already ravelled to an np vector!
    returnLambda: whether to also return the threshold value
                  (default: False)
    """
    returnLambda = kwargs.get('returnLambda', False)

    if sigmaSquared is None:
        sigma = kwargs.get('sigma', np.sqrt(y.size))
        if sigma is None:
            sigma = np.sqrt(y.size)
        sigmaSquared = y.size
    else:
        sigma = np.sqrt(sigmaSquared)

    if sigma > np.linalg.norm(y):
        if returnLambda:
            return (np.zeros(y.size), np.abs(y).max())
        else:
            return np.zeros(y.size)
    y_shape = y.size  # store shape before ravel
    y_sgn = np.sign(y)  # store sign for later
    y = np.abs(y)  # absolute value
    sortOrder = np.argsort(y)  # store seq for ++ing order
    y = y[sortOrder]  # sort y in ++ing order
    n = y.size  # num elements in y

    lam = np.insert(y, 0, 0)  # λ_j values
    S = np.zeros(lam.size)  # S_j values
    f = np.zeros(lam.size)  # f_j values

    for j in range(1, n+1):
        S[j] = S[j-1] + lam[j-1]**2  # previously killed
        f[j] = (n - (j-1))*lam[j]**2 + S[j]  # still active
        if sigmaSquared < f[j]:  # if σ^2 is in I_j = [f[j-1], f[j]]
            lam_star = np.sqrt((sigmaSquared - S[j])/(n - (j-1)))
            xHat = np.zeros(n)
            xHat[j-1:] = (y[j-1:] - lam_star) * y_sgn[sortOrder][j-1:]
            if returnLambda:
                return (unsort(xHat, sortOrder).reshape(y_shape), lam_star)
            else:
                return unsort(xHat, sortOrder).reshape(y_shape)
    raise Exception('Could not find interval in which sigmaSquared lies')


# # # # #
# Methods to compute MSE of proximal denoising solution

def pdmse_spgl1(N, theta=None, **kwargs):
    """
    pdmse_spgl1 computes the proximal denoising mean-squared error for an x of
    the form
    x = np.array([N for _ in range(s)] + [0 for _ in range(N-s)]),
    using the spgl1 package.
    Input:
    N : the dimension of the signal x and noise z to be constructed. Also the
        magnitude of the non-zero entries of x.
    theta : the threshold value (default None uses theta=np.sqrt(x.size);
            'sqNormZ' uses theta=norm(z,2); other values must be numeric)
    s : number of non-zero elements of x (all of size N)
    eta : the standard deviation of the normal-random noise
    spgParms : parameters to be passed to spgl1 (cf. spgSetParms)
    """
    s = kwargs.get('s', 1)
    eta = kwargs.get('eta', 1)
    spgParms = kwargs.get('spgParms', [])
    x = np.zeros(N)
    x[-s:] = N
    z = np.random.randn(N)
    if theta is None:
        theta = np.sqrt(N)
    elif theta is 'sqNormZ':
        theta = np.linalg.norm(z)
    elif isinstance(theta, str):
        raise ValueError('theta must be numeric or equal to \'sqNormZ\'.')
    y = x + eta*z
    xstar = spgl1(np.eye(N), y, sigma=theta, options=spgParms)[0]
    return np.linalg.norm(x - xstar)**2


def pdmse_golden_tau(N, theta=None, **kwargs):
    s = kwargs.get('s', 1)
    eta = kwargs.get('eta', 1)
    x = np.zeros(N)
    x[-s:] = N
    z = np.random.randn(N)
    if theta is None:
        tau = np.linalg.norm(x, 1)
    else:
        tau = theta
    y = x + eta*z
    xstar = pd_golden_tau(y, tau)
    return np.linalg.norm(x - xstar)**2
    

def pdmse_homotopy(N, theta=None, **kwargs):
    """
    pdmse_homotopy uses the "homotopy method" for computing the exact solution
    to the proximal denoising problem
    xstar = argmin norm(x, 1) s.t. norm(x - y, 2) ≤ sigma
    where y = x + eta*z and z ~ N(0, I) is an N dimensional Gaussian random
    vector.

    Input:
    N : the dimension of the signal x and noise z to be constructed. Also the
        magnitude of the non-zero entries of x.
    theta : the threshold value (default None uses theta=np.sqrt(x.size);
            'sqNormZ' uses theta=norm(z,2); other values must be numeric)
    s : number of non-zero elements of x (all of size N)
    eta : the standard deviation of the normal-random noise
    """
    # Use cython where possible...
    s = kwargs.get('s', 1)
    eta = kwargs.get('eta', 1)
    z = np.random.randn(N)
    if theta is None:
        theta = np.sqrt(N)
    elif theta is 'sqNormZ':
        theta = np.linalg.norm(z)
    elif isinstance(theta, 'str'):
        raise ValueError('theta must be numeric or equal to \'sqNormZ\'.')
    x = np.zeros(N)
    x[-s:] = N
    y = x + eta*z
    xstar = pd_homotopy(y, theta**2)
    return np.linalg.norm(x - xstar)**2


# # # # #
# Method to compute mse in batch and return mean

def pdmse(N, theta=None, **kwargs):
    """
    pdmse uses func to compute several mean-squared errors for the same
    problem set-up (but with different normal random vectors z) and returns the
    mean of the result.
    Input:
    N : the dimension of the signal x and noise z to be constructed. Also the
        magnitude of the non-zero entries of x.
    theta : the threshold value (default None uses theta=np.sqrt(x.size);
            'sqNormZ' uses theta=norm(z,2); other values must be numeric)
    iters : the number of inner iterations to compute (controls size of vector
            whose mean is returned in pdmse)
    s : number of non-zero elements of x (all of size N)
    eta : the standard deviation of the normal-random noise
    spgParms : parameters to be passed to spgl1 (cf. spgSetParms)
    """
    iters = kwargs.pop('iters', 10)
    func = kwargs.pop('func', pdmse_homotopy)
    return np.mean([func(N, theta, **kwargs) for _ in range(iters)])


# # # # #
# computes a batch of mean-MSEs

def pdmse_batch(logNmax=5, theta=None, **kwargs):
    """
    pdmse_batch uses pdmse to compute the mean-squared error over a range of
    signal lengths [10**1, 10**2, ..., 10**logNmax] and returns the result as a
    matrix of shape (logNmax, outerIters)

    Input:
    eta : the standard deviation of the normal-random noise
    logNmax : the log (base 10) of the largest signal length on which to compute
    iters : the number of inner iterations to compute (controls size of vector
            whose mean is returned in pdmse)
    outerIters : the number of outer iterations to compute (controls number of
                 columns of output matrix)
    s : number of non-zero elements of x (all of size N)
    spgParms : parameters to be passed to spgl1 (cf. spgSetParms)
    theta : the threshold value (default None uses theta=np.sqrt(x.size);
            'sqNormZ' uses theta=norm(z,2); other values must be numeric)
    verbose : if True print progress to sys.stdout; if string, print to file
              whose name is that string.
    """
    outerIters = kwargs.pop('outerIters', 100)

    verbose = kwargs.pop('verbose', False)
    if verbose is True:
        import sys
        logFile = sys.stdout
        fp = open(logFile, 'w', encoding='utf-8')
    elif isinstance(verbose, str):
        logFile = verbose
        with open(logFile, 'w', encoding='utf-8') as fp:
            fp.write('Initializing algorithm.\n')
        verbose = True

    Nvec = [10**j for j in range(1, logNmax+1)]
    retmat = np.zeros((logNmax, outerIters))

    for i, N in enumerate(Nvec):
        if verbose:
            with open(logFile, 'a', encoding='utf-8') as fp:
                print('\n\nStarting outer iteration sequence {} of {}.'.format(i+1, logNmax), file=fp)
        for j in range(outerIters):
            retmat[i, j] = pdmse(N, theta, **kwargs)
            if verbose and (np.mod(j+1, 10) == 0):
                with open(logFile, 'a', encoding='utf-8') as fp:
                    print('\tfinished outer iter {} of {}.'.format(j+1, outerIters), file=fp)
        if verbose:
            with open(logFile, 'a', encoding='utf-8') as fp:
                fp.write('\n')
            jsonWrite.array(logFile, retmat[i, :])
    return retmat
