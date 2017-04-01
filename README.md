# Empirical Lasso MSE

## Summary

Contains code for computing empirically the mean-squared error of Lasso
methods.

## `python`

1. Algorithms for solving ell1 problems

iPython notebook that contains an AMP algorithm, and some other methods.

2. Empirical Lasso MSE

iPython notebook that uses `spgl1` to compute the empirical MSE for several
parameter settings in the underdetermined normal-random case.

3. jsonWriteDict

writes a dictionary containing numpy arrays to an easy-to-read json file. 

4. pdmse

contains methods for computing the mean-squared error of proximal denoising
solutions. 

5. ProxDenoisSPGL1HistData

contains code that uses both pdmse and jsonWriteDict to compute the
mean-squared error in the proximal denoising case a large number of times for
several signal dimensions; stores the result in a json file. 


