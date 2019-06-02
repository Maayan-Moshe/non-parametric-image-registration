# Non Parametric Image Registration 

We present here non-parametric image registration using [Horn–Schunck method](https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method).

##Introduction

In addition to the standard "Horn-Schunck" smoothing term which punishes large x and y displacement gradients separately 
we introduce an option to smooth according to the curl and\or the divergence of the displacement field.

My thought is that the only three terms which are:
* Q in the first spatial derivative of the displacement field (up to linear dependency).
* Positive definite.
* Invariant to rotations.

This is the most general smoothing term that one can write according to the above assumptions.

## Requirements 

1. Python 3.6
2. unittest
3. numpy
4. matplotlib.pyplot
5. scipy

## License
[MIT](https://choosealicense.com/licenses/mit/)