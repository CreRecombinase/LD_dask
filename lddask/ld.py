import math

import numpy as np


# nmsum <- sum(1 / (1:(2*m-1)))
#  (1/nmsum) / (2*m + 1/nmsum)

def hmean(msize):
    ''' calculate the harmonic mean(?)'''
    ret = 0.0
    for i in range(1, msize + 1):
        ret += 1.0 / i
    return ret


def calc_theta(m):
    ''' calculate theta from reference panel size'''
    msize = (2 * m - 1)
    nmsum = hmean(msize)
    return (1 / nmsum) / (2 * m + 1 / nmsum)


def ld_dist(map_dist, ne, m, cutoff):
    ''' Compute the shrinkage factor'''
    rho = 4 * ne * (map_dist) / 100;
    rho = -rho / (2 * m);
    rho = math.exp(rho);
    return rho


def ld_var(var_1, theta, genomult):
    ''' Compute variance using ldshrink'''
    pre_mult = 0.5 * theta * (1 - 0.5 * theta)
    (var_1 * ((1 - theta) * (1 - theta) * genomult) + pre_mult)
    return var_1


def shrink_cov(x_1, x_2, rho, theta, GenoMult, var_1, var_2):
    ''' Calculate the ldshrinkage for a pair of variants '''
    Nm1 = x_1.size - 1
    r = ((x_1.dot(x_2) / Nm1) * rho * GenoMult * (1 - theta) * (1 - theta))
    r = r / (math.sqrt(var_1 * var_2))
    return r


def ldshrink(X, map_data, m, ne, cutoff):
    ''' compute ldshrink'''
    X = X - X.mean(axis=0, keepdims=True)
    n, p = X.shape
    genomult = 0.5
    S = np.eye(p, p)
    theta = calc_theta(m)
    var_i = np.square(X).sum(axis=0) / (n - 1)
    var_list = [ld_var(var_i[i], theta, genomult) for i in range(p)]
    for i in range(p):
        tX = X[:, i]
        tv = var_list[i]
        for j in range(i + 1, p):
            rho = ld_dist(map_data[j] - map_data[i], ne, m, cutoff)
            if (rho > cutoff):
                S[i, j] = shrink_cov(tX, X[:, j], rho, theta, genomult, tv, var_list[j])

    S = S + S.T - np.diag(S.diagonal())

    return S
