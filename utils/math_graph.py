import os
import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from scipy.sparse.csgraph import laplacian
from geopy import distance

def scaled_laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :return: np.matrix, [n_route, n_route].
    '''
    L = laplacian(W, normed=True)
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(np.shape(W)[0]))


def cheb_poly_approx(L, Ks, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(
            f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def first_approx(W, n):
    '''
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    '''
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def create_weight_matrix(loc_file, locs = 'all', threshold = np.inf):
    '''
    Create a weight matrix from a json file containing lat, long of all places.
    :param loc_file: dict, key="place name" value=(lat, long).
    :param locs: str|list, list of places to choose. If 'all', create matrix with all stations.
    Order of locs is important.
    :param threshold: float, distance threshold beyond which co-relation is neglected.
    :return: np.array, 2D weight matrix.
    '''
    with open(os.path.join('dataset', loc_file)) as f:
        coors = json.load(f)

    if isinstance(locs, str):
        if locs == 'all':
            sel_coors = list(coors.values())
        else:
            raise NotImplementedError('ERROR: locs should either be "all" or a list of locations')
    elif isinstance(locs, list):
        sel_coors = [coors[loc] for loc in locs]
    else:
        raise NotImplementedError('ERROR: locs should either be "all" or a list of locations')
    
    distances = []
    for i in sel_coors:
        dist = []
        for j in sel_coors:
            val = np.inf
            if i != j:
                val = distance.distance(i, j).km
                if val > threshold:
                    val = np.inf
            dist.append(val)
        distances.append(dist)

    W = np.array(distances)
    dist_pairs = np.setdiff1d(np.ravel(np.tril(W, k=-1)), [0, np.inf])
    sigma = np.std(dist_pairs)
    W = (W/sigma)**2
    return np.exp(-W)
    # return np.exp(-W) * (np.exp(-W) >= epsilon)