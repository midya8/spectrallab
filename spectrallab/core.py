"""Normalized spectral clustering. NumPy supplies symmetric eigenvalue solves, not clustering."""
from __future__ import annotations
import math
import numpy as np


def points_array(value):
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('expected at least three finite feature rows')
    return x


def graph(points, kind='union', neighbors=6, sigma=0.25):
    """RBF full graph or unweighted directed-kNN union/mutual, diagonal zero."""
    x = points_array(points)
    n = len(x)
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    if kind == 'rbf':
        if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('sigma must be finite and positive')
        w = np.exp(-distances / (2 * sigma * sigma))
    elif kind in ('union', 'mutual'):
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or not 1 <= neighbors < n:
            raise ValueError('neighbors must be an integer from 1 to n-1')
        directed = np.zeros((n, n), dtype=bool)
        for i in range(n):
            order = sorted((j for j in range(n) if j != i), key=lambda j: (float(distances[i, j]), j))
            directed[i, order[:neighbors]] = True
        w = (directed | directed.T if kind == 'union' else directed & directed.T).astype(float)
    else:
        raise ValueError('unknown graph kind')
    np.fill_diagonal(w, 0)
    return w


def matrix(value):
    w = np.asarray(value, dtype=float)
    if w.ndim != 2 or len(w) < 2 or w.shape[0] != w.shape[1] or not np.isfinite(w).all():
        raise ValueError('finite square affinity required')
    if (w < 0).any() or not np.array_equal(w, w.T) or np.any(np.diag(w) != 0):
        raise ValueError('affinity must be symmetric, nonnegative, zero diagonal')
    return w


def components(weights):
    w = matrix(weights)
    unseen = set(range(len(w)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, group = [start], []
        while stack:
            i = stack.pop()
            group.append(i)
            for j in sorted(unseen):
                if w[i, j] > 0:
                    unseen.remove(j)
                    stack.append(j)
        groups.append(sorted(group))
    return groups


def laplacian(weights):
    w = matrix(weights)
    degree = w.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError('isolated vertices have no normalized degree inverse')
    return np.eye(len(w)) - w / np.sqrt(degree[:, None] * degree[None, :])


def normalize_rows(vectors):
    u = np.asarray(vectors, dtype=float)
    if u.ndim != 2 or not np.isfinite(u).all():
        raise ValueError('finite matrix required')
    norm = np.linalg.norm(u, axis=1)
    if np.any(norm <= 1e-14):
        raise ValueError('zero embedding row: refusing an arbitrary assignment')
    return u / norm[:, None]


def canonical_labels(labels):
    lookup = {}
    return [lookup.setdefault(int(label), len(lookup)) for label in labels]


def kmeans(points, seed=20261108, restarts=12, max_iter=200):
    """Two clusters, uniform distinct-row starts, stable argmin ties; no label access."""
    x = points_array(points)
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError('positive integer restarts required')
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('positive integer iteration limit required')
    rng = np.random.default_rng(seed)
    runs = []
    for run in range(restarts):
        initial = rng.choice(len(x), size=2, replace=False).tolist()
        centers = x[initial].copy()
        history, previous = [], None
        for step in range(max_iter):
            d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d, axis=1)
            counts = np.bincount(labels, minlength=2)
            if np.any(counts == 0):
                break
            centers = np.array([x[labels == c].mean(axis=0) for c in range(2)])
            inertia = float(np.sum((x - centers[labels]) ** 2))
            history.append(inertia)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
        else:
            raise ValueError('k-means iteration limit')
        if np.any(counts == 0):
            runs.append({'run': run, 'initial': initial, 'status': 'empty-cluster', 'history': history})
            continue
        final_assignment = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        if not np.array_equal(final_assignment, labels):
            raise ValueError('k-means not at a fixed assignment')
        runs.append({'run': run, 'initial': initial, 'status': 'ok', 'history': history,
                     'inertia': inertia, 'labels': canonical_labels(labels)})
    valid = [r for r in runs if r['status'] == 'ok']
    if not valid:
        raise ValueError('all restarts have empty clusters')
    best = min(valid, key=lambda r: (r['inertia'], r['run']))
    return {'labels': best['labels'], 'inertia': best['inertia'], 'selected_restart': best['run'], 'restarts': runs}


def cut_score(weights, labels):
    w = matrix(weights)
    y = np.asarray(labels)
    if y.shape != (len(w),) or set(y.tolist()) != {0, 1}:
        raise ValueError('nonempty binary labels required')
    degree = w.sum(axis=1)
    volumes = [float(degree[y == c].sum()) for c in range(2)]
    if min(volumes) <= 0:
        raise ValueError('zero cluster volume')
    cut = float(w[np.ix_(y == 0, y == 1)].sum())
    return {'cut_weight': cut, 'volumes': volumes, 'normalized_cut': cut * (1 / volumes[0] + 1 / volumes[1])}


def ari(truth, labels):
    """Pair-count adjusted Rand index. Not used for fitting or selection."""
    if len(truth) != len(labels) or len(truth) < 2:
        raise ValueError('matching labels required')
    table = {}
    a, b = {}, {}
    for u, v in zip(truth, labels):
        table[u, v] = table.get((u, v), 0) + 1
        a[u] = a.get(u, 0) + 1
        b[v] = b.get(v, 0) + 1
    choose = lambda n: n * (n - 1) // 2
    pairs = choose(len(labels))
    joint, aa, bb = (sum(choose(n) for n in d.values()) for d in (table, a, b))
    expected = aa * bb / pairs
    denominator = (aa + bb) / 2 - expected
    return 1.0 if denominator == 0 else (joint - expected) / denominator


def spectral(weights, seed=20261108):
    w = matrix(weights)
    if len(w) < 3:
        raise ValueError('at least three vertices required for a two-cluster boundary witness')
    groups = components(w)
    degree = w.sum(axis=1)
    base = {'components': groups, 'degrees': degree.tolist(), 'edges': int(np.count_nonzero(w) // 2)}
    if np.any(degree <= 0):
        return dict(base, status='refused-isolated', reason='A vertex has zero degree; no normalized inverse is invented.')
    if len(groups) > 2:
        return dict(base, status='refused-components', reason='More than two connected components: a two-dimensional nullspace truncation is not unique.')
    lap = laplacian(w)
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    # Canonical signs are display-only. They do not remove repeated-eigenspace ambiguity.
    u = eigenvectors[:, :3].copy()
    for c in range(3):
        pivot = int(np.argmax(np.abs(u[:, c])))
        if u[pivot, c] < 0:
            u[:, c] *= -1
    gap = float(eigenvalues[2] - eigenvalues[1])
    residual = float(np.max(np.abs(lap @ u - u * eigenvalues[:3])))
    orthogonality = float(np.max(np.abs(u.T @ u - np.eye(3))))
    base.update(eigenvalues=eigenvalues.tolist(), eigenvectors=u.tolist(), eigengap=gap,
                max_residual=residual, max_orthogonality=orthogonality)
    if gap <= 1e-10:
        return dict(base, status='refused-boundary-tie', reason='The second/third eigenvalue boundary is numerically tied; no unique selected subspace.')
    embedding = normalize_rows(u[:, :2])
    fit = kmeans(embedding, seed=seed)
    return dict(base, status='ok', embedding=embedding.tolist(), fit=fit,
                cut=cut_score(w, fit['labels']))
