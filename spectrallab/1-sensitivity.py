"""Fixed-edge graph sensitivity. Manual derivatives; NumPy supplies symmetric eigensolves.

The returned projector derivative is not a derivative of discrete cluster labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import matrix, laplacian, spectral, cut_score, normalize_rows, kmeans

GAP_TOL = 1e-10
STEPS = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 25.0]


def symmetric(value, shape=None):
    a = np.asarray(value, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 3
            or not np.isfinite(a).all() or not np.array_equal(a, a.T)
            or (shape is not None and a.shape != shape)):
        raise ValueError('finite symmetric square matrices of matching size >= 3 required')
    return a


def edge_direction(size, a, b):
    if (type(size) is not int or size < 3 or type(a) is not int or type(b) is not int
            or not 0 <= a < size or not 0 <= b < size or a == b):
        raise ValueError('two distinct integer vertex indices required')
    e = np.zeros((size, size)); e[a, b] = e[b, a] = 1.0
    return e


def laplacian_tangent(weights, direction):
    """d(I-D^-1/2 W D^-1/2), including both degree-normalization terms."""
    w = matrix(weights)
    e = symmetric(direction, w.shape)
    if np.any(np.diag(e) != 0):
        raise ValueError('zero diagonal direction required')
    d = w.sum(axis=1)
    if np.any(d <= 0):
        raise ValueError('isolated vertex: no normalized derivative')
    s = e.sum(axis=1)
    normalized = w / np.sqrt(d[:, None] * d[None, :])
    response = (-e / np.sqrt(d[:, None] * d[None, :])
                + 0.5 * normalized * (s[:, None] / d[:, None] + s[None, :] / d[None, :]))
    return response


def projector_tangent(operator, direction, rank=2):
    """Bottom-rank orthogonal projector derivative; only cross-boundary denominators.

    Internal repeated eigenvalues are allowed. A tied retained boundary is refused.
    A symmetric operator and an unconstrained symmetric tangent define this local
    derivative even at a zero graph edge; graph-valid changes there are one-sided.
    """
    a = symmetric(operator)
    e = symmetric(direction, a.shape)
    if type(rank) is not int or not 1 <= rank < len(a):
        raise ValueError('rank must be an integer from 1 to n-1')
    values, q = np.linalg.eigh(a)
    gap = float(values[rank] - values[rank-1])
    if gap <= GAP_TOL:
        raise ValueError('retained boundary gap <= 1e-10: projector derivative refused')
    selected, other = q[:, :rank], q[:, rank:]
    coupling = other.T @ e @ selected
    coefficients = coupling / (values[:rank][None, :] - values[rank:][:, None])
    cross = other @ coefficients @ selected.T
    p = selected @ selected.T
    dp = cross + cross.T
    return {'eigenvalues': values.tolist(), 'eigenvectors': q.tolist(), 'eigengap': gap,
            'projector': p.tolist(), 'projector_tangent': dp.tolist(),
            'cross_coefficients': coefficients.tolist(),
            'tangent_frobenius': float(np.linalg.norm(dp)),
            'commutator_residual': float(np.max(np.abs(a @ dp-dp @ a-p @ e+e @ p))),
            'idempotence_derivative_residual': float(np.max(np.abs(p @ dp+dp @ p-dp))),
            'trace_derivative': float(np.trace(dp))}


def fixed_cut_tangent(weights, direction, labels):
    """Derivative of this fixed partition's normalized cut, not a reoptimized cut."""
    w = matrix(weights); e = symmetric(direction, w.shape)
    if np.any(np.diag(e) != 0): raise ValueError('zero diagonal direction required')
    base = cut_score(w, labels)
    y = np.asarray(labels)
    dc = float(e[np.ix_(y == 0, y == 1)].sum())
    dv = [float(e.sum(axis=1)[y == c].sum()) for c in (0, 1)]
    c, v = base['cut_weight'], base['volumes']
    derivative = dc * (1/v[0]+1/v[1]) - c * (dv[0]/v[0]**2+dv[1]/v[1]**2)
    return {'cut_weight_derivative': dc, 'volume_derivatives': dv, 'normalized_cut_derivative': derivative}


def fixtures():
    # Original integer graph, shared with the v0.1 rounding witness.
    rounding = [[0,7,4,3,4,0,2],[7,0,3,1,0,1,5],[4,3,0,6,3,1,4],
                [3,1,6,0,6,7,6],[4,0,3,6,0,5,5],[0,1,1,7,5,0,2],[2,5,4,6,5,2,0]]
    triangles = np.zeros((6, 6))
    for group in ([0,1,2], [3,4,5]):
        for i in group:
            for j in group:
                if i != j: triangles[i,j] = 1
    weak = triangles.copy(); weak[0,3] = weak[3,0] = 0.04
    path = np.zeros((5,5))
    for i in range(4): path[i,i+1] = path[i+1,i] = 1
    cycle = np.zeros((6,6))
    for i in range(6): cycle[i,(i+1)%6] = cycle[(i+1)%6,i] = 1
    isolated = triangles.copy(); isolated[5,:] = isolated[:,5] = 0
    return [('rounding', 'Seven-vertex rounding graph / add missing edge 0-5', rounding, (0,5)),
            ('weak-bridge', 'Two triangles / strengthen the 0.04 bridge 0-3', weak.tolist(), (0,3)),
            ('two-components', 'Two disconnected triangles / add bridge 0-3', triangles.tolist(), (0,3)),
            ('path', 'Five-vertex path / add closing edge 0-4', path.tolist(), (0,4)),
            ('boundary-tie', 'Six-cycle / tied second-third boundary', cycle.tolist(), (0,3)),
            ('isolated', 'Isolated vertex / no normalized inverse', isolated.tolist(), (0,5))]


def response(weights, edge, steps=STEPS):
    w = matrix(weights)
    e = edge_direction(len(w), *edge)
    if (not isinstance(steps, (list, tuple)) or not steps or len(steps) > 128
            or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t)
                   or abs(t) > 1e6 for t in steps)):
        raise ValueError('1-128 finite bounded steps required')
    if any(np.any(w + t*e < 0) for t in steps):
        raise ValueError('step makes an edge negative')
    result = {'weights': w.tolist(), 'edge': list(edge), 'steps': list(steps),
              'base_edge_weight': float(w[tuple(edge)]), 'base_fit': spectral(w)}
    try:
        dl = laplacian_tangent(w, e)
        witness = projector_tangent(laplacian(w), dl)
    except ValueError as error:
        return dict(result, status='refused', reason=str(error), states=[])
    p = np.array(witness['projector']); dp = np.array(witness['projector_tangent'])
    labels = result['base_fit']['fit']['labels']
    dc = fixed_cut_tangent(w, e, labels)
    fixed_base = cut_score(w, labels)['normalized_cut']
    states = []
    for t in steps:
        changed = w + t*e
        fit = spectral(changed)
        state = {'step': t, 'edge_weight': float(changed[tuple(edge)]), 'fit': fit,
                 'fixed_cut': cut_score(changed, labels)['normalized_cut'],
                 'fixed_cut_linear': fixed_base+t*dc['normalized_cut_derivative']}
        try:
            now = projector_tangent(laplacian(changed), np.zeros_like(w))
            pt = np.array(now['projector'])
            state.update(projector=pt.tolist(), projector_distance=float(np.linalg.norm(pt-p)),
                         linear_projector_distance=float(abs(t)*np.linalg.norm(dp)),
                         linearization_error=float(np.linalg.norm(pt-p-t*dp)))
        except ValueError as error:
            state.update(projector=None, projector_distance=None, linear_projector_distance=float(abs(t)*np.linalg.norm(dp)),
                         linearization_error=None, projector_refusal=str(error))
        if fit['status'] == 'ok':
            mismatches = sum(a != b for a,b in zip(labels, fit['fit']['labels']))
            state['changed_labels_up_to_swap'] = min(mismatches, len(w)-mismatches)
        else:
            state['changed_labels_up_to_swap'] = None
        states.append(state)
    return dict(result, status='ok', laplacian_tangent=dl.tolist(), witness=witness,
                fixed_cut_tangent=dc, states=states)


def build():
    return {'schema': 'spectrallab-sensitivity-1', 'scope': 'Fixed graph-weight path, not coordinate or graph-rule differentiation, label derivative, confidence or finite-step guarantee.',
            'cases': [dict(response(w, edge), id=name, title=title) for name,title,w,edge in fixtures()]}


def validate_fit(actual, weights):
    """Validate archived geometry and replay rounding on the archived embedding.

    A fresh eigensolve can rotate a repeated eigenspace or round a tied k-means
    comparison differently. Do not require fresh labels to resolve those ties.
    """
    from .report import equivalent, close
    expected = spectral(weights)
    if set(actual) != set(expected): raise ValueError('fit fields')
    for key in ('status', 'components', 'edges'):
        equivalent(actual[key], expected[key], key)
    close(actual['degrees'], expected['degrees'], 'degrees')
    if 'eigenvalues' not in actual:
        equivalent(actual['reason'], expected['reason'], 'reason')
        return
    lap = laplacian(weights); q = np.asarray(actual['eigenvectors']); values = np.asarray(actual['eigenvalues'])
    if q.shape != (len(lap), 3): raise ValueError('fit eigenvector shape')
    close(values, expected['eigenvalues'], 'fit spectrum')
    close(q.T @ q, np.eye(3), 'fit orthonormality', atol=2e-9)
    close(lap @ q, q*values[:3], 'fit eigenpair equations', atol=2e-9)
    close(actual['eigengap'], values[2]-values[1], 'fit gap')
    close(actual['max_residual'], np.max(np.abs(lap @ q-q*values[:3])), 'fit residual')
    close(actual['max_orthogonality'], np.max(np.abs(q.T @ q-np.eye(3))), 'fit orthogonality')
    if actual['status'] != 'ok':
        equivalent(actual['reason'], expected['reason'], 'reason'); return
    v = np.asarray(expected['eigenvectors'])[:,:2]
    close(q[:,:2] @ q[:,:2].T, v @ v.T, 'fit retained projector', atol=2e-9)
    embedding = np.asarray(actual['embedding'])
    close(embedding, normalize_rows(q[:,:2]), 'archived normalized rows')
    replay = kmeans(embedding)
    equivalent(actual['fit']['restarts'], replay['restarts'], 'archived-embedding restart replay')
    fit = actual['fit']
    if set(fit) != set(expected['fit']) or any(type(x) is not int for x in fit['labels']):
        raise ValueError('fit rounding fields/labels')
    runs = [r for r in fit['restarts'] if r['status']=='ok']
    selected = min(runs, key=lambda r:(r['inertia'], r['run']))
    if fit['selected_restart'] != selected['run'] or fit['labels'] != selected['labels']:
        raise ValueError('archived least-inertia selection')
    close(fit['inertia'], selected['inertia'], 'selected inertia')
    close(fit['inertia'], replay['inertia'], 'replayed inertia')
    equivalent(actual['cut'], cut_score(weights, fit['labels']), 'actual fitted cut')


def audit(report):
    """Replay every path; validate eigenspaces, not arbitrary fresh tie choices."""
    from .report import equivalent, close
    expected = build()
    if not isinstance(report, dict) or set(report) != set(expected): raise ValueError('report fields')
    equivalent(report['schema'], expected['schema']); equivalent(report['scope'], expected['scope'])
    if not isinstance(report['cases'], list) or len(report['cases']) != len(expected['cases']): raise ValueError('case count')
    for actual, reference in zip(report['cases'], expected['cases']):
        a, b = json.loads(json.dumps(actual)), json.loads(json.dumps(reference))
        validate_fit(a['base_fit'], a['weights'])
        del a['base_fit']; del b['base_fit']
        if reference['status'] == 'ok':
            q = np.asarray(a['witness']['eigenvectors']); vals = np.asarray(a['witness']['eigenvalues'])
            lap = laplacian(a['weights']); n = len(lap)
            if q.shape != (n,n): raise ValueError('full basis shape')
            close(q.T @ q, np.eye(n), 'orthonormal full basis', atol=2e-9)
            close(lap @ q, q*vals, 'full eigenpair equations', atol=2e-9)
            close(q[:,:2] @ q[:,:2].T, a['witness']['projector'], 'retained projector', atol=2e-9)
            del a['witness']['eigenvectors']; del b['witness']['eigenvectors']
            coeff = np.asarray(a['witness']['cross_coefficients'])
            reconstructed = q[:,2:] @ coeff @ q[:,:2].T
            close(reconstructed+reconstructed.T, a['witness']['projector_tangent'], 'cross contraction', atol=2e-9)
            del a['witness']['cross_coefficients']; del b['witness']['cross_coefficients']
            labels = actual['base_fit']['fit']['labels']
            e = edge_direction(n,*a['edge'])
            equivalent(a['fixed_cut_tangent'], fixed_cut_tangent(a['weights'],e,labels), 'fixed-cut derivative')
            del a['fixed_cut_tangent']; del b['fixed_cut_tangent']
            if len(a['states']) != len(b['states']): raise ValueError('state count')
            for state, target in zip(a['states'], b['states']):
                weights = np.asarray(a['weights'])+state['step']*e
                validate_fit(state['fit'], weights)
                fit = state['fit']
                changed = None
                if fit['status']=='ok':
                    mismatch = sum(x!=y for x,y in zip(labels,fit['fit']['labels']))
                    changed = min(mismatch,n-mismatch)
                if state['changed_labels_up_to_swap'] != changed: raise ValueError('changed labels')
                del state['changed_labels_up_to_swap']; del target['changed_labels_up_to_swap']
                close(state['fixed_cut'], cut_score(weights,labels)['normalized_cut'], 'fixed cut')
                derivative = actual['fixed_cut_tangent']['normalized_cut_derivative']
                close(state['fixed_cut_linear'], cut_score(a['weights'],labels)['normalized_cut']+state['step']*derivative, 'fixed cut linear')
                for key in ('fit','fixed_cut','fixed_cut_linear'):
                    del state[key]; del target[key]
        equivalent(a, b, 'sensitivity replay')
    return {'cases': 6, 'smooth_cases': 4, 'refusals': 2, 'finite_states': 28}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'audit', 'export'])
    parser.add_argument('path')
    args = parser.parse_args()
    try:
        if args.command == 'build':
            data = build(); Path(args.path).write_text(json.dumps(data, separators=(',', ':'), allow_nan=False)+'\n')
        else:
            def pairs(items):
                out = {}
                for k,v in items:
                    if k in out: raise ValueError('duplicate key')
                    out[k] = v
                return out
            data = json.loads(Path(args.path).read_text(), object_pairs_hook=pairs,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        checked = audit(data)
        if args.command == 'export':
            print('globalThis.SPECTRAL_SENSITIVITY='+json.dumps(data,separators=(',', ':'),allow_nan=False)+';')
        else: print(json.dumps(checked))
    except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
        parser.exit(2, f'Invalid sensitivity request: {error}\n')

if __name__ == '__main__': main()
