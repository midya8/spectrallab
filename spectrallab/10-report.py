import hashlib
import json
import math
import numpy as np
from .core import graph, spectral, kmeans, ari, cut_score, laplacian, normalize_rows, components
from .data import datasets, CONFIGS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build():
    result = {'schema': 'spectrallab-1', 'seed': 20261108,
              'scope': 'Original synthetic full-data graph clustering, not held-out prediction. NumPy eigh supplies eigensolves.',
              'configurations': CONFIGS, 'datasets': {}}
    for name, data in datasets().items():
        x, truth = data['points'], data['truth']
        raw = kmeans(x)
        raw['ari'] = ari(truth, raw['labels'])
        entry = dict(data, raw=raw, fits=[])
        for config in CONFIGS:
            w = graph(x, **{k: v for k, v in config.items() if k != 'id'})
            fit = spectral(w)
            fit.update(id=config['id'], graph_sha256=digest({'points': x, 'configuration': config}), raw_cut=cut_score(w, raw['labels']))
            if fit['status'] == 'ok':
                fit['ari'] = ari(truth, fit['fit']['labels'])
                fit['truth_cut'] = cut_score(w, truth)
            entry['fits'].append(fit)
        entry['points_sha256'] = digest(x)
        result['datasets'][name] = entry
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message, atol=2e-9):
    require(np.asarray(a).shape == np.asarray(b).shape and np.allclose(a, b, rtol=atol, atol=atol), message)


def equivalent(actual, expected, where='value'):
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, where)
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, where)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and not isinstance(actual, bool) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10), where)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), where)
        for i, (a, e) in enumerate(zip(actual, expected)): equivalent(a, e, f'{where}/{i}')
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), where)
        for key, value in expected.items(): equivalent(actual[key], value, f'{where}/{key}')
    else: raise ValueError('unsupported reference type')


def audit(report):
    """Bind every source/configuration and independently recheck archived numerical witnesses."""
    require(report.get('schema') == 'spectrallab-1' and report.get('seed') == 20261108, 'schema/seed')
    require(report.get('configurations') == CONFIGS, 'fixed configurations')
    original = datasets()
    require(set(report['datasets']) == set(original), 'datasets')
    checked = {'graphs': 0, 'eigenvectors': 0, 'restarts': 0, 'points': 0}
    for name, entry in report['datasets'].items():
        close(entry['points'], original[name]['points'], 'original generator coordinates', atol=2e-14)
        require(entry['truth'] == original[name]['truth'], 'original generator labels')
        require(entry['seed'] == 20261108 and entry['points_sha256'] == digest(entry['points']), 'point digest')
        checked['points'] += len(entry['points'])
        raw = kmeans(entry['points'])
        require(entry['raw']['labels'] == raw['labels'], 'raw labels')
        close(entry['raw']['inertia'], raw['inertia'], 'raw inertia')
        close(entry['raw']['ari'], ari(entry['truth'], raw['labels']), 'raw ari')
        equivalent(entry['raw']['restarts'], raw['restarts'], 'raw restart replay')
        require(len(entry['fits']) == len(CONFIGS), 'fit count')
        for config, fit in zip(CONFIGS, entry['fits']):
            w = graph(entry['points'], **{k: v for k, v in config.items() if k != 'id'})
            require(fit['id'] == config['id'], 'graph identity')
            require(fit['graph_sha256'] == digest({'points': entry['points'], 'configuration': config}), 'graph input digest')
            require(fit['components'] == components(w), 'components')
            require(fit['edges'] == int(np.count_nonzero(w) // 2), 'edges')
            close(fit['degrees'], w.sum(axis=1), 'degrees')
            for key, value in cut_score(w, raw['labels']).items():
                close(fit['raw_cut'][key], value, 'raw cut')
            expected = spectral(w)
            require(fit['status'] == expected['status'], 'status')
            checked['graphs'] += 1
            if 'eigenvalues' not in fit:
                continue
            lap = laplacian(w)
            values, u = np.asarray(fit['eigenvalues']), np.asarray(fit['eigenvectors'])
            require(values.shape == (len(w),) and u.shape == (len(w), 3) and np.isfinite(values).all() and np.isfinite(u).all(), 'eigen shapes')
            close(values, expected['eigenvalues'], 'spectrum')
            residual = float(np.max(np.abs(lap @ u - u * values[:3])))
            orthogonality = float(np.max(np.abs(u.T @ u - np.eye(3))))
            require(residual < 1e-9 and orthogonality < 1e-9, 'eigenpair/orthogonality')
            close(fit['max_residual'], residual, 'residual recorded', atol=1e-14)
            close(fit['max_orthogonality'], orthogonality, 'orthogonality recorded', atol=1e-14)
            close(fit['eigengap'], values[2] - values[1], 'gap')
            checked['eigenvectors'] += 3
            if fit['status'] != 'ok':
                continue
            close(fit['embedding'], normalize_rows(u[:, :2]), 'row normalization')
            reproduction = kmeans(fit['embedding'])
            equivalent(fit['fit'], reproduction, 'embedding restart replay')
            checked['restarts'] += len(reproduction['restarts'])
            for key, value in cut_score(w, fit['fit']['labels']).items():
                close(fit['cut'][key], value, 'cut')
            for key, value in cut_score(w, entry['truth']).items():
                close(fit['truth_cut'][key], value, 'truth cut')
            close(fit['ari'], ari(entry['truth'], fit['fit']['labels']), 'ari')
    return dict(status='PASS', **checked)


def export(report, folder):
    from pathlib import Path
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    audit(report)
    # One bounded file per dataset. No large file rename transport is needed.
    index = {k: v for k, v in report.items() if k != 'datasets'}
    index['datasets'] = []
    for name, entry in report['datasets'].items():
        body = json.dumps(entry, separators=(',', ':'), allow_nan=False)
        (p / f'{name}.js').write_text(f'globalThis.SPECTRAL_{name}={body};\n')
        index['datasets'].append(name)
    (p / 'data.js').write_text('globalThis.SPECTRAL_META=' + json.dumps(index, separators=(',', ':')) + ';\n')
