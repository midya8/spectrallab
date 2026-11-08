"""Original small seeded synthetic fixtures, not evidence of real-world performance."""
import math
import numpy as np

CONFIGS = [{'id': f'{kind}-{k}', 'kind': kind, 'neighbors': k}
           for kind in ('union', 'mutual') for k in (3, 6, 12)] + [
           {'id': f'rbf-{s}', 'kind': 'rbf', 'sigma': s} for s in (0.08, 0.25, 1.0)]


def datasets(seed=20261108):
    rng = np.random.default_rng(seed)
    count = 48
    t = np.linspace(0.02, math.pi - 0.02, count)
    moons = np.vstack((np.column_stack((np.cos(t), np.sin(t))),
                       np.column_stack((1 - np.cos(t), 0.45 - np.sin(t)))))
    moons += rng.normal(0, 0.025, moons.shape)
    t = np.arange(count) * 2 * math.pi / count
    circles = np.vstack([np.column_stack((r * np.cos(t), r * np.sin(t))) for r in (0.45, 1.0)])
    circles += rng.normal(0, 0.018, circles.shape)
    return {name: {'points': x.tolist(), 'truth': [0] * count + [1] * count,
                   'seed': seed, 'description': description} for name, x, description in (
        ('moons', moons, 'Two original noisy interleaving arcs; 96 points. Transductive illustration only.'),
        ('circles', circles, 'Two original noisy concentric rings; 96 points. Transductive illustration only.'))}
