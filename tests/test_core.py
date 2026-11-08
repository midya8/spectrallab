import copy
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import unittest
import numpy as np
from spectrallab.core import *
from spectrallab.report import build, audit, equivalent
from spectrallab.data import datasets, CONFIGS


class GraphTests(unittest.TestCase):
    def test_knn_exhaustive_scalar_reference(self):
        # Independent scalar distances and rank membership, 48 fixtures, all k and both symmetries.
        rng = np.random.default_rng(819)
        count = 0
        for n in range(3, 11):
            for repeat in range(6):
                x = rng.integers(-4, 5, size=(n, 2)).tolist()
                ranks = []
                for i in range(n):
                    ranks.append(sorted([j for j in range(n) if j != i],
                                        key=lambda j: (sum((F(x[i][d]) - F(x[j][d])) ** 2 for d in range(2)), j)))
                for k in range(1, n):
                    for kind in ('union', 'mutual'):
                        expected = [[int(i != j and ((j in ranks[i][:k] or i in ranks[j][:k]) if kind == 'union'
                                     else (j in ranks[i][:k] and i in ranks[j][:k]))) for j in range(n)] for i in range(n)]
                        np.testing.assert_array_equal(graph(x, kind, k), expected)
                        count += 1
        self.assertEqual(count, 528)

    def test_rbf_scalar_reference(self):
        x = [[0, 0], [0, 0], [3, 4], [-1, 2]]
        for sigma in (.1, 1, 5):
            expected = [[0 if i == j else math.exp(-sum((a-b)**2 for a,b in zip(x[i], x[j]))/(2*sigma*sigma))
                         for j in range(4)] for i in range(4)]
            np.testing.assert_allclose(graph(x, 'rbf', sigma=sigma), expected, atol=1e-15)

    def test_validation(self):
        for value in ([[0]], [[0, 1], [0, 0]], [[1, 1], [1, 0]], [[0, -1], [-1, 0]], [[0, math.nan], [math.nan, 0]]):
            with self.assertRaises(ValueError): matrix(value)
        for kind, k, sigma in [('other', 1, 1), ('union', True, 1), ('union', 3, 1), ('rbf', 1, 0), ('rbf', 1, math.inf)]:
            with self.assertRaises(ValueError): graph([[0],[1],[2]], kind, k, sigma)
        for x in ([[0]], [[0],[1]], [[0],[1],[math.inf]]):
            with self.assertRaises(ValueError): graph(x)

        with self.assertRaises(ValueError): spectral([[0,1],[1,0]])
        for value in (0, -1, True, 2.5):
            with self.assertRaises(ValueError): kmeans([[0],[1],[2]], max_iter=value)

    def test_graph_components_refusal(self):
        isolated = np.zeros((6,6)); isolated[0,1] = isolated[1,0] = 1
        self.assertEqual(spectral(isolated)['status'], 'refused-isolated')
        w = np.zeros((6,6))
        for i in range(0,6,2): w[i,i+1] = w[i+1,i] = 1
        self.assertEqual(spectral(w)['status'], 'refused-components')
        self.assertEqual(components(w), [[0,1],[2,3],[4,5]])
        with self.assertRaises(ValueError): laplacian(isolated)

    def test_tied_boundary_not_silently_picked(self):
        complete = np.ones((6,6)) - np.eye(6)
        fit = spectral(complete)
        self.assertEqual(fit['status'], 'refused-boundary-tie')
        self.assertNotIn('fit', fit)

    def test_analytic_cycle_spectrum_and_energy(self):
        for n in (5, 6, 7, 8, 12):
            w = np.zeros((n,n))
            for i in range(n): w[i,(i+1)%n] = w[(i+1)%n,i] = 1
            lap = laplacian(w)
            expected = sorted(1-math.cos(2*math.pi*j/n) for j in range(n))
            np.testing.assert_allclose(np.linalg.eigvalsh(lap), expected, atol=1e-14)
            v = np.arange(n, dtype=float) / 3
            energy = sum(w[i,j] * (v[i]/math.sqrt(2)-v[j]/math.sqrt(2))**2 for i in range(n) for j in range(i+1,n))
            self.assertAlmostEqual(float(v@lap@v), energy)
            np.testing.assert_allclose(lap @ np.sqrt(w.sum(axis=1)), 0, atol=1e-14)

    def test_cut_fraction_exhaustive_oracle(self):
        # Every nonempty binary cut of 18 original rational weighted graphs, not a solver reference.
        rng = np.random.default_rng(5)
        total = 0
        for _ in range(18):
            n = 6
            w = [[F(0) for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(i+1,n): w[i][j] = w[j][i] = F(int(rng.integers(1,9)), 8)
            for bits in itertools.product((0,1), repeat=n-1):
                y = [0, *bits]
                if sum(y) == 0: continue
                cut = sum(w[i][j] for i in range(n) for j in range(i+1,n) if y[i] != y[j])
                vols = [sum(w[i][j] for i in range(n) if y[i] == c for j in range(n)) for c in (0,1)]
                expected = cut * (1/vols[0] + 1/vols[1])
                self.assertAlmostEqual(cut_score(w, y)['normalized_cut'], float(expected), places=13)
                total += 1
        self.assertEqual(total, 558)

    def test_relaxation_is_not_exact_discrete_optimizer(self):
        # The archived original graph and exhaustive rational cuts, not a repeated search.
        example = json.loads((Path(__file__).resolve().parents[1]/'examples/rounding.json').read_text())
        w = example['weights']; n = len(w)
        records = []
        for bits in itertools.product((0,1), repeat=n-1):
            if not sum(bits): continue
            y = [0,*bits]
            cut = sum(w[i][j] for i in range(n) for j in range(i+1,n) if y[i]!=y[j])
            volumes = [sum(w[i][j] for i in range(n) if y[i]==c for j in range(n)) for c in (0,1)]
            records.append((F(cut,volumes[0])+F(cut,volumes[1]), y))
        exact, labels = min(records)
        self.assertEqual(len(records), 63)
        self.assertEqual(str(exact), example['exact']['normalized_cut'])
        self.assertEqual(labels, example['exact']['labels'])
        fit = spectral(w)
        self.assertEqual(fit['fit']['labels'], example['spectral']['fit']['labels'])
        self.assertGreater(fit['cut']['normalized_cut'], float(exact)+1e-6)

    def test_row_and_subspace_rotation_invariance(self):
        rng = np.random.default_rng(1)
        u = rng.normal(size=(20,2)); q = np.array([[.6,-.8],[.8,.6]])
        a, b = normalize_rows(u), normalize_rows(u@q)
        np.testing.assert_allclose(a@a.T, b@b.T, atol=1e-14)
        fa, fb = kmeans(a), kmeans(b)
        self.assertEqual(fa['labels'], fb['labels'])
        np.testing.assert_allclose(fa['inertia'], fb['inertia'])
        with self.assertRaises(ValueError): normalize_rows([[0,0],[1,0]])

    def test_repeated_boundary_subspace_can_change(self):
        # Complete graph has a repeated nontrivial eigenvalue: distinct valid choices change geometry.
        n = 6; w = np.ones((n,n))-np.eye(n); lap = laplacian(w)
        constant = np.ones(n)/math.sqrt(n)
        a = np.array([1,-1,0,0,0,0])/math.sqrt(2)
        b = np.array([0,0,1,-1,0,0])/math.sqrt(2)
        for v in (a,b): np.testing.assert_allclose(lap@v, n/(n-1)*v, atol=1e-14)
        ua, ub = (np.column_stack((constant,v)) for v in (a,b))
        self.assertGreater(float(np.max(np.abs(ua@ua.T-ub@ub.T))), .4)
        self.assertEqual(spectral(w)['status'], 'refused-boundary-tie')

    def test_scale_and_permutation_covariance(self):
        x = np.array(datasets()['moons']['points']); p = np.arange(len(x))[::-1]
        w = graph(x, 'rbf', sigma=.25)
        np.testing.assert_allclose(laplacian(w), laplacian(15*w), atol=1e-14)
        np.testing.assert_allclose(graph(x*7, 'rbf', sigma=.25*7), w, atol=1e-14)
        np.testing.assert_allclose(laplacian(w[np.ix_(p,p)]), laplacian(w)[np.ix_(p,p)], atol=1e-14)

    def test_kmeans_histories_are_nonincreasing(self):
        for entry in datasets().values():
            fit = kmeans(entry['points'])
            for r in fit['restarts']:
                self.assertTrue(all(b <= a + 1e-12 for a,b in zip(r['history'], r['history'][1:])))
            self.assertEqual(fit['inertia'], min(r['inertia'] for r in fit['restarts'] if r['status']=='ok'))
            x = np.array(entry['points']); y = np.array(fit['labels'])
            centers = np.array([x[y==c].mean(axis=0) for c in (0,1)])
            self.assertEqual(canonical_labels(np.argmin(((x[:,None]-centers)**2).sum(axis=2),axis=1)), y.tolist())

    def test_ari_pair_count_independent(self):
        # Independent explicit same-pair contingency for balanced labels.
        truth = [0,0,0,1,1,1]; labels = [0,0,1,0,1,1]
        pairs = list(itertools.combinations(range(6),2))
        same_a = sum(truth[i]==truth[j] for i,j in pairs)
        same_b = sum(labels[i]==labels[j] for i,j in pairs)
        both = sum(truth[i]==truth[j] and labels[i]==labels[j] for i,j in pairs)
        expected = (F(both)-F(same_a*same_b,len(pairs)))/(F(same_a+same_b,2)-F(same_a*same_b,len(pairs)))
        self.assertAlmostEqual(ari(truth, labels), float(expected))
        self.assertEqual(ari(truth, [1-u for u in truth]), 1)
        self.assertAlmostEqual(ari(truth, labels), ari(labels, truth))


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1]/'examples/report.json').read_text())

    def test_full_archive_audit(self):
        self.assertEqual(audit(self.report), {'status':'PASS','graphs':18,'eigenvectors':51,'restarts':204,'points':192})

    def test_fresh_18_graphs_same_metrics_not_eigen_signs(self):
        fresh = build()
        for name, d in self.report['datasets'].items():
            for stored, new in zip(d['fits'], fresh['datasets'][name]['fits']):
                self.assertEqual(stored['status'], new['status'])
                if stored['status'] == 'ok':
                    self.assertEqual(stored['fit']['labels'], new['fit']['labels'])
                    self.assertAlmostEqual(stored['ari'], new['ari'])
                    self.assertAlmostEqual(stored['cut']['normalized_cut'], new['cut']['normalized_cut'])
                    u,v = np.array(stored['eigenvectors'])[:,:2],np.array(new['eigenvectors'])[:,:2]
                    np.testing.assert_allclose(u@u.T,v@v.T,atol=1e-9)

    def test_truth_not_used_for_fits(self):
        # Fit API accepts points/weights only, so arbitrary synthetic truth cannot change fitted outputs.
        for entry in self.report['datasets'].values():
            for fit, config in zip(entry['fits'], CONFIGS):
                new = spectral(graph(entry['points'], **{k:v for k,v in config.items() if k!='id'}))
                if fit['status']=='ok': self.assertEqual(new['fit']['labels'], fit['fit']['labels'])
            changed = [int(i%3==0) for i in range(96)]
            self.assertNotEqual(ari(changed, entry['raw']['labels']), entry['raw']['ari'])

    def test_archive_mutations_rejected(self):
        mutations = [
            lambda r: r['datasets']['moons']['points'][0].__setitem__(0,99),
            lambda r: r['datasets']['moons']['truth'].__setitem__(0,1),
            lambda r: r['configurations'][0].__setitem__('neighbors',4),
            lambda r: r['datasets']['moons']['fits'][0]['degrees'].__setitem__(0,99),
            lambda r: r['datasets']['moons']['fits'][0]['eigenvectors'][0].__setitem__(0,.9),
            lambda r: r['datasets']['moons']['fits'][0].__setitem__('ari',.5),
            lambda r: r['datasets']['moons']['fits'][0]['fit']['labels'].__setitem__(0,1),
            lambda r: r['datasets']['moons']['fits'][0]['cut'].__setitem__('normalized_cut',5),
            lambda r: r['datasets']['moons']['fits'][3].__setitem__('status','ok'),
            lambda r: r['datasets']['moons']['fits'][0].__setitem__('graph_sha256','a'*64),
        ]
        for mutation in mutations:
            r = copy.deepcopy(self.report); mutation(r)
            with self.assertRaises((ValueError,KeyError)): audit(r)

    def test_large_gap_is_not_truth_certificate(self):
        moon = self.report['datasets']['moons']['fits']
        small = next(f for f in moon if f['id']=='union-3')
        large = next(f for f in moon if f['id']=='rbf-1.0')
        self.assertGreater(large['eigengap'], small['eigengap'])
        self.assertLess(large['ari'], small['ari'])

    def test_tolerance_is_finite_and_discrete_strict(self):
        equivalent({'x':.2,'id':1}, {'x':.2+1e-12,'id':1})
        for bad in ({'x':math.nan,'id':1},{'x':.2,'id':1.0},{'x':.3,'id':1}):
            with self.assertRaises(ValueError): equivalent(bad, {'x':.2,'id':1})
