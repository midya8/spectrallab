import copy
import json
import math
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from spectrallab.core import laplacian, cut_score
from spectrallab.sensitivity import (laplacian_tangent, projector_tangent, fixed_cut_tangent,
                                    edge_direction, fixtures, response, build, audit)

ROOT = Path(__file__).resolve().parents[1]

class SensitivityTests(unittest.TestCase):
    def test_laplacian_all_edge_central_differences_and_degree_terms(self):
        rng = np.random.default_rng(782)
        count = 0
        for n in range(3, 9):
            w = rng.uniform(.3, 3, (n,n)); w = w+w.T; np.fill_diagonal(w,0)
            for i in range(n):
                for j in range(i):
                    e = edge_direction(n,i,j); dl = laplacian_tangent(w,e); h=1e-5
                    fd = (laplacian(w+h*e)-laplacian(w-h*e))/(2*h)
                    np.testing.assert_allclose(dl,fd,atol=3e-11,rtol=3e-8)
                    # Holding degrees fixed is the wrong derivative.
                    frozen = -e/np.sqrt(w.sum(1)[:,None]*w.sum(1)[None,:])
                    self.assertGreater(np.linalg.norm(dl-frozen), .001)
                    self.assertTrue(np.all(np.diag(dl)==0)); count += 1
        self.assertEqual(count,83)

    def test_projector_central_differences_and_identities(self):
        rng=np.random.default_rng(300)
        for n in range(3,9):
            q,_=np.linalg.qr(rng.normal(size=(n,n)))
            a=q@np.diag(np.arange(n,dtype=float))@q.T; a=(a+a.T)/2
            e=rng.normal(size=(n,n)); e=(e+e.T)/2
            r=projector_tangent(a,e); p=np.array(r['projector']); dp=np.array(r['projector_tangent']);h=1e-5
            def project(b):
                _,u=np.linalg.eigh(b);return u[:,:2]@u[:,:2].T
            np.testing.assert_allclose(dp,(project(a+h*e)-project(a-h*e))/(2*h),atol=2e-9,rtol=2e-7)
            np.testing.assert_allclose(p@dp+dp@p,dp,atol=2e-14)
            np.testing.assert_allclose(a@dp-dp@a,p@e-e@p,atol=2e-14)
            self.assertLess(abs(np.trace(dp)),2e-14)
            self.assertLess(r['commutator_residual'],2e-14)

    def test_repeated_internal_eigenvalues_basis_invariant(self):
        # Exact repeated eigenvalues on BOTH sides, separated retained boundary.
        a=np.diag([0.,0.,2.,2.]);e=np.array([[0.,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]])
        r=projector_tangent(a,e);dp=np.array(r['projector_tangent'])
        expected=np.zeros((4,4));expected[:2,2:]=-e[:2,2:]/2;expected[2:,:2]=expected[:2,2:].T
        np.testing.assert_allclose(dp,expected,atol=1e-15)
        # Conjugating by arbitrary rotations within repeated blocks leaves the
        # physical derivative unchanged after converting back to original axes.
        for angle in (.2,.7,1.3):
            c,s=math.cos(angle),math.sin(angle)
            t=np.array([[c,-s,0,0],[s,c,0,0],[0,0,s,c],[0,0,-c,s]])
            rotated=t.T@e@t; rotated=(rotated+rotated.T)/2
            changed=projector_tangent(a,rotated)
            np.testing.assert_allclose(t@np.array(changed['projector_tangent'])@t.T,dp,atol=6e-15)
        # Explicit off-block denominators: no division by the internal zero gap.
        self.assertTrue(np.isfinite(r['cross_coefficients']).all())

    def test_repeated_boundary_refused_not_regularized(self):
        for a in (np.diag([0.,1.,1.,2.]),np.diag([0.,1.,1.+1e-12,2.])):
            with self.assertRaisesRegex(ValueError,'boundary'):projector_tangent(a,np.eye(4))
        with self.assertRaisesRegex(ValueError,'isolated'):laplacian_tangent(np.zeros((3,3)),edge_direction(3,0,1))

    def test_fraction_fixed_cut_derivative_all_partitions_all_edges(self):
        # Independent exact rational quotient derivative for all 7 partitions of K4.
        w=[[0,2,3,1],[2,0,4,2],[3,4,0,5],[1,2,5,0]];count=0
        for mask in range(1,8):
            labels=[0]+[(mask>>(i-1))&1 for i in range(1,4)]
            groups=[[i for i in range(4) if labels[i]==c] for c in (0,1)]
            v=[sum(F(w[i][j]) for i in g for j in range(4)) for g in groups]
            cut=sum(F(w[i][j]) for i in groups[0] for j in groups[1])
            for i in range(4):
                for j in range(i):
                    dc=F(labels[i]!=labels[j]);dv=[F((labels[i]==c)+(labels[j]==c)) for c in (0,1)]
                    oracle=dc*(1/v[0]+1/v[1])-cut*(dv[0]/v[0]**2+dv[1]/v[1]**2)
                    got=fixed_cut_tangent(w,edge_direction(4,i,j),labels)
                    self.assertAlmostEqual(got['normalized_cut_derivative'],float(oracle),places=15);count+=1
        self.assertEqual(count,42)

    def test_zero_edge_one_sided_and_finite_error_rate(self):
        _,_,w,edge=fixtures()[2];e=edge_direction(6,*edge);dl=laplacian_tangent(w,e)
        r=projector_tangent(laplacian(w),dl);p=np.array(r['projector']);dp=np.array(r['projector_tangent'])
        errors=[]
        for h in (.004,.002,.001):
            p1=np.array(projector_tangent(laplacian(np.array(w)+h*e),np.zeros((6,6)))['projector'])
            errors.append(np.linalg.norm(p1-p-h*dp))
        self.assertTrue(3.95<errors[0]/errors[1]<4.05)
        self.assertTrue(3.95<errors[1]/errors[2]<4.05)
        with self.assertRaisesRegex(ValueError,'negative'):response(w,edge,[-.01])

    def test_scalar_analytic_rank_one(self):
        # Rank-one bottom projector, analytic 2x2 mixing embedded in 3 dimensions.
        a=np.diag([0.,2.,5.]);e=np.array([[0.,3.,0.],[3.,0.,0.],[0.,0.,0.]])
        r=projector_tangent(a,e,rank=1)
        np.testing.assert_allclose(r['projector_tangent'],[[0,-1.5,0],[-1.5,0,0],[0,0,0]],atol=0)
        np.testing.assert_allclose(projector_tangent(a,np.eye(3),rank=1)['projector_tangent'],0,atol=0)

    def test_archived_all_states_and_mutations(self):
        archived=json.loads((ROOT/'examples/sensitivity.json').read_text())
        self.assertEqual(audit(archived),{'cases':6,'smooth_cases':4,'refusals':2,'finite_states':28})
        self.assertEqual(len(archived['cases'][0]['states']),7)
        # Local differentiability is not a label certificate, even at a tiny step.
        path=archived['cases'][3]
        self.assertEqual(path['states'][1]['changed_labels_up_to_swap'],1)
        self.assertLess(path['states'][1]['projector_distance'],.001)
        self.assertEqual(path['states'][4]['fit']['status'],'refused-boundary-tie')
        self.assertIsNone(path['states'][4]['projector'])
        for field in ('projector_tangent','laplacian_tangent'):
            bad=copy.deepcopy(archived)
            target=bad['cases'][0]['witness'] if field=='projector_tangent' else bad['cases'][0]
            target[field][0][0]+=.01
            with self.assertRaises(ValueError):audit(bad)
        bad=copy.deepcopy(archived);bad['cases'][0]['states'][1]['fit']['fit']['labels'][0]^=1
        with self.assertRaises(ValueError):audit(bad)
        bad=copy.deepcopy(archived);bad['cases'][-1]['reason']='invented'
        with self.assertRaises(ValueError):audit(bad)
        bad=copy.deepcopy(archived);bad['cases'][0]['edge']=[1,3]
        with self.assertRaises(ValueError):audit(bad)

    def test_archive_allows_internal_basis_rotation_not_signed_identity(self):
        r=json.loads((ROOT/'examples/sensitivity.json').read_text())
        c=r['cases'][2];q=np.array(c['witness']['eigenvectors'])
        angle=.34;t=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        q[:,:2]=q[:,:2]@t
        c['witness']['eigenvectors']=q.tolist()
        values=np.array(c['witness']['eigenvalues']);dl=np.array(c['laplacian_tangent'])
        c['witness']['cross_coefficients']=((q[:,2:].T@dl@q[:,:2])/(values[:2][None,:]-values[2:][:,None])).tolist()
        self.assertEqual(audit(r)['smooth_cases'],4)

    def test_validation_and_no_mutation(self):
        w=np.array(fixtures()[0][2],dtype=float);original=w.copy();e=edge_direction(7,0,5)
        laplacian_tangent(w,e);np.testing.assert_array_equal(w,original)
        for rank in (True,0,7,1.5):
            with self.assertRaises(ValueError):projector_tangent(laplacian(w),e,rank)
        for edge in ((0,0),(-1,0),(0,True),(0,9)):
            with self.assertRaises(ValueError):edge_direction(7,*edge)
        for steps in ([],[float('nan')],[True],[1e9]):
            with self.assertRaises(ValueError):response(w,(0,5),steps)
        for bad in ([[1,2],[3,4]],np.full((3,3),np.nan)):
            with self.assertRaises(ValueError):projector_tangent(bad,bad)
        bad=e.copy();bad[0,0]=1
        with self.assertRaises(ValueError):laplacian_tangent(w,bad)

    def test_cli_strict_and_exact_export(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'fresh.json'
            for args in (['build',str(p)],['audit',str(p)]):
                run=subprocess.run([sys.executable,'-m','spectrallab.sensitivity',*args],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
            run=subprocess.run([sys.executable,'-m','spectrallab.sensitivity','export',str(ROOT/'examples/sensitivity.json')],cwd=ROOT,capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(run.stdout,(ROOT/'docs/sensitivity-data.js').read_bytes())
            for text in ('{"schema":1,"schema":2}','{"x":NaN}','broken'):
                p.write_text(text)
                run=subprocess.run([sys.executable,'-m','spectrallab.sensitivity','audit',str(p)],cwd=ROOT,capture_output=True)
                self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,b'')
