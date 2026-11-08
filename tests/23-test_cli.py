import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from spectrallab.report import audit

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'spectrallab', *map(str,args)],cwd=ROOT,
                              env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)

    def test_fit_audit_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);report=p/'report.json'
            r=self.run_cli('fit','--output',report);self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(r.stdout)['graphs'],18)
            r=self.run_cli('audit',report);self.assertEqual(r.returncode,0,r.stderr)
            r=self.run_cli('export',report,'--output',p/'web');self.assertEqual(r.returncode,0,r.stderr)
            for name in ('moons','circles'):
                text=(p/'web'/f'{name}.js').read_text();data=json.loads(text.split('=',1)[1].rstrip(';\n'))
                self.assertEqual(data,json.loads(report.read_text())['datasets'][name])
            self.assertEqual(sorted(x.name for x in (p/'web').iterdir()),['circles.js','data.js','moons.js'])

    def test_rejects_malformed_duplicate_nonfinite_and_tamper(self):
        report=json.loads((ROOT/'examples/report.json').read_text())
        report['datasets']['moons']['fits'][0]['embedding'][0][0]+=1
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.json'
            for text in ('{','{"schema":1,"schema":2}','{"value":NaN}',json.dumps(report)):
                p.write_text(text)
                r=self.run_cli('audit',p)
                self.assertEqual(r.returncode,2,r.stderr)
                self.assertIn('Invalid report',r.stderr)

    def test_module_api_imports_and_real_graph(self):
        import spectrallab
        self.assertEqual(spectrallab.__version__,'0.1.0')
        self.assertEqual(len(spectrallab.graph([[0],[1],[2]],neighbors=1)),3)
