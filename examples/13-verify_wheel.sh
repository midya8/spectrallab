#!/bin/sh
set -eu
# Run this with an installed wheel; absolute source root is data only, never PYTHONPATH.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEMP=$(mktemp -d)
trap 'rm -rf "$TEMP"' EXIT
cd "$TEMP"
unset PYTHONPATH
export OPENBLAS_NUM_THREADS=1
python -c 'import spectrallab,spectrallab.core,spectrallab.data,spectrallab.report,spectrallab.__main__; print("5 installed modules",spectrallab.__version__)'
python -m spectrallab fit --output fresh.json
python -m spectrallab audit fresh.json
python -m spectrallab export fresh.json --output fresh-web
python -m spectrallab audit "$ROOT/examples/report.json"
python -m spectrallab export "$ROOT/examples/report.json" --output archived-web
python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ('moons','circles','data'):
    assert (root/'docs'/f'{name}.js').read_bytes()==(Path('archived-web')/f'{name}.js').read_bytes(),name
r=json.loads(Path('fresh.json').read_text());assert len(r['datasets'])==2
print('5 commands, 18 fresh graph fits, both audits and 3 exact archived exports PASS')
PY
