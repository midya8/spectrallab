# SpectralLab

The graph changes the answer. Inspect normalized spectral clustering through every affinity weight, eigenpair row and cut contribution, rather than treating an embedding as ground truth.

[Live graph inspector](https://midya8.github.io/spectrallab/) · [Method and limits](METHOD.md) · [Reproducible archive](examples/report.json)

## What runs

- Manual full RBF, union-kNN and mutual-kNN graphs, with explicit zero self-loops and distance/index tie order.
- Symmetric normalized Laplacian, its two smallest eigenvectors including the trivial mode, unit-row normalization, and manual two-cluster k-means. **NumPy supplies `eigh`/LAPACK**, array arithmetic and random numbers. This is not a handwritten eigensolver or a wrapper around a clustering library.
- All 18 fixed graphs on two original seeded 96-point synthetic fixtures. Twelve uniform distinct-row starts per fit; restart selection uses embedding inertia, never generator labels. Raw-coordinate k-means is retained. There is no selected winning graph.
- Complete degree/component/eigenspectrum/three-eigenvector/embedding/restart/cut witnesses, reconstructed in scalar JavaScript. Pick any graph and point, compare `L u` with `lambda u`, inspect all neighbors, change display-only edge threshold and export real JSON.
- Refusal of isolated vertices, more than two components, a retained-boundary eigengap at most 1e-10, or a zero embedding row. No invented edges or arbitrary nullspace truncation.

## Results that remain visible

These are **designed synthetic transductive illustrations**, not held-out accuracy or real-world validation. The geometry and all graph settings were fixed before running them. Generator labels are used only for post-fit ARI audits. Default union-6 is an explicit UI default, not a learned choice.

| Fixture | Graph | Components | Spectral ARI | Raw k-means ARI |
|---|---|---:|---:|---:|
| Moons | union-3 | 2 | 1.000000 | 0.242035 |
| Moons | mutual-3 | 4 | refused | 0.242035 |
| Moons | RBF sigma 1.0 | 1 | 0.242035 | 0.242035 |
| Rings | union-6 | 2 | 1.000000 | -0.010638 |
| Rings | union-12 | 1 | -0.010195 | -0.010638 |
| Rings | RBF sigma 1.0 | 1 | -0.010638 | -0.010638 |

All 18 rows are in the live inspector and archive. Moons RBF sigma 1.0 has eigengap **0.350982**, much larger than union-3's **0.003156**, but worse generator-label agreement. Eigengap is not a semantic correctness certificate. Normalized cut is comparable as an objective only on the **same graph**.

A fixed original seven-vertex graph makes the rounding limitation concrete: spectral relaxation plus k-means returns normalized cut **0.860832137733142**, while exhaustive rational enumeration of all 63 distinct nonempty binary cuts gives **3450/4181 = 0.8251614446304711**. [Full graph and fitted witness](examples/rounding.json). This tiny optimum does not make the large fits optimal.

## Run offline

Python 3.10+ and NumPy 1.24+. No model API, external dataset, network request, SciPy, sklearn, PyTorch or frontend package is needed at runtime.

```sh
python -m pip install .
python -m spectrallab fit --output /tmp/spectral-report.json
python -m spectrallab audit /tmp/spectral-report.json
python -m spectrallab export /tmp/spectral-report.json --output /tmp/spectral-web-data
python -m unittest discover -s tests -v
node --test tests/*.cjs
```

Open `docs/index.html` directly or serve `docs/` with `python -m http.server --directory docs`. The checked browser dataset scripts are lossless JSON exports of the archive, not refitted in JavaScript. Clipboard failure has a visible fallback; Download JSON emits an actual Blob.

```python
from spectrallab import graph, spectral
weights = graph([[0, 0], [0, 1], [3, 0], [3, 1]], kind="union", neighbors=1)
fit = spectral(weights)
print(fit["status"], fit.get("fit", {}).get("labels"))
```

`graph` accepts finite feature matrices; `spectral` accepts symmetric nonnegative zero-diagonal affinities. Graph clustering is full-data: adding a point changes the graph and can change every assignment. There is **no inductive predict method**.

## Verification

**22 Python tests** include 528 independent Fraction-distance kNN graph references, 558 rational normalized-cut references, analytic cycle spectra/energy identities, degree and component edge cases, row/subspace rotation invariance, a repeated-eigenvalue boundary counterexample, all 18 fresh fits, full archive audits, provenance/metric/eigenvector/label mutations and strict CLI validation.

**Six Node/real-Chrome tests** check all 18 scalar graphs and 4,896 eigenpair-row equations, all 17 fitted cuts and 18 raw cuts, independent BigInt enumeration of 63 tiny-graph cuts, every dataset script, 1,728 browser graph/point states, refusal clearing, physical keyboard threshold changes that do not alter a fit, graph-point selection, clipboard success/failure, actual downloaded Blob JSON, and 320/390/1280 layouts with 44-pixel controls. The browser test requires Node 22+ and Chrome; it uses Chrome's debugging protocol without a browser automation package.

Installed-wheel reproduction: `pip wheel --no-deps . -w dist`, install the wheel, then `sh examples/verify_wheel.sh`. It changes to a clean temporary directory, unsets `PYTHONPATH`, imports all five installed modules, executes five commands, fits all 18 graphs, audits fresh and archived reports, and verifies three archived web scripts byte-for-byte. CI covers Python 3.10-3.13, package reproduction and Chrome. Floating comparisons are finite/tolerance-bound; discrete metadata stays exact. Eigenspace comparisons use projectors, never eigenvector signs.

## Scope

The full-data affinity is a design assumption. Positive floating weights, including tiny RBF values, define connectivity; underflowed zeros are absent edges. Counts do not use a plotting threshold. kNN ties choose the lower original row index, so duplicates may be index-sensitive. Retained eigenvalue boundary ties are refused. K-means may reach a local optimum; restarts do not certify a global optimum. ARI uses supplied generator labels, not independent natural classes. No predictive, density, uncertainty, fairness, robustness, causal or deployment claim follows from these fixtures.

MIT code and original synthetic fixtures. Algorithm attribution and primary references are in [METHOD.md](METHOD.md). Attribution is not endorsement.
