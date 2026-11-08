# Smooth geometry, discrete labels

SpectralLab v0.2 adds a fixed-graph-weight sensitivity laboratory. It does not differentiate point coordinates, nearest-neighbor identities, RBF bandwidths or a cluster-label function. It does not learn a graph. The original v0.1 graph inspector and all its data remain intact.

## Run and inspect

```sh
python -m spectrallab.sensitivity build examples/sensitivity.json
python -m spectrallab.sensitivity audit examples/sensitivity.json
python -m spectrallab.sensitivity export examples/sensitivity.json > docs/sensitivity-data.js
```

Open `docs/sensitivity.html` locally or use the [published inspector](https://midya8.github.io/spectrallab/sensitivity.html). No service, model download, private data or browser eigensolver is used. Python/NumPy produces actual finite refits; scalar JavaScript independently reconstructs degree response, full projector derivative, fixed-cut derivatives, distances and selected matrix rows. Browser controls select archived fits, not new online optimization.

## What changes

For a named undirected edge `(a,b)`, let `E_ab = E_ba = 1`, all other entries zero, and `W(t) = W + t E`. A missing edge permits only `t >= 0` within the nonnegative graph domain. An analytic derivative in the surrounding symmetric-matrix space supplies the one-sided graph-domain derivative. No negative graph is used for a finite refit. At a positive edge, small negative steps are allowed only while weights remain nonnegative.

Let `d = W1`, `s = E1` and `N_ij = W_ij/sqrt(d_i d_j)`. Differentiating every degree term of `L = I-N` gives

```
L'_ij = -E_ij/sqrt(d_i d_j) + 0.5 N_ij (s_i/d_i + s_j/d_j).
```

Freezing the degrees and taking only the first term is wrong. An edge changes both endpoint degrees and therefore every normalized incident edge, even those with `E_ij = 0`. Zero-degree vertices are refused; no inverse convention or added connection is invented.

## What is differentiated

`projector_tangent(A, A_prime, rank=2)` is a general symmetric operator primitive. It uses NumPy `eigh` for the complete eigenbasis in ascending eigenvalue order. If `U` contains the first two eigenvectors, `P = U U^T` is their orthogonal projector. Unlike individual eigenvectors, P is invariant under orthogonal changes of basis within the same subspace. We differentiate P before row normalization, not the normalized-row Gram matrix, Euclidean labels or the k-means algorithm.

For retained index `i < 2` and excluded index `j >= 2`,

```
C_ji = (u_j^T L' u_i)/(lambda_i-lambda_j)
P' = sum_i sum_j C_ji (u_j u_i^T + u_i u_j^T).
```

Only cross-boundary denominators are needed. Internal repeats on either side are harmless to this formula. A repeat straddling the boundary is not: the selected subspace is no longer unique. We refuse an absolute boundary gap at most `1e-10` without replacing its denominator by epsilon. This is a numerical policy, not a theorem that gaps above it are safe.

Differentiated identities give independent numerical witnesses:

- `P P' + P' P = P'`, differentiating idempotence.
- `L P' - P' L = P L' - L' P`, differentiating commutation.
- `trace(P') = 0`, since retained rank stays fixed locally.

The tangent is generally indefinite. `P+tP'` need not be an orthogonal projector. The report stores actual fresh `P(t)` and reports Frobenius norms of `P(t)-P`, `tP'` and their difference. These are **not** operator-norm distances or a Davis-Kahan certificate. No radius or finite-step error guarantee is asserted. All dense full-spectrum work has small-data scope.

## Original controls, all retained

Six base graphs and seven predeclared steps `0, .001, .01, .1, 1, 5, 25` are fixed in source. None is selected using downstream labels.

| Graph | Edited edge | Purpose |
|---|---|---|
| Original seven-vertex weighted rounding graph | Missing 0-5 | Full-weight local/large-step contrast; shares the v0.1 original cut witness |
| Two triangles, bridge weight .04 | 0-3 | A connected graph approaching the disconnected limit |
| Two disconnected triangles | Missing 0-3 | A repeated zero eigenvalue wholly retained, with a valid smooth projector |
| Five-vertex path | Missing 0-4 | Rounding ambiguity, plus an actual finite boundary collision |
| Six-cycle | Missing 0-3 | Repeated eigenvalue across the base retained boundary: refused |
| Two triangles with vertex 5 isolated | Missing 0-5 | No normalized base inverse: refused |

Four valid base cases have 28 finite states, including their four t=0 states. The path at t=1 becomes a five-cycle and its second/third eigenvalues tie; that finite projector and clustering are refused while the base tangent remains displayable. There are 27 valid finite projectors and cluster fits, not 28 successful fits. The two refused base cases have no fabricated finite path.

### Rounding ambiguity is not a uniquely preferred label change

In the committed path fixture, t=.001 changes one archived assignment up to label swap while `||P(t)-P(0)||F < .001`. But two distinct tested restart partitions have objectives equal within `1e-12`, and their least binary64 values differ at roundoff scale. This is a **rounding ambiguity control**, not evidence of a unique optimal partition that changes discontinuously or a stable empirical label-change rate. Fresh eigensolves may choose another valid near-tied partition. The full restart records are retained, so this caveat can be inspected rather than hidden.

The audit checks archived eigenpair equations, subspaces and row normalization, then replays all 12 restarts on the archived embedding. It verifies the recorded least-inertia restart and labels there. It does not require a fresh eigensolver to reproduce arbitrary signed vectors or choose the same near-tied rounding partition. The full-base eigenbasis and cross coefficients may be rotated within repeated blocks; their contracted projector derivative must still agree. A test explicitly rotates the two-dimensional zero eigenspace in an archived witness and requires its audit to pass.

## Fixed partition versus refitted partition

For an original fixed binary label vector, let `c` be its cross-edge weight, `v0,v1` its two volumes, and `c',v0',v1'` their edge responses. Then

```
Ncut' = c' (1/v0+1/v1) - c (v0'/v0^2+v1'/v1^2).
```

This is the derivative of that supplied partition's score. The actual finite fit reruns spectral row normalization and 12-start k-means. Its label choice is not differentiated. Comparing fixed and refitted cuts on a single finite graph is meaningful; treating cut values on different W(t) as a quality ranking is not. Cluster-label disagreement uses the smaller Hamming count after swapping both binary label names, not semantic truth. No labels choose a graph or edge.

## Validation

The full project has 33 Python tests and nine Node/real-Chrome tests. The v0.2 additions include:

- 83 positive weighted graph/edge central differences for the full normalized-Laplacian derivative; every case rejects the frozen-degree shortcut.
- Six random symmetric projector central differences, differentiated commutator/idempotence/trace checks, an analytic rank-one mixing result and an exact repeated-eigenvalue block control.
- 42 independent Fraction quotient-derivative references across every binary partition of a four-vertex integer graph and every edge.
- One-sided zero-edge Taylor error ratios near four when the step is halved, internal-basis conjugation, boundary-tie/isolated/shape/rank/nonfinite/negative-step refusals, no input mutation and archive mutation rejection.
- Strict duplicate/nonfinite/malformed JSON CLI errors with no partial stdout, fresh and archived audits, byte-exact archived browser export and installed-wheel execution outside checkout.
- Scalar JavaScript reconstruction of 146 full-base eigenpair equations and 146 matrix entries in each projector identity; 1,022 selected numeric row cells over all 28 states, including the finite refusal. Independent actual-state projectors use archived finite eigenvectors, not the tangent.
- Real Chrome checks 168 actual case/step/row states, cleared matrices and rows after base refusals, the finite tie's blank actual heatmap and retained valid base tangent, physical keyboard selection, both clipboard outcomes, an actual Blob download, and 320/390/1280 layouts with 44-pixel controls.

No single residual is a formal proof. Tolerances are finite numerical checks. The general API accepts finite symmetric matrices of size at least three; affinities must be symmetric, nonnegative and zero diagonal. Supplied matrices are not silently symmetrized. `rank` is an integer between one and n-1. A response path has 1-128 finite scalar steps of magnitude at most 1e6 and refuses negative resulting weights.

## Sources and distinction from other stability results

- [Anil Damle, Cornell CS4220, subspace perturbation notes](https://www.cs.cornell.edu/courses/cs4220/2024sp/Perturbation.pdf): orthogonal-projector subspace geometry and gap-dependent bounds. Those notes use the operator norm; our displayed distances explicitly use Frobenius.
- [Muhammad F. Kasim, derivatives of partial eigendecomposition](https://ar5iv.labs.arxiv.org/html/2011.04366): separated subspaces, degeneracy and eigenderivatives. This project implements the simpler full symmetric cross-boundary projector contraction, not that paper's partial/generalized solver.
- [Ulrike von Luxburg, spectral clustering tutorial](https://pure.mpg.de/rest/items/item_1790853_2/component/file_3009830/content): symmetric versus random-walk Laplacians, retained subspaces and the distinction between relaxation and rounded labels.
- [NumPy `eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html): ascending eigenvalues, eigenvectors in columns and LAPACK `_syevd` attribution. NumPy is a dependency, not a handwritten eigensolver claim.
- [PyTorch `eigh` documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.md): individual-eigenvector gradient nonuniqueness and small-gap warnings. No PyTorch or automatic differentiation is used. Those warnings are not a reason to reject a completely retained repeated block's projector derivative.
- [Julia community discussion](https://discourse.julialang.org/t/derivative-of-eigenvalues-and-eigenvectors-of-hermitian-matrix-by-automatic-differentiation/11563): practical distinction between individual eigenvectors, basis-invariant functions and cancellation. Community explanation is supporting context, not a proof.
- [Average sensitivity of spectral clustering](https://eprints.whiterose.ac.uk/id/eprint/161662/1/sensitivityclustering.pdf): random edge-removal sensitivity under stated assumptions and rounding procedures. It does not certify this deterministic additive weighted path or its row-normalized k-means labels; none of its guarantees are claimed here.

Independent implementation of standard matrix calculus. No paper figure, source code or external dataset is reproduced. Original graphs and browser code use this repository's MIT license.
